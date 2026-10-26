import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

# MuJoCo selects its GL backend at import time, before scene creation.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if os.environ.get("GR3_RENDER_DEVICE", "cpu") == "cpu" and mesa.is_file():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(mesa))
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import yaml

from .geometry import RGBDFrame, rotation_error, transform
from .grasps import GraspGen, geometric_grasps, fdh6_grasps
from .model import ROOT, build_scene
from .perception import GroundingDINO, segment_tabletop
from .planning import make_planner


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def perceive(frame, config, args, output, sim=None):
    box = None
    if args.target_text:
        box = GroundingDINO(device=args.detector_device).detect(frame.rgb, args.target_text)
    objects = segment_tabletop(frame, config["perception"], box=box)
    if not objects:
        raise RuntimeError("No RGB-D foreground objects in workspace; inspect camera.png and camera calibration")
    if not 0 <= args.object_index < len(objects):
        raise ValueError(f"object-index must be between 0 and {len(objects) - 1}")
    annotated = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(annotated)
    for index, cloud in enumerate(objects):
        x1, y1, x2, y2 = cloud.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="lime" if index == args.object_index else "white", width=2)
        draw.text((x1, max(0, y1 - 14)), f"object {index}", fill="white")
    annotated.save(output / "detections.png")
    target = objects[args.object_index]
    Image.fromarray((target.mask * 255).astype(np.uint8)).save(output / "target_mask.png")
    scene_points, scan = None, None
    if config.get("head_scan", {}).get("enabled", False):
        if sim is None:
            raise ValueError("Head scanning requires a live simulation; infer-frame accepts one saved RGB-D frame")
        from .head_scan import capture_head_scan
        scene_points, target, scan = capture_head_scan(sim, frame, target, config, output)
    np.savez_compressed(output / "target_cloud.npz", points_camera=target.points_camera,
                        points_world=target.points_world, rgb=target.rgb,
                        T_world_camera=frame.T_world_camera)
    metadata = None
    if args.backend == "graspgen":
        backend = GraspGen(config["graspgen"])
        try:
            proposals = backend.infer(target.points_camera, frame.T_world_camera)
            metadata = backend.metadata
        finally:
            backend.close()
    elif args.backend == "m2t2":
        from .m2t2_backend import M2T2Backend, retarget_fdh6
        backend = M2T2Backend(config.get("m2t2", {}))
        try:
            proposals = backend.infer(frame, target, config["scene"], config["perception"], output,
                                      scene_points_world=scene_points,
                                      target_context_margin_m=scan["settings"]["model_context_margin_m"] if scan else None)
        finally:
            write_json(output / "m2t2_metadata.json", backend.metadata)
        write_json(output / "m2t2_source_grasps.json", [asdict(g) for g in proposals])
        metadata = backend.metadata
        write_json(output / "m2t2_metadata.json", metadata)
        if config["robot"]["hand"] == "fdh6":
            side = args.arm if args.arm != "auto" else ("left" if target.centroid_world[1] > 0 else "right")
            if config.get("m2t2", {}).get("fdh6_offset_profile_side", side) != side:
                raise ValueError("The loaded FDH-6 offset profile was calibrated for the right hand")
            offset = config.get("m2t2", {}).get("fdh6_contact_offset_m", [-0.012, 0, -0.045])
            proposals = retarget_fdh6(proposals, side, offset)
            metadata.update(fdh6_contact_offset_m=offset, fdh6_axis_transfer="source Z -> hand approach X; source X -> hand closing Z")
            write_json(output / "m2t2_metadata.json", metadata)
    else:
        side = args.arm if args.arm != "auto" else ("left" if target.centroid_world[1] > 0 else "right")
        proposals = (fdh6_grasps(target, config["scene"]["table_top_m"], side)
                     if config["robot"]["hand"] == "fdh6" else geometric_grasps(target))
    if getattr(args, "candidate_index", None) is not None:
        index = args.candidate_index
        if not 0 <= index < len(proposals):
            raise ValueError(f"candidate-index must be in [0, {len(proposals) - 1}]")
        proposals = [proposals[index]]
    write_json(output / "grasps.json", [asdict(grasp) for grasp in proposals])
    return proposals, {"foreground_objects": len(objects), "selected_object": args.object_index,
                       "selected_point_count": len(target.points_world),
                       "selected_centroid_world_m": target.centroid_world,
                       "centroids_world_m": [obj.centroid_world for obj in objects],
                       "detector": "grounding-dino+depth_foreground" if box is not None else "depth_foreground",
                       "grasp_backend": args.backend, "requested_candidate_index": getattr(args, "candidate_index", None),
                       "grasp_model_metadata": metadata, "head_scan": scan}


def simulate(config, args, output):
    from .simulation import Simulation

    scene = build_scene(config, output / "model")
    with Simulation(scene, config) as sim:
        sim.settle()
        frame = sim.capture()
        frame.save(output / "frame.npz")
        Image.fromarray(frame.rgb).save(output / "camera.png")
        Image.fromarray(sim.overview()).save(output / "overview.png")
        proposals, report = perceive(frame, config, args, output, sim=sim)
        if args.backend == "m2t2":
            source = json.loads((output / "m2t2_source_grasps.json").read_text())
            Image.fromarray(sim.grasp_candidates_view([item["T_world_tcp"] for item in source])).save(output / "m2t2_grasps.png")
        report.update(stage="perception_complete", grasp_success=False,
                      grasp_executed=False, hand=config["robot"]["hand"], fixed_base=True,
                      limitation=("FDH-6 tabletop contact baseline; fixed base and assumed calibration."
                                  if config["robot"]["hand"] == "fdh6" else
                                  "Fixed dummy hand. Only pregrasp reaching is implemented; no contact grasp or lift."))
        write_json(output / "report.json", report)
        side = args.arm
        if side == "auto":
            side = "left" if report["selected_centroid_world_m"][1] > 0 else "right"
        report["arm"] = side
        if config["robot"]["hand"] == "fdh6":
            from .grasping import run_grasp
            context = (nullcontext(None) if args.plan_only or args.no_video else
                       imageio.get_writer(output / "grasp.mp4", fps=15, macro_block_size=1))
            with context as video:
                report = run_grasp(sim, proposals, report, args, output, write_json, video)
            Image.fromarray(sim.overview()).save(output / "final_overview.png")
            Image.fromarray(sim.capture().rgb).save(output / "final_camera.png")
            if not args.plan_only:
                Image.fromarray(sim.grasp_view(sim.model.body(report["evaluation_object"]).id, report["stage"])).save(output / "final_grasp.png")
            print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))
            if not args.plan_only and not report["grasp_success"]:
                raise RuntimeError("Contact grasp did not pass the lift/hold/contact criteria")
            return
        planner = make_planner(sim.model, sim.data, config, side, output / "curobo")
        report["motion_planner"] = planner.metadata
        start = sim.data.qpos[planner.qids].copy()
        plan = planner.plan(proposals, start)
        np.savez_compressed(output / "trajectory.npz", joint_names=plan.joint_names,
                            waypoints=plan.waypoints, timestamps_s=plan.timestamps_s,
                            T_world_pregrasp=plan.target_tcp)
        write_json(output / "plan.json", asdict(plan))
        report.update(stage="pregrasp_planned", collision_checked_samples=plan.checked_samples,
                      ik_position_error_m=plan.ik_position_error_m, ik_rotation_error_rad=plan.ik_rotation_error_rad)
        if not args.plan_only:
            actuator_ids = np.array([sim.model.actuator(f"servo_{name}").id for name in plan.joint_names])
            start_time = sim.data.time
            next_frame = 0.0
            worst_contacts = []
            max_tracking_error = 0.0
            with imageio.get_writer(output / "pregrasp.mp4", fps=15, macro_block_size=1) as video:
                while sim.data.time - start_time < plan.timestamps_s[-1] + 0.5:
                    elapsed = sim.data.time - start_time
                    command = np.array([np.interp(elapsed, plan.timestamps_s, plan.waypoints[:, j]) for j in range(7)])
                    sim.data.ctrl[actuator_ids] = command
                    sim.step()
                    max_tracking_error = max(max_tracking_error, float(np.max(np.abs(sim.data.qpos[planner.qids] - command))))
                    # Recheck using current object poses, including contact-induced motion.
                    planner.data.qpos[:] = sim.data.qpos
                    contacts = planner.collisions(sim.data.qpos[planner.qids])
                    if contacts:
                        worst_contacts = contacts
                        raise RuntimeError(f"Execution collided: {contacts}")
                    if elapsed >= next_frame:
                        video.append_data(sim.overview())
                        next_frame += 1 / 15
            actual = transform(sim.data.site_xmat[planner.site_id].reshape(3, 3), sim.data.site_xpos[planner.site_id])
            pos_error = float(np.linalg.norm(actual[:3, 3] - plan.target_tcp[:3, 3]))
            rot_error = float(np.linalg.norm(rotation_error(plan.target_tcp[:3, :3], actual[:3, :3])))
            reached = pos_error < 0.015 and rot_error < 0.10
            report.update(stage="pregrasp_reached" if reached else "pregrasp_tracking_failed",
                          pregrasp_reached=reached, execution_position_error_m=pos_error,
                          execution_rotation_error_rad=rot_error, max_joint_tracking_error_rad=max_tracking_error,
                          execution_collisions=worst_contacts, simulation_time_s=float(sim.data.time))
            Image.fromarray(sim.overview()).save(output / "final_overview.png")
            Image.fromarray(sim.capture().rgb).save(output / "final_camera.png")
        write_json(output / "report.json", report)
        print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))
        if report.get("pregrasp_reached") is False:
            raise RuntimeError("Physics tracking did not reach the pregrasp tolerances")


def main():
    parser = argparse.ArgumentParser(description="GR3 RGB-D / FDH-6 contact grasp simulation. No physical motor commands.")
    parser.add_argument("command", choices=["build", "simulate", "infer-frame", "robot-state"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sim.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--backend", choices=["geometry", "graspgen", "m2t2"], default="geometry")
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, help="Evaluate one predicted/retargeted grasp candidate by index")
    parser.add_argument("--arm", choices=["auto", "right", "left"], default="auto")
    parser.add_argument("--planner", choices=["scipy", "curobo"], help="Motion planner; cuRobo requires the separate GPU environment")
    parser.add_argument("--fdh6-offset-profile", type=Path, help="JSON profile produced by scripts/calibrate_wrist_offset.py")
    parser.add_argument("--target-text", help="English object description; requires .[perception]")
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-close", action="store_true", help="Negative control: execute with fingers open; grasp success must fail")
    parser.add_argument("--no-video", action="store_true", help="Skip per-step video rendering for faster evaluation")
    parser.add_argument("--head-scan", action="store_true", help="Fuse RGB-D views acquired by physically moving head servos")
    parser.add_argument("--legacy-m2t2-filter", action="store_true", help="Reproduce the upstream odd-mask-count bug for baseline comparison")
    parser.add_argument("--domain-id", type=int, default=123)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    if args.planner:
        config["planning"]["backend"] = args.planner
    if args.fdh6_offset_profile:
        from .wrist_calibration import load_offset_profile
        if args.backend != "m2t2" or config["robot"]["hand"] != "fdh6":
            parser.error("--fdh6-offset-profile requires --backend m2t2 and an FDH-6 model")
        config.setdefault("m2t2", {})["fdh6_contact_offset_m"] = load_offset_profile(args.fdh6_offset_profile)
        config["m2t2"]["fdh6_offset_profile_side"] = "right"
    if args.head_scan:
        config.setdefault("head_scan", {})["enabled"] = True
    if args.legacy_m2t2_filter:
        config.setdefault("m2t2", {})["fix_object_mask_filter"] = False
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "report.json", {"stage": "started", "grasp_success": False,
                                            "grasp_executed": False})
    try:
        if args.command == "build":
            print(build_scene(config, args.output / "model"))
        elif args.command == "simulate":
            simulate(config, args, args.output)
        elif args.command == "infer-frame":
            if args.frame is None:
                parser.error("infer-frame requires --frame")
            frame = RGBDFrame.load(args.frame)
            Image.fromarray(frame.rgb).save(args.output / "camera.png")
            _, report = perceive(frame, config, args, args.output)
            write_json(args.output / "report.json", {**report, "stage": "proposals_only", "grasp_executed": False})
            print(json.dumps(json_ready(report), indent=2))
        else:
            from .hardware import read_aurora_state
            state = read_aurora_state(args.domain_id)
            write_json(args.output / "robot_state.json", state)
            print(json.dumps(state, indent=2))
    except Exception as error:
        failure = {"stage": "failed", "error": str(error), "grasp_success": False}
        report_path = args.output / "report.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            failure = {**previous, "failed_stage": previous.get("stage"), **failure}
        write_json(report_path, failure)
        print(f"ERROR: {error}\nReport: {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
