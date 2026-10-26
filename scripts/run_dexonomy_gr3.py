#!/usr/bin/env python3
"""Plan and execute a known Dexonomy grasp with the full GR3 right arm."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import traceback

# Set headless rendering before importing MuJoCo.
from gr3_grasping.cli import write_json
import numpy as np
import yaml
from gr3_grasping.geometry import transform, transform_points
from gr3_grasping.model import ROOT
from gr3_grasping.simulation import Simulation
from gr3_grasping.dexonomy_execution import (
    build_experiment, pose, GraspPathPlanner, MotorExecution, visible_object_vertices, DexterousPlan)
from gr3_grasping.pose_estimation import DEFAULTS as POSE_DEFAULTS, model_points, observe_object_pose, pose_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dexonomy_gr3_mug.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-close", action="store_true", help="Keep the initial pregrasp fingers along the identical arm path")
    parser.add_argument("--replay-plan", type=Path, help="Directory containing plans.json and planning_state.npz")
    parser.add_argument("--yaw-deg", type=float, help="Object yaw about the table normal (scene placement)")
    parser.add_argument("--registration", choices=["oracle", "camera"], default="oracle",
                        help="camera: head RGB-D scan + known-mesh fit; the true pose is only used for evaluation")
    parser.add_argument("--perceive-only", action="store_true", help="Camera registration only; no arm motion")
    parser.add_argument("--depth-noise", type=float, help="Camera depth noise std (m)")
    parser.add_argument("--depth-dropout", type=float, help="Camera depth dropout fraction")
    parser.add_argument("--candidate", type=Path, help="Override the config candidate: <root>/run/grasp_data/<template>/<object>/<n>_grasp.npy")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Preserve each failed trial.
    settings = yaml.safe_load(args.config.read_text())
    if args.candidate:
        candidate = args.candidate.resolve()
        root = candidate.parents[3].parent
        item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"]
                    if i["id"] == candidate.parent.name)
        settings.update(candidate=str(candidate), hand_metadata=str(root / "assets/hand/fdh6/hand_meta.json"),
                        object_id=item["id"], object_folder=item["folder"], mass_kg=item["mass_kg"])
    if args.yaw_deg is not None:
        settings["yaw_deg"] = args.yaw_deg
    if args.perceive_only and args.registration != "camera":
        parser.error("--perceive-only requires --registration camera")
    if args.registration == "camera":
        settings["registration"] = "camera"
        for key, value in (("depth_noise_std_m", args.depth_noise), ("depth_dropout", args.depth_dropout)):
            if value is not None:
                settings.setdefault("camera", {})[key] = value
    path, cfg, grasp, source, provenance = build_experiment(settings, output / "model")
    meta = json.loads((ROOT / settings["hand_metadata"]).read_text())
    camera = args.registration == "camera"
    if camera:
        cfg["camera"].update(settings.get("camera", {}))
        provenance["registration"] = "head RGB-D scan, known-mesh planar fit; true pose used only for evaluation"
    write_json(output / "provenance.json", provenance)
    with Simulation(path, cfg, render=args.video or camera) as sim:
        executor = MotorExecution(sim, meta, output, settings, args.video, args.no_close)
        try:
            sim.settle()
            T_estimate = None
            if camera:
                # Static scene: scan with the head servos before any arm motion.
                T_estimate, perception = observe_object_pose(sim, cfg, output / "perception")
                target = sim.model.body("target").id
                T_true = transform(sim.data.xmat[target].reshape(3, 3), sim.data.xpos[target])
                estimate = dict(T_world_body=T_estimate, T_world_scene=T_estimate @ np.linalg.inv(pose(np.array(source["pose"]))),
                    best=perception["best"], evaluation_only_T_world_body_true=T_true,
                    evaluation_only_errors=pose_errors(T_estimate, T_true, model_points(sim.model, target, POSE_DEFAULTS["model_voxel_m"])))
                write_json(output / "camera_registration.json", estimate)
                if args.perceive_only:
                    raise SystemExit(0)
            active = [meta["joint_order"].index(n) for n in meta["active_joints"]]
            executor.preform(grasp["pregrasp_qpos"][0][7:][active])
            state = dict(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), ctrl=sim.data.ctrl.copy(), time_s=float(sim.data.time))
            np.savez_compressed(output / "planning_state.npz", **state)
            if args.replay_plan:
                previous = json.loads((args.replay_plan / "provenance.json").read_text())
                model_key = "compiled_model_sha256" if "compiled_model_sha256" in previous else "model_sha256"
                if previous[model_key] != provenance[model_key] or previous["source_sha256"] != provenance["source_sha256"]:
                    raise RuntimeError("Replay model or candidate differs")
                initial = np.load(args.replay_plan / "planning_state.npz")
                if any(not np.allclose(initial[k], state[k], atol=1e-10, rtol=0) for k in state):
                    raise RuntimeError("Replay initial physical state differs; refusing state teleport")
                if camera:
                    planned = json.loads((args.replay_plan / "registration.json").read_text())
                    if not np.allclose(planned["T_world_scene"], estimate["T_world_scene"], atol=1e-9):
                        raise RuntimeError("Replay camera estimate differs from the planning estimate")
                records = json.loads((args.replay_plan / "plans.json").read_text())
                write_json(output / "replay_metadata.json", dict(source_directory=str(args.replay_plan.resolve()),
                    plans_sha256=hashlib.sha256((args.replay_plan / "plans.json").read_bytes()).hexdigest(),
                    model_sha256=provenance["model_sha256"], initial_state_verified=True,
                    compiled_model_sha256=provenance["compiled_model_sha256"],
                    no_state_reset=True, no_close=args.no_close))
                plans = {k: DexterousPlan(np.array(v["arm"]), np.array(v["hand"]), np.array(v["times"]),
                    np.array(v["target_tcp"]), v["source"]) for k, v in records.items()}
            else:
                target = sim.model.body("target").id
                hand = sim.model.body("R_hand_base_link").id
                tcp = sim.model.site("right_tcp").id
                T_object = transform(sim.data.xmat[target].reshape(3, 3), sim.data.xpos[target])
                T_hand = transform(sim.data.xmat[hand].reshape(3, 3), sim.data.xpos[hand])
                T_tcp = transform(sim.data.site_xmat[tcp].reshape(3, 3), sim.data.site_xpos[tcp])
                if camera:
                    world_scene = estimate["T_world_scene"]
                    obstacle = transform_points(T_estimate, model_points(sim.model, target, POSE_DEFAULTS["model_voxel_m"]))
                else:
                    world_scene = T_object @ np.linalg.inv(pose(np.array(source["pose"])))
                    obstacle = visible_object_vertices(sim.model, sim.data, target)
                hand_tcp = np.linalg.inv(T_hand) @ T_tcp
                write_json(output / "registration.json", dict(T_world_scene=world_scene, T_hand_tcp=hand_tcp,
                                                              source="camera" if camera else "oracle"))
                planner = GraspPathPlanner(sim, cfg, settings, meta, output / "curobo", obstacle,
                                          object_pose=T_estimate)
                try:
                    plans = planner.plan(grasp, world_scene, hand_tcp)
                finally:
                    write_json(output / "planner_metadata.json", planner.arm.metadata)
                    write_json(output / "planning_diagnostics.json", planner.diagnostics)
            write_json(output / "plans.json", {k: asdict(v) for k, v in plans.items()})
            print("Plan ready: " + ", ".join(f"{k}={len(v.arm)}" for k, v in plans.items()), flush=True)
            if not args.plan_only:
                executor.execute(plans)
            result = executor.finish("plan_only_no_execution" if args.plan_only else None)
        except Exception as error:
            traceback.print_exc()
            result = executor.finish(f"{type(error).__name__}: {error}")
    print((output / "execution_report.json").read_text(), end="")
    return 0 if result["success"] or (args.plan_only and (output / "plans.json").is_file()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
