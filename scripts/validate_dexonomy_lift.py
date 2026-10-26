#!/usr/bin/env python3
"""Gravity lift check for offline FDH-6 poses, with a controlled floating wrist.

No GR3 arm or camera. No synthesis forces, pose projection, or object weld.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from gr3_grasping.dexonomy_backend import ROOT


def validate(path, output, video=False, no_close=False, timestep=.0005, noslip=0, results_output=None,
             integrator="discrete", wrist_timeconstant=.004):
    from dexonomy.sim import HandCfg, MuJoCo_EvalEnv, MuJoCo_EvalCfg
    from dexonomy.util.file_util import load_scene_cfg

    grasp = np.load(path, allow_pickle=True).item()
    scene = load_scene_cfg(grasp["scene_path"])
    name = scene["task"]["obj_name"]
    provenance = json.loads((Path(scene["scene"][name]["xml_path"]).parent / "provenance.json").read_text())
    mass = provenance["mass_kg"]
    info = json.loads(Path(scene["scene"][name]["info_path"]).read_text())
    scene["scene"][name]["density"] = mass / (info["mass"] / info["density"])
    hand = output / "assets/hand/fdh6"
    meta = json.loads((hand / "hand_meta.json").read_text())
    env = MuJoCo_EvalEnv(HandCfg(str(hand / "hand.xml"), freejoint=True), scene,
        MuJoCo_EvalCfg(timestep=timestep, miu_coef=(1., .005)), debug_view=False)
    model, data = env._model, env._data
    model.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + integrator.upper())
    for eq in range(model.neq):
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_WELD:
            model.eq_solref[eq] = [wrist_timeconstant, 1]
    # Match the existing GR3 solver's lack of the optional no-slip pass.
    # Keep this explicit for comparison with upstream's noslip_iterations=2.
    model.opt.noslip_iterations = noslip
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    model.opt.gravity[:] = [0, 0, -9.81]
    for bid in range(model.nbody):
        if model.body(bid).name.startswith("hand-"):
            model.body_gravcomp[bid] = 1
    active = [model.jnt_qposadr[model.joint("hand-" + n).id] for n in meta["active_joints"]]
    def joint_diagnostics():
        errors, torques = {}, {}
        for eq in range(model.neq):
            if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                j1, j2 = model.eq_obj1id[eq], model.eq_obj2id[eq]
                errors[model.joint(int(j1)).name] = float(data.qpos[model.jnt_qposadr[j1]] -
                    model.eq_data[eq, 1] * data.qpos[model.jnt_qposadr[j2]])
        for aid, name in enumerate(meta["active_joints"]):
            torques[name] = float(data.actuator_force[aid])
        return dict(mimic_errors_rad=errors, motor_torque_nm=torques,
            all_joint_qpos_rad=data.qpos[7:18].tolist(),
            solver_iterations=np.asarray(data.solver_niter).tolist(),
            equality_residual=data.efc_pos[:11].tolist(),
            equality_force=data.efc_force[:11].tolist())
    phases = [("pregrasp", grasp["pregrasp_qpos"][0], .5)]
    for q in grasp["pregrasp_qpos"][1:]:
        phases.append(("approach", q, 2. / max(len(grasp["pregrasp_qpos"]) - 1, 1)))
    phases.extend([("grasp", grasp["grasp_qpos"][0], 2.),
                   ("squeeze", grasp["squeeze_qpos"][0], 2.)])
    lift = grasp["squeeze_qpos"][0].copy()
    lift[2] += .16
    phases.extend([("lift", lift, 3.), ("hold", lift, 2.)])
    start = phases[0][1].copy()
    env.reset_qpos(start)
    # All equality constraints are hand mimic joints and the hand-to-mocap weld.
    # No equality is allowed to reference the object.
    for eq in range(model.neq):
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_WELD:
            assert all(not model.body(int(bid)).name.startswith("obj-")
                       for bid in (model.eq_obj1id[eq], model.eq_obj2id[eq]))
    oid = env._obj_active_bodyid
    actual_mass = float(model.body_subtreemass[oid])
    if not np.isclose(actual_mass, mass, rtol=1e-5):
        raise RuntimeError(f"Object mass mismatch: {actual_mass} vs {mass}")
    baseline = float(data.subtree_com[oid, 2])
    frames, frame_states, samples, failure = [], [], [], None
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .17]
    camera.distance, camera.azimuth, camera.elevation = .85, 135, -25
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    counter, maximum_penetration, maximum_mimic = 0, 0., 0.
    sample_every = round(.02 / timestep)
    hold_steps, opposed_steps, table_hold = 0, 0, False
    minimum_hold, peak_lift, maximum_hand_collision = float("inf"), -float("inf"), 0.
    maximum_wrist_position, maximum_wrist_rotation = 0., 0.
    try:
        for phase, target, duration in phases:
            begin = start.copy()
            slerp = Slerp([0, 1], Rotation.from_quat(np.stack([begin[3:7], target[3:7]])[:, [1, 2, 3, 0]]))
            steps = round(duration / model.opt.timestep)
            for i in range(steps):
                alpha = (i + 1) / steps
                command = begin * (1 - alpha) + target * alpha
                command[3:7] = slerp(alpha).as_quat()[[3, 0, 1, 2]]
                if no_close:
                    command[7:] = phases[0][1][7:]
                # Commands only set the mocap wrist and six motor targets.
                env.set_ctrl(command)
                data.qfrc_applied[:] = 0
                data.xfrc_applied[:] = 0
                mujoco.mj_step(model, data)
                counter += 1
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                    failure = "numerical_instability"
                    break
                fingers, penetration, table = set(), 0., False
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    bnames = [model.body(model.geom_bodyid[int(g)]).name for g in con.geom]
                    geoms = [model.geom(int(g)).name for g in con.geom]
                    n_hand = sum(b.startswith("hand-") for b in bnames)
                    if n_hand == 2 or (n_hand and any(g.startswith("plane-") for g in geoms)):
                        maximum_hand_collision = max(maximum_hand_collision, max(0., -float(con.dist)))
                    if not any(b.startswith("obj-" + name) for b in bnames):
                        continue
                    hname = next((b for b in bnames if b.startswith("hand-")), None)
                    if hname:
                        penetration = max(penetration, max(0., -float(con.dist)))
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, force)
                        if force[0] > .01:
                            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                                if f"R_{finger}_" in hname:
                                    fingers.add(finger)
                    else:
                        table = True
                mimic = 0.
                for eq in range(model.neq):
                    if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                        j1, j2 = model.eq_obj1id[eq], model.eq_obj2id[eq]
                        mimic = max(mimic, abs(data.qpos[model.jnt_qposadr[j1]] -
                            model.eq_data[eq, 1] * data.qpos[model.jnt_qposadr[j2]]))
                maximum_penetration = max(maximum_penetration, penetration)
                maximum_mimic = max(maximum_mimic, mimic)
                wrist_position_error = float(np.linalg.norm(data.qpos[:3] - command[:3]))
                wrist_rotation_error = float(2 * np.arccos(np.clip(abs(np.dot(data.qpos[3:7], command[3:7])), 0, 1)))
                maximum_wrist_position = max(maximum_wrist_position, wrist_position_error)
                maximum_wrist_rotation = max(maximum_wrist_rotation, wrist_rotation_error)
                lift_height = float(data.subtree_com[oid, 2] - baseline)
                peak_lift = max(peak_lift, lift_height)
                if phase == "hold":
                    hold_steps += 1
                    opposed_steps += int("thumb" in fingers and len(fingers) > 1)
                    table_hold |= table
                    minimum_hold = min(minimum_hold, lift_height)
                if counter % sample_every == 0:
                    sample = dict(time_s=float(data.time), phase=phase,
                        lift_m=float(data.subtree_com[oid, 2] - baseline),
                        opposed="thumb" in fingers and len(fingers) > 1,
                        table_contact=table, penetration_m=penetration, mimic_error_rad=mimic,
                        motor_torque_max_nm=float(abs(data.actuator_force).max()),
                        wrist_position_error_m=wrist_position_error,
                        wrist_rotation_error_rad=wrist_rotation_error,
                        active_target_rad=command[active].tolist(),
                        active_actual_rad=data.qpos[active].tolist())
                    sample.update(joint_diagnostics())
                    samples.append(sample)
                    if renderer:
                        frame_states.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                            ctrl=data.ctrl.copy(), mocap_pos=data.mocap_pos.copy(),
                            mocap_quat=data.mocap_quat.copy(), time_s=float(data.time)))
                        renderer.update_scene(data, camera, option)
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 45), fill="black")
                        draw.text((8, 5), f"FDH-6 offline mesh / floating wrist | {name} | {phase}", fill="white")
                        draw.text((8, 23), f"t={data.time:.2f}s COM lift={sample['lift_m']*100:.1f}cm | no GR3 arm or camera", fill="white")
                        frames.append(np.asarray(frame))
                if maximum_penetration > .003:
                    failure = "penetration_exceeds_3mm"
                    break
                if maximum_mimic > .01:
                    failure = "mimic_error_exceeds_0.01rad"
                    break
                if maximum_hand_collision > .003:
                    failure = "hand_self_or_table_penetration_exceeds_3mm"
                    break
                if maximum_wrist_position > .015 or maximum_wrist_rotation > .10:
                    failure = "wrist_tracking_exceeds_tolerance"
                    break
            if failure:
                break
            start = target.copy()
    finally:
        if renderer:
            renderer.close()
    hold_duration = hold_steps * timestep
    minimum = minimum_hold if hold_steps else 0.
    opposed = opposed_steps / hold_steps if hold_steps else 0.
    success = not failure and hold_duration >= 1.99 and minimum >= .1 and opposed >= .9 and not table_hold
    if not success and failure is None:
        failure = "insufficient_lift_or_opposed_hold"
    tag = f"mj{mujoco.__version__}_{integrator}_wrist{round(wrist_timeconstant*1e6)}us_noslip{noslip}_dt{round(timestep*1e6)}us"
    destination = (results_output or output) / f"lift_validation_{tag}" / (path.relative_to(output / "run/grasp_data").with_suffix(""))
    if no_close:
        destination = destination.with_name(destination.name + "_open_control")
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(path), success=bool(success), failure=failure,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        no_close=no_close, protocol="gravity floating-wrist lift; no camera/arm; no optimizer projection",
        timestep_s=timestep,
        noslip_iterations=noslip,
        mujoco_version=mujoco.__version__, integrator=integrator,
        wrist_timeconstant_s=wrist_timeconstant,
        mass_kg=actual_mass, hold_duration_s=hold_duration, hold_min_lift_m=minimum,
        peak_lift_m=peak_lift if np.isfinite(peak_lift) else 0.,
        opposed_fraction=opposed, max_penetration_m=maximum_penetration,
        max_mimic_error_rad=maximum_mimic, samples=samples)
    report["physics"] = dict(motor_kp=model.actuator_gainprm[:, 0].tolist(),
        motor_kv=(-model.actuator_biasprm[:, 2]).tolist(),
        motor_force_range_nm=model.actuator_forcerange.tolist(),
        friction=[1., .005], condim=4, impratio=float(model.opt.impratio),
        gravity=model.opt.gravity.tolist(), hand_gravity_compensated=True,
        object_attachment=False, runtime_joint_projection=False,
        success_lift_m=.1, hold_duration_s=2., opposed_fraction_min=.9,
        penetration_limit_m=.003, mimic_error_limit_rad=.01,
        wrist_position_limit_m=.015, wrist_rotation_limit_rad=.1)
    report["max_hand_self_or_table_penetration_m"] = maximum_hand_collision
    report["max_wrist_position_error_m"] = maximum_wrist_position
    report["max_wrist_rotation_error_rad"] = maximum_wrist_rotation
    report["termination_phase"] = phase
    report["termination_time_s"] = float(data.time)
    report["termination_joints"] = joint_diagnostics()
    np.savez_compressed(destination / "terminal_state.npz", qpos=data.qpos, qvel=data.qvel,
        ctrl=data.ctrl, mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if frames:
        np.savez_compressed(destination / "video_states.npz",
                            **{key: np.stack([s[key] for s in frame_states]) for key in frame_states[0]})
        imageio.mimwrite(destination / "lift.mp4", frames, fps=50, codec="libx264")
    return {k: v for k, v in report.items() if k != "samples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dexonomy_fdh6_v2")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-close", action="store_true")
    parser.add_argument("--object")
    parser.add_argument("--timestep", type=float, default=.0005,
                        help="0.5 ms resolves the URDF's 2 ms mimic constraint time constant")
    parser.add_argument("--noslip-iterations", type=int, choices=[0, 2], default=0)
    parser.add_argument("--results-output", type=Path, help="Keep source candidates unchanged and save a separate audit")
    parser.add_argument("--integrator", choices=["implicitfast", "implicit", "discrete"], default="discrete")
    parser.add_argument("--wrist-timeconstant", type=float, default=.004,
                        help="Floating-wrist servo time constant; upstream used .02 s")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel candidate processes")
    args = parser.parse_args()
    root = args.output.resolve()
    if not hasattr(mujoco.mjtIntegrator, "mjINT_" + args.integrator.upper()):
        parser.error(f"MuJoCo {mujoco.__version__} does not support {args.integrator}; run bash scripts/setup_dexonomy_eval.sh")
    if args.timestep <= 0 or args.timestep > .002 or not np.isclose(.02 / args.timestep, round(.02 / args.timestep)):
        parser.error("timestep must divide .02 seconds and be in (0, .002]")
    if args.wrist_timeconstant < 2 * args.timestep:
        parser.error("wrist time constant must be at least twice the timestep")
    paths = sorted((root / "run/grasp_data").rglob("*.npy"))
    if args.object:
        paths = [p for p in paths if p.parent.name == args.object]
    if not paths:
        parser.error("No complete grasp candidates found for this selection; run synthesis first")
    results_output = args.results_output.resolve() if args.results_output else root
    results_output.mkdir(parents=True, exist_ok=True)
    params = [(p, root, args.video, args.no_close, args.timestep, args.noslip_iterations,
               results_output, args.integrator, args.wrist_timeconstant) for p in paths]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            reports = pool.starmap(validate, params)  # Keeps the sorted path order.
    else:
        reports = [validate(*param) for param in params]
    suffix = ("_" + args.object if args.object else "") + ("_open_control" if args.no_close else "")
    filename = f"lift_summary_mj{mujoco.__version__}_{args.integrator}_wrist{round(args.wrist_timeconstant*1e6)}us_noslip{args.noslip_iterations}_dt{round(args.timestep*1e6)}us{suffix}.json"
    (results_output / filename).write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
