"""FDH-6 contact grasp execution. Objects remain unconstrained free bodies."""
from dataclasses import asdict

import mujoco
import numpy as np

from .geometry import rotation_error, transform
from .planning import make_planner


class FDH6:
    def __init__(self, model, side):
        prefix = "R" if side == "right" else "L"
        self.names = [f"{prefix}_thumb_proximal_yaw_joint", f"{prefix}_thumb_proximal_pitch_joint"]
        self.names += [f"{prefix}_{finger}_proximal_joint" for finger in ("index", "middle", "ring", "pinky")]
        self.actuators = np.array([model.actuator(f"servo_{name}").id for name in self.names])
        self.open = np.array([-1.4, 0, 0, 0, 0, 0])
        self.closed = np.array([-1.4, 1.1, -1.4, -1.4, -1.4, -1.4])


def associate_evaluation_body(sim, centroid):
    """Associate the observed target to the scene oracle for contact checks/evaluation.

    Ground-truth poses must never be used to construct grasp poses or IK targets.
    """
    names = [obj["name"] for obj in sim.config["scene"]["objects"]]
    ids = [sim.model.body(name).id for name in names]
    if len(ids) == 1:
        # A single-target evaluation needs no spatial association. Scanned
        # body origins may be far from the visible mass (e.g. a hammer).
        # This selects the evaluation body only; it never supplies grasp poses.
        return ids[0]
    distances = [np.linalg.norm(sim.data.xpos[bid, :2] - centroid[:2]) for bid in ids]
    best = int(np.argmin(distances))
    if distances[best] > 0.08:
        raise RuntimeError("Observed target cannot be associated to an evaluation object within 8 cm")
    return ids[best]


def object_contacts(model, data, target_body, side):
    fingers, normal_force, penetration, table = set(), 0.0, 0.0, False
    prefix = "R_" if side == "right" else "L_"
    for index, contact in enumerate(data.contact):
        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if target_body not in (b1, b2):
            continue
        other = b2 if b1 == target_body else b1
        name = model.body(other).name
        table |= "table" in (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
        if name.startswith(prefix) and "hand_base" not in name:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if force[0] > 0.01:
                fingers.add(name.split("_")[1])
                normal_force += float(force[0])
            penetration = max(penetration, float(max(0, -contact.dist)))
    return fingers, normal_force, penetration, bool(table)


def run_grasp(sim, proposals, report, args, output, write_json, video=None):
    """Plan all stages, then execute position servos through contact physics."""
    model, data, config = sim.model, sim.data, sim.config
    side = report["arm"]
    hand = FDH6(model, side)
    target_body = associate_evaluation_body(sim, np.asarray(report["selected_centroid_world_m"]))
    cloud_path = output / "target_cloud.npz"
    target_points = np.load(cloud_path)["points_world"] if cloud_path.is_file() else None
    planner = make_planner(model, data, config, side, output / "curobo", target_points)
    report["motion_planner"] = planner.metadata
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(output / "planning_start_state.npz", integration_state=state)
    try:
        plans = planner.plan_grasp(proposals, data.qpos[planner.qids].copy(), target_body,
                                   config["hand"]["lift_distance_m"])
    finally:
        write_json(output / "planning_diagnostics.json", getattr(planner, "grasp_diagnostics", []))
    write_json(output / "plan.json", {name: asdict(plan) for name, plan in plans.items()})
    np.savez_compressed(output / "trajectory.npz", joint_names=planner.names,
                        **{f"{name}_waypoints": plan.waypoints for name, plan in plans.items()},
                        **{f"{name}_timestamps_s": plan.timestamps_s for name, plan in plans.items()})
    report.update(stage="grasp_planned", grasp_executed=False, grasp_success=False,
                  evaluation_object=model.body(target_body).name,
                  selected_candidate_index=planner.grasp_diagnostics[-1]["candidate_index"],
                  collision_checked_samples=sum(plan.checked_samples for plan in plans.values()),
                  limitation="Fixed base and ideal robot gravity compensation; assumed camera intrinsics and simulation hand gains. Only separated upright tabletop objects are supported by this baseline.")
    write_json(output / "report.json", report)
    if args.plan_only:
        return report
    arm_actuators = np.array([model.actuator(f"servo_{name}").id for name in planner.names])
    height_reference = config["hand"].get("evaluation_height_reference", "body_origin")
    if height_reference not in ("body_origin", "center_of_mass"):
        raise ValueError("evaluation_height_reference must be body_origin or center_of_mass")

    def object_position():
        return data.xipos[target_body] if height_reference == "center_of_mass" else data.xpos[target_body]

    initial_z = float(object_position()[2])
    records, stage_results = [], []
    start_time, next_frame, next_sample = data.time, 0.0, 0.0
    lift_min = config["hand"].get("success_lift_m", 0.08)
    max_penetration = 0.0
    report.update(grasp_executed=True, initial_object_height_m=initial_z, object_height_reference=height_reference)

    def step(stage, allowed_object):
        nonlocal next_frame, next_sample, max_penetration
        sim.step()
        fingers, force, penetration, table = object_contacts(model, data, target_body, side)
        max_penetration = max(max_penetration, penetration)
        elapsed = data.time - start_time
        # Object poses and passive fingers change under contact: use measured state.
        planner.data.qpos[:] = data.qpos
        forbidden = planner.collisions(data.qpos[planner.qids], allowed_object)
        if forbidden:
            raise RuntimeError(f"Forbidden collision during {stage}: {forbidden}")
        if penetration > config["hand"].get("max_contact_penetration_m", 0.003):
            raise RuntimeError(f"Excessive target/finger penetration: {penetration:.4f} m")
        if elapsed >= next_sample:
            records.append({"time_s": float(elapsed), "stage": stage,
                            "object_height_m": float(object_position()[2]),
                            "object_position_m": object_position().copy(),
                            "lift_m": float(object_position()[2] - initial_z),
                            "fingers": sorted(fingers), "normal_force_n": force,
                            "penetration_m": penetration, "table_contact": table})
            next_sample += 0.02
        if video is not None and elapsed >= next_frame:
            video.append_data(sim.grasp_view(target_body, stage))
            next_frame += 1 / 15

    def motion(stage, plan, allowed_object):
        begin = data.time
        while data.time - begin < plan.timestamps_s[-1] + 0.5:
            elapsed = data.time - begin
            data.ctrl[arm_actuators] = [np.interp(elapsed, plan.timestamps_s, plan.waypoints[:, j]) for j in range(7)]
            step(stage, allowed_object)
        actual = transform(data.site_xmat[planner.site_id].reshape(3, 3), data.site_xpos[planner.site_id])
        pe = float(np.linalg.norm(actual[:3, 3] - plan.target_tcp[:3, 3]))
        re = float(np.linalg.norm(rotation_error(plan.target_tcp[:3, :3], actual[:3, :3])))
        stage_results.append({"stage": stage, "position_error_m": pe, "rotation_error_rad": re})
        if pe > 0.015 or re > 0.10:
            raise RuntimeError(f"{stage} tracking failed: {pe:.4f} m, {re:.4f} rad")

    def progress(stage):
        report.update(stage=stage, stages=stage_results)
        write_json(output / "report.json", report)

    try:
        data.ctrl[hand.actuators] = hand.open
        progress("pregrasp")
        motion("pregrasp", plans["pregrasp"], None)
        report["pregrasp_reached"] = True
        progress("approach")
        motion("approach", plans["approach"], target_body)
        progress("close")
        begin = data.time
        duration = config["hand"]["close_duration_s"]
        while data.time - begin < duration:
            alpha = min(1, (data.time - begin) / max(0.1, duration - 0.5))
            data.ctrl[hand.actuators] = hand.open if args.no_close else hand.open + alpha * (hand.closed - hand.open)
            step("close", target_body)
        progress("lift")
        motion("lift", plans["lift"], target_body)
        progress("hold")
        begin = data.time
        while data.time - begin < config["hand"]["hold_duration_s"]:
            step("hold", target_body)
        hold = [record for record in records if record["stage"] == "hold"]
        retained = ["thumb" in r["fingers"] and len(set(r["fingers"]) - {"thumb"}) > 0 for r in hold]
        min_height = min(r["lift_m"] for r in hold)
        success = min_height >= lift_min and not any(r["table_contact"] for r in hold) and np.mean(retained) >= 0.9
        report.update(stage="grasp_succeeded" if success else "grasp_failed", grasp_success=bool(success),
                      final_lift_m=float(object_position()[2] - initial_z),
                      hold_min_lift_m=min_height, hold_duration_s=config["hand"]["hold_duration_s"],
                      hold_opposed_contact_fraction=float(np.mean(retained)),
                      max_target_finger_penetration_m=max_penetration,
                      success_lift_threshold_m=lift_min, no_close_control=args.no_close,
                      simulation_time_s=float(data.time))
    finally:
        np.savez_compressed(output / "final_state.npz", qpos=data.qpos, qvel=data.qvel,
                            ctrl=data.ctrl, time_s=data.time)
        write_json(output / "contact_trace.json", records)
        write_json(output / "report.json", report)
    return report
