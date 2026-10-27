"""Use externally supplied RGB-D/state to plan in an isolated digital twin."""
from dataclasses import asdict
import hashlib
import json
import xml.etree.ElementTree as ET
import uuid

import numpy as np
import yaml

from .protocol import (VERSION, ARM_NAMES, HAND_NAMES, COMMAND_NAMES, LIVE_SOURCES, digest,
                       validate_observation, validate_plan, save, write_json)


def configure_scene(path, obs):
    """Compile measured fixed geometry into the model, including its collision BVH."""
    from scipy.spatial.transform import Rotation
    tree = ET.parse(path)
    root = tree.getroot()
    base = root.find("worldbody/body[@name='base_link']")
    T_base = np.asarray(obs["T_world_base"])
    values = lambda a: " ".join(str(float(v)) for v in a)
    base.set("pos", values(T_base[:3,3]))
    base.set("quat", values(Rotation.from_matrix(T_base[:3,:3]).as_quat()[[3,0,1,2]]))
    table = root.find("worldbody/geom[@name='table']")
    table.set("pos", values([*obs["table_center_xy"], obs["table_top_m"]-.025]))
    table.set("size", values([*obs["table_half_size_xy"],.025]))
    tree.write(path)


class DexonomyInference:
    name = "dexonomy_known_mesh_geometry_curobo"

    def __init__(self, config):
        self.config = config

    def plan(self, obs, directory):
        # Import after CLI configures headless rendering, and only on server.
        from gr3_grasping.cli import write_json as _headless_setup
        import mujoco
        from scipy.spatial.transform import Rotation
        from gr3_grasping.geometry import RGBDFrame, transform, transform_points
        from gr3_grasping.model import ROOT
        from gr3_grasping.dexonomy_backend import expand_hand_q
        from gr3_grasping.dexonomy_execution import build_experiment, pose, MotorExecution, GraspPathPlanner, compiled_model_sha256
        from gr3_grasping.head_scan import voxel_fuse
        from gr3_grasping.perception import segment_tabletop
        from gr3_grasping.pose_estimation import model_points, estimate_upright_pose
        from gr3_grasping.simulation import Simulation

        validate_observation(obs)
        if obs["object_id"] != self.config["inference"]["object_id"]:
            raise ValueError("MVP serves only the configured known object")
        save(directory / "observation.msgpack", obs)
        failures = []
        for index, candidate in enumerate(self.config["inference"]["candidates"]):
            attempt = directory/f"attempt_{index:02d}"
            attempt.mkdir()
            settings = yaml.safe_load((ROOT/self.config["inference"]["arm_config"]).read_text())
            settings["candidate"] = candidate
            settings["registration"] = "external_rgbd_geometry"
            path, cfg, grasp, source, provenance = build_experiment(settings, attempt/"model")
            cfg["scene"]["table_top_m"] = float(obs["table_top_m"])
            cfg["perception"].update(self.config["inference"]["perception"])
            # The robot side measures the scene, so its workspace box (if sent) overrides the server's.
            cfg["perception"].update(obs.get("perception_workspace", {}))
            configure_scene(path, obs)
            with Simulation(path, cfg, render=False) as sim:
                T_base = np.asarray(obs["T_world_base"])
                provenance.update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  compiled_model_sha256=compiled_model_sha256(sim.model))
                cfg["scene"].update(table_center_xy=obs["table_center_xy"], table_half_size_xy=obs["table_half_size_xy"])
                for name, value in obs["joint_positions_rad"].items():
                    joint = sim.model.joint(name)  # Unknown names are rejected, never silently ignored.
                    if joint.type != mujoco.mjtJoint.mjJNT_HINGE or not joint.range[0]-1e-5 <= value <= joint.range[1]+1e-5:
                        raise ValueError(f"Measured joint outside model: {name}")
                    sim.data.qpos[joint.qposadr[0]] = np.clip(value, *joint.range)
                # Passive finger coordinates are estimates from the official coupling, not measured truth.
                meta = json.loads((ROOT/settings["hand_metadata"]).read_text())
                active_ids = [meta["joint_order"].index(n) for n in meta["active_joints"]]
                hand_q = [obs["joint_positions_rad"][n] for n in meta["active_joints"]]
                sim.data.qpos[[sim.model.joint(n).qposadr[0] for n in meta["joint_order"]]] = expand_hand_q(hand_q, meta)
                sim.data.ctrl[:] = sim.data.qpos[sim.motor_qpos]
                mujoco.mj_forward(sim.model, sim.data)
                frames = [RGBDFrame(**r) for r in obs["frames"]]
                points, colors = [], []
                for frame in frames:
                    clouds = segment_tabletop(frame, cfg["perception"])
                    if clouds:
                        target = max(clouds, key=lambda cloud: len(cloud.points_world))
                        points.append(target.points_world); colors.append(target.rgb)
                if not points:
                    raise RuntimeError("No observed tabletop object in configured workspace")
                observed, _ = voxel_fuse(np.concatenate(points), np.concatenate(colors), .003)
                target = sim.model.body("target").id
                body_points = model_points(sim.model, target, .003)
                T_object, diagnostics = estimate_upright_pose(observed, body_points, obs["table_top_m"], frames)
                write_json(attempt/"pose_estimate.json", dict(T_world_object=T_object, **diagnostics))
                if diagnostics["best"]["residual_m"] > self.config["inference"]["max_pose_residual_m"]:
                    raise RuntimeError("Observed known-mesh pose residual exceeds threshold")
                object_q = sim.model.joint("target_free").qposadr[0]
                sim.data.qpos[object_q:object_q+7] = np.r_[T_object[:3,3],
                    Rotation.from_matrix(T_object[:3,:3]).as_quat()[[3,0,1,2]]]
                mujoco.mj_forward(sim.model, sim.data)
                initial = np.array([obs["joint_positions_rad"][n] for n in COMMAND_NAMES])
                first_hand = grasp["pregrasp_qpos"][0][7:][active_ids]
                settings["preform_duration_s"] = max(settings["preform_duration_s"],
                    float(np.max(abs(first_hand-initial[7:])))/settings["max_hand_joint_speed_rad_s"])
                preview_dir = attempt/"preform_preview"
                preview_dir.mkdir()
                preform = MotorExecution(sim, meta, preview_dir, settings)
                try:
                    preform.preform(grasp["pregrasp_qpos"][0][7:][active_ids])
                    preform.finish("digital_twin_preform_only")
                    root_id, tcp = sim.model.body("R_hand_base_link").id, sim.model.site("right_tcp").id
                    hand_tcp = np.linalg.inv(transform(sim.data.xmat[root_id].reshape(3,3), sim.data.xpos[root_id])) @ transform(
                        sim.data.site_xmat[tcp].reshape(3,3), sim.data.site_xpos[tcp])
                    world_scene = T_object @ np.linalg.inv(pose(np.asarray(source["pose"])))
                    planner = GraspPathPlanner(sim, cfg, settings, meta, attempt/"curobo",
                        transform_points(T_object, body_points), object_pose=T_object)
                    try:
                        plans = planner.plan(grasp, world_scene, hand_tcp)
                    finally:
                        write_json(attempt/"planner_metadata.json", planner.arm.metadata)
                    write_json(attempt/"provenance.json", {**provenance,
                        "registration": "external RGB-D, no simulator camera or object truth read",
                        "T_world_base": T_base, "table_top_m": obs["table_top_m"]})
                    write_json(attempt/"source_plans.json", {k:asdict(v) for k,v in plans.items()})
                except Exception as error:
                    failures.append(dict(candidate=candidate, error=f"{type(error).__name__}: {error}"))
                    write_json(directory/"planning_failures.json", failures)
                    continue
                stages = []
                preform_end = np.r_[initial[:7], first_hand]
                stages.append(dict(name="preform", times_s=np.array([0., settings["preform_duration_s"], settings["preform_duration_s"]+.5]),
                    positions_rad=np.stack([initial, preform_end, preform_end])))
                previous = preform_end.copy()
                for name, plan in plans.items():
                    q = np.column_stack([plan.arm, plan.hand])
                    # Include tiny preform servo drift as a speed-bounded connector.
                    connector = max(.02, float(np.max(abs(q[0]-previous)/np.array(self.config["execution"]["max_speed_rad_s"]))))
                    times = np.r_[0., plan.times+connector, plan.times[-1]+connector+.5]
                    positions = np.vstack([previous, q, q[-1]])
                    stages.append(dict(name=name, times_s=times, positions_rad=positions))
                    previous = q[-1].copy()
                stages.append(dict(name="hold", times_s=np.array([0.,settings["hold_duration_s"]]),
                                   positions_rad=np.stack([previous,previous])))
                eligible = bool(obs.get("calibration_verified") and obs.get("state_mapping_verified")
                    and obs.get("source") in LIVE_SOURCES and self.config["inference"]["real_rgb_pose_validated"])
                result = dict(version=VERSION, task_id=uuid.uuid4().hex, observation_id=obs["observation_id"],
                    observation_digest=digest(obs), joint_names=COMMAND_NAMES, stages=stages,
                    context_joint_positions_rad={n:v for n,v in obs["joint_positions_rad"].items()
                        if n not in COMMAND_NAMES and not n.startswith("R_")},
                    hardware_eligible=eligible, candidate=candidate, T_world_object=T_object,
                    pose_method="partial RGB-D planar geometry; color/symmetry disambiguation not validated on hardware",
                    grasp_success=None, scope="Server inference from external observations; contact preview is an approximate digital twin")
                validate_plan(result, obs["observation_id"], self.config["execution"])
                return result
        raise RuntimeError(f"No planned candidate: {failures}")


class TransportDemo:
    """No perception or grasp planning. Exercises the exact same transport/executor."""
    name = "transport_demo_no_inference"

    def plan(self, obs, directory):
        validate_observation(obs)
        q = np.array([obs["joint_positions_rad"][n] for n in COMMAND_NAMES])
        return dict(version=VERSION, task_id=uuid.uuid4().hex, observation_id=obs["observation_id"],
            observation_digest=digest(obs), joint_names=COMMAND_NAMES, hardware_eligible=False,
            context_joint_positions_rad={n:v for n,v in obs["joint_positions_rad"].items()
                if n not in COMMAND_NAMES and not n.startswith("R_")},
            scope="Transport demo: constant positions, no inference, no grasp success claim", grasp_success=None,
            stages=[dict(name=n, times_s=np.array([0.,.1]), positions_rad=np.stack([q,q]))
                    for n in ("preform","pregrasp","approach","grasp","squeeze","lift","hold")])
