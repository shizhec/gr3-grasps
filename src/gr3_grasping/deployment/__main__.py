"""python -m gr3_grasping.deployment --help"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import yaml

from .protocol import (COMMAND_NAMES, LIVE_SOURCES, VERSION, digest, load, save, validate_observation,
    validate_plan, write_json)

ROOT = Path(__file__).resolve().parents[3]


def sample(source, output, object_id):
    """Convert saved simulation observations into a marked replay, not real-camera evidence."""
    from gr3_grasping.cli import write_json as _headless_setup
    import mujoco
    from gr3_grasping.geometry import RGBDFrame, transform
    model = mujoco.MjModel.from_xml_path(str(source/"model/scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = np.load(source/"planning_state.npz")["qpos"]
    mujoco.mj_forward(model, data)
    base, table = model.body("base_link").id, model.geom("table").id
    frames = [asdict(RGBDFrame.load(p)) for p in sorted((source/"perception/head_scan").glob("frame_*.npz"))]
    positions = {model.joint(i).name:float(data.qpos[model.jnt_qposadr[i]]) for i in range(model.njnt)
                 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    observation = dict(version=VERSION, observation_id=uuid.uuid4().hex, object_id=object_id,
        source="simulation_replay", frames=frames, joint_positions_rad=positions,
        T_world_base=transform(data.xmat[base].reshape(3,3), data.xpos[base]),
        table_top_m=float(model.geom_pos[table,2]+model.geom_size[table,2]),
        table_center_xy=model.geom_pos[table,:2].tolist(), table_half_size_xy=model.geom_size[table,:2].tolist(),
        calibration_verified=False, state_mapping_verified=False)
    validate_observation(observation)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, observation)
    print(f"Saved replay observation {observation['observation_id']} to {output}")


def record(config, output):
    from .adapters import AuroraRobot, capture_oak
    calibration = json.loads(Path(config["camera"]["calibration_file"]).read_text())
    if calibration.get("calibrated") is not True:
        raise ValueError("Fill measured OAK extrinsics and fixed-head reference before recording")
    robot = AuroraRobot(config["aurora"])
    try:
        before = robot.read_state()
        frame = capture_oak(config["camera"]["calibration_file"])
        state = robot.read_state()
        head = [state[n] for n in ("head_yaw_joint","head_pitch_joint")]
        if np.max(abs(np.array(head)-calibration["head_reference_rad"])) > .01:
            raise RuntimeError("Fixed-head calibration does not match current head pose")
        if any(abs(state[n]-before[n]) > .01 for n in state):
            raise RuntimeError("Robot moved during RGB-D capture; reacquire synchronized observation")
        scene = config["scene"]
        observation = dict(version=VERSION, observation_id=uuid.uuid4().hex,
            object_id=config["inference"]["object_id"], source="oak_live", frames=[asdict(frame)],
            joint_positions_rad=state, T_world_base=scene["T_world_base"],
            table_top_m=scene["table_top_m"], table_center_xy=scene["table_center_xy"],
            table_half_size_xy=scene["table_half_size_xy"],
            calibration_verified=bool(scene["calibration_verified"]),
            state_mapping_verified=bool(config["aurora"]["mapping_verified"]),
            recorded_unix_s=time.time())
        validate_observation(observation)
        output.parent.mkdir(parents=True, exist_ok=True)
        save(output, observation)
        print(f"Saved OAK/state observation to {output}; no motion commands sent")
    finally:
        robot.close()


def record_ros(config, output):
    """Head camera + joint state + TF over ROS 2; subscriptions only, no motion commands."""
    from .ros_adapter import RosRobot
    from .ros_camera import RosCamera, transform_change
    ros, camera_config, scene = config["ros"], config["ros_camera"], config["scene"]
    names = COMMAND_NAMES + list(ros["observation_extra_joints"])
    robot = RosRobot(ros)
    camera = None
    try:
        camera = RosCamera(camera_config)
        before, T_before = robot.read_state(), camera.base_camera()
        frame, capture = camera.capture(scene["T_world_base"])
        state, T_after = robot.read_state(), camera.base_camera()
        if any(abs(state[n]-before[n]) > .01 for n in names):
            raise RuntimeError("Robot moved during RGB-D capture; reacquire synchronized observation")
        moved_m, moved_rad = transform_change(T_before, T_after)
        if moved_m > camera_config["max_tf_change_m"] or moved_rad > camera_config["max_tf_change_rad"]:
            raise RuntimeError("Camera pose (TF) changed during capture; reacquire")
        observation = dict(version=VERSION, observation_id=uuid.uuid4().hex,
            object_id=config["inference"]["object_id"], source="ros_live", frames=[asdict(frame)],
            joint_positions_rad={n: state[n] for n in names}, T_world_base=scene["T_world_base"],
            table_top_m=scene["table_top_m"], table_center_xy=scene["table_center_xy"],
            table_half_size_xy=scene["table_half_size_xy"],
            calibration_verified=bool(scene["calibration_verified"] and camera_config["extrinsics_verified"]),
            state_mapping_verified=bool(ros["mapping_verified"]), recorded_unix_s=time.time(),
            capture=dict(capture, tf_change_m=moved_m, tf_change_rad=moved_rad))
        validate_observation(observation)
        output.parent.mkdir(parents=True, exist_ok=True)
        save(output, observation)
        print(json.dumps(dict(output=str(output), observation_id=observation["observation_id"],
            calibration_verified=observation["calibration_verified"],
            state_mapping_verified=observation["state_mapping_verified"], **observation["capture"],
            head_rad=[state["head_yaw_joint"], state["head_pitch_joint"]]),
            indent=2, default=lambda v: np.round(v, 5).tolist()))
        print("Saved ROS head-camera/state observation; no motion commands sent")
    finally:
        if camera:
            camera.close()
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/deployment_mvp.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("server", help="Server inference; never connects to robot motors")
    server.add_argument("--bind", default="tcp://127.0.0.1:5557")
    server.add_argument("--planner", choices=["dexonomy","transport-demo"], default="dexonomy")
    server.add_argument("--output", type=Path, required=True)
    client = sub.add_parser("client", help="Upload observation, save plan; optionally execute locally")
    client.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    client.add_argument("--observation", type=Path, required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--robot", choices=["mock","aurora","ros"], default="mock")
    client.add_argument("--execute", action="store_true")
    client.add_argument("--time-scale", type=float, default=1.)
    client.add_argument("--stop-file", type=Path)
    sample_parser = sub.add_parser("sample", help="Marked replay from saved simulation RGB-D/state")
    sample_parser.add_argument("--source", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    sample_parser.add_argument("--object-id", default="coffee_mug")
    capture = sub.add_parser("record", help="Read-only RGB-D + joint state capture (ROS head camera or OAK + Aurora)")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--source", choices=["ros", "oak"], default="ros")
    sub.add_parser("probe", help="Installed versions only; no SDK connection or motor access")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == "probe":
        from .adapters import probe
        print(json.dumps(probe(), indent=2))
        return
    if args.command == "sample":
        sample(args.source, args.output, args.object_id)
        return
    if args.command == "record":
        (record_ros if args.source == "ros" else record)(config, args.output)
        return
    token = os.environ.get("GR3_DEPLOY_TOKEN", "")
    if len(token) < 16:
        raise ValueError("Set GR3_DEPLOY_TOKEN to at least 16 characters on both hosts")
    from .transport import Client, Service
    if args.command == "server":
        from .inference import DexonomyInference, TransportDemo
        planner = DexonomyInference(config) if args.planner == "dexonomy" else TransportDemo()
        Service(planner, token, args.output).serve(args.bind)
        return
    obs = load(args.observation)
    validate_observation(obs)
    if args.robot != "mock":
        if obs.get("source") not in LIVE_SOURCES or time.time()-obs.get("recorded_unix_s",0) > config["execution"]["max_observation_age_s"]:
            raise ValueError("Hardware requires a fresh live camera/state observation")
    args.output.mkdir(parents=True, exist_ok=False)
    rpc = Client(args.endpoint, token, config["network"]["timeout_s"])
    result = rpc.call("plan", observation=obs)
    plan = result["plan"]
    validate_plan(plan, obs["observation_id"], config["execution"])
    if plan["observation_digest"] != digest(obs):
        raise ValueError("Plan is bound to another observation")
    save(args.output/"plan.msgpack", plan)
    write_json(args.output/"plan.json", plan)
    if not args.execute:
        print(f"Plan saved to {args.output}; no robot adapter opened, no commands sent")
        return
    from .adapters import MockRobot
    from .executor import execute
    robot = None
    try:
        # Check before SDK construction, not merely before the first command.
        if args.robot != "mock":
            if not plan["hardware_eligible"] or not config[args.robot]["motion_enabled"]:
                raise RuntimeError("Live execution disabled: verify calibration, RGB pose, SDK/hand mappings and local controller first")
            if time.time()-obs["recorded_unix_s"] > config["execution"]["max_observation_age_s"]:
                raise RuntimeError("Observation aged out during inference; reacquire and replan")
        if args.robot == "mock":
            robot = MockRobot(obs["joint_positions_rad"])
        elif args.robot == "ros":
            from .ros_adapter import RosRobot
            robot = RosRobot(config["ros"])
        else:
            from .adapters import AuroraRobot
            robot = AuroraRobot(config["aurora"])
        report = execute(plan, robot, config["execution"], args.output, obs["observation_id"],
                         args.execute, args.stop_file, args.time_scale)
    except Exception as error:
        report = dict(observation_id=obs["observation_id"], task_id=plan["task_id"], trajectory_completed=False,
            grasp_success=None, hardware=args.robot!="mock", error=f"{type(error).__name__}: {error}")
    finally:
        if robot:
            robot.close()
    write_json(args.output/"robot_report.json", report)
    try:
        rpc.call("report", report=report)
    except Exception as error:
        print(f"Report retained locally; upload failed: {error}")
    print(json.dumps(report, indent=2))
    if not report["trajectory_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
