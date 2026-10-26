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

from .protocol import VERSION, digest, load, save, validate_observation, validate_plan, write_json

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
    client.add_argument("--robot", choices=["mock","aurora"], default="mock")
    client.add_argument("--execute", action="store_true")
    client.add_argument("--time-scale", type=float, default=1.)
    client.add_argument("--stop-file", type=Path)
    sample_parser = sub.add_parser("sample", help="Marked replay from saved simulation RGB-D/state")
    sample_parser.add_argument("--source", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    sample_parser.add_argument("--object-id", default="coffee_mug")
    capture = sub.add_parser("record", help="OAK + Aurora read-only capture, with fixed head")
    capture.add_argument("--output", type=Path, required=True)
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
        record(config, args.output)
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
    if args.robot == "aurora":
        if obs.get("source") != "oak_live" or time.time()-obs.get("recorded_unix_s",0) > config["execution"]["max_observation_age_s"]:
            raise ValueError("Hardware requires a fresh OAK/state observation")
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
    from .adapters import AuroraRobot, MockRobot
    from .executor import execute
    robot = None
    try:
        # Check before SDK construction, not merely before the first command.
        if args.robot == "aurora":
            if not plan["hardware_eligible"] or not config["aurora"]["motion_enabled"]:
                raise RuntimeError("Live execution disabled: verify calibration, RGB pose, SDK/hand mappings and local controller first")
            if time.time()-obs["recorded_unix_s"] > config["execution"]["max_observation_age_s"]:
                raise RuntimeError("Observation aged out during inference; reacquire and replan")
        robot = MockRobot(obs["joint_positions_rad"]) if args.robot == "mock" else AuroraRobot(config["aurora"])
        report = execute(plan, robot, config["execution"], args.output, obs["observation_id"],
                         args.execute, args.stop_file, args.time_scale)
    except Exception as error:
        report = dict(observation_id=obs["observation_id"], task_id=plan["task_id"], trajectory_completed=False,
            grasp_success=None, hardware=args.robot=="aurora", error=f"{type(error).__name__}: {error}")
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
