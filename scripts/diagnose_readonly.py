#!/usr/bin/env python3
"""Read-only site diagnostics for the RealSense camera and robot state.

Needs no calibration. Never creates a publisher, action client or service client, never sends a
command, never changes FSM state, modules or motor gains. AuroraClient is deliberately not used:
its constructor creates eight command publishers and a service client.

The sections are independent so each can run where its SDK works:
  camera  pyrealsense2, on the machine the RealSense is plugged into;
  ros     fourier_control ROS 2 bridge (/joint_states, /module_states, graph). Source ROS 2 and
          the fourier_control install first; ROS_DOMAIN_ID/RMW/CYCLONEDDS_URI come from the env;
  aurora  fourier_aurora_client DDSInterface subscribers. The PyPI wheel bundles x86-64
          FastDDS libraries, so on aarch64 this section only works with a native build.
Optional sampling (--sample-s) records joint positions so joints can be moved by hand
(or by the vendor tools) and the moving channel and sign identified afterwards.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

AURORA_TOPICS = ("aurora_state", "robot_control_group_state", "robot_motor_cfg_group_state",
                 "robot_cartesian_state")


def intrinsics_dict(intr):
    return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, ppx=intr.ppx,
                ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs))


def depth_stats(depth, scale):
    valid = depth > 0
    meters = depth[valid].astype(float)*scale
    stats = dict(shape=list(depth.shape), dtype=str(depth.dtype), valid_ratio=float(valid.mean()))
    if meters.size:
        stats.update(zip(("p01_m", "p50_m", "p99_m"), np.percentile(meters, [1, 50, 99]).round(4).tolist()))
    return stats


def camera(out, args):
    import pyrealsense2 as rs
    rs_info = rs.camera_info
    devices = []
    for device in rs.context().query_devices():
        info = {name: device.get_info(key) for name, key in (("name", rs_info.name),
            ("serial", rs_info.serial_number), ("firmware", rs_info.firmware_version),
            ("usb_type", rs_info.usb_type_descriptor), ("product_line", rs_info.product_line))
            if device.supports(key)}
        info["sensors"] = [s.get_info(rs_info.name) for s in device.query_sensors()]
        devices.append(info)
    result = dict(sdk_version=rs.__version__ if hasattr(rs, "__version__") else None, devices=devices)
    if not devices:
        return dict(result, error="No RealSense device found")
    serial = args.serial or devices[0]["serial"]
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        scale = depth_sensor.get_depth_scale()
        for _ in range(args.warmup):  # Auto exposure settles over the first frames.
            pipeline.wait_for_frames(5000)
        frames = pipeline.wait_for_frames(5000)
        aligned = rs.align(rs.stream.color).process(frames)
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        depth_aligned = np.asanyarray(aligned.get_depth_frame().get_data()).copy()
        color_profile = color_frame.get_profile().as_video_stream_profile()
        depth_profile = depth_frame.get_profile().as_video_stream_profile()
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        options = {name: depth_sensor.get_option(option) for name, option in (
            ("emitter_enabled", rs.option.emitter_enabled), ("laser_power", rs.option.laser_power),
            ("visual_preset", rs.option.visual_preset)) if depth_sensor.supports(option)}
    finally:
        pipeline.stop()
    color_K = intrinsics_dict(color_profile.get_intrinsics())
    np.savez_compressed(out/"camera_frame.npz", rgb=color, depth_raw=depth, depth_aligned_to_color=depth_aligned,
        depth_scale_m=scale, color_K=np.array([[color_K["fx"], 0, color_K["ppx"]], [0, color_K["fy"], color_K["ppy"]], [0, 0, 1]]))
    try:
        import cv2
        cv2.imwrite(str(out/"camera_rgb.png"), color[..., ::-1])
        preview = cv2.applyColorMap(cv2.convertScaleAbs(depth_aligned, alpha=255/max(1, int(2.0/scale))), cv2.COLORMAP_JET)
        preview[depth_aligned == 0] = 0
        cv2.imwrite(str(out/"camera_depth_aligned.png"), preview)
    except ImportError:
        pass
    return dict(result, serial=serial, depth_scale_m=scale, depth_units="uint16 * depth_scale_m = meters",
        color_intrinsics=color_K, depth_intrinsics=intrinsics_dict(depth_profile.get_intrinsics()),
        depth_to_color=dict(rotation_col_major=list(extrinsics.rotation), translation_m=list(extrinsics.translation)),
        depth_options=options, depth_raw=depth_stats(depth, scale),
        depth_aligned_to_color=dict(depth_stats(depth_aligned, scale), aligned=True,
            note="Aligned with rs.align(color); depth_raw is in the depth camera frame"))


def parse_aurora(topic, data, names):
    if topic == "aurora_state":
        return dict(whole_body_fsm_state=int(data.whole_body_fsm_state()),
            whole_body_fsm_name=names["whole"].get(int(data.whole_body_fsm_state())),
            upper_body_fsm_state=int(data.upper_body_fsm_state()),
            upper_body_fsm_name=names["upper"].get(int(data.upper_body_fsm_state())),
            velocity_command_source=int(data.velocity_command_source()),
            allow_upper_body_override=bool(data.allow_upper_body_override()))
    if topic == "robot_control_group_state":
        return dict(group_motion_state=[int(v) for v in data.group_motion_state()],
            groups={g.group_name(): dict(position=list(g.joint_position()), velocity=list(g.joint_velocity()),
                effort=list(g.joint_effort())) for g in data.group_state()})
    if topic == "robot_motor_cfg_group_state":
        return {g.group_name(): dict(mode=int(g.mode()), mode_name=names["motor"].get(int(g.mode())),
            pd_kp=list(g.pd_kp()), pd_kd=list(g.pd_kd()), pos_kp=list(g.pos_kp()), pos_ki=list(g.pos_ki()),
            pos_kd=list(g.pos_kd())) for g in data.group_state()}
    return {g.group_name(): dict(pose=list(g.pose()), twist=list(g.twist()), wrench=list(g.wrench()))
            for g in data.cartesian_state()}


def aurora(out, args):
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    try:
        from fourier_aurora_client import DDSInterface, SubscriberQosProfile
        from fourier_aurora_client.aurora_state_map import WholeBodyFsmState, UpperBodyFsmState, MotorMode
        import fourier_msgs.msg.AuroraState as AuroraState
        import fourier_msgs.msg.MotionControlState as MotionControlState
        import fourier_msgs.msg.MotorCfgState as MotorCfgState
    except (ImportError, OSError) as exc:
        return dict(error=f"{type(exc).__name__}: {exc}", hint="The PyPI fourier_aurora_client bundles x86-64 "
            "FastDDS; run this section on the x86 robot host or with an aarch64 FastDDS build")
    names = dict(whole={s.value: s.name for s in WholeBodyFsmState}, upper={s.value: s.name for s in UpperBodyFsmState},
                 motor={s.value: s.name for s in MotorMode})
    types = dict(aurora_state=(AuroraState.AuroraStatePubSubType, AuroraState.AuroraState),
        robot_control_group_state=(MotionControlState.RobotControlGroupStatePubSubType, MotionControlState.RobotControlGroupState),
        robot_motor_cfg_group_state=(MotorCfgState.RobotMotorCfgGroupStatePubSubType, MotorCfgState.RobotMotorCfgGroupState),
        robot_cartesian_state=(MotionControlState.RobotCartesianStatePubSubType, MotionControlState.RobotCartesianState))
    latest, counts, samples, errors = {}, dict.fromkeys(AURORA_TOPICS, 0), [], {}

    def callback(topic):
        def receive(data):
            try:
                latest[topic] = (time.monotonic(), parse_aurora(topic, data, names))
                counts[topic] += 1
            except Exception as exc:  # Keep the DDS listener thread alive; report the first parse error.
                errors.setdefault(topic, f"{type(exc).__name__}: {exc}")
        return receive

    dds = DDSInterface(args.domain_id)
    try:
        for topic in AURORA_TOPICS:
            pub_sub, data_type = types[topic]
            dds.create_subscriber(topic, pub_sub(), data_type(), callback(topic), SubscriberQosProfile.best_effort())
        deadline = time.monotonic() + args.match_timeout_s
        while time.monotonic() < deadline and not all(t in latest for t in AURORA_TOPICS):
            time.sleep(.05)
        snapshot = {topic: latest[topic][1] for topic in AURORA_TOPICS if topic in latest}
        if args.sample_s > 0 and "robot_control_group_state" in latest:
            print(f"Sampling joint state for {args.sample_s:.1f} s; move joints now ...", flush=True)
            end, period = time.monotonic() + args.sample_s, 1/args.sample_hz
            while time.monotonic() < end:
                stamp, state = latest["robot_control_group_state"]
                samples.append((stamp, {g: v["position"] for g, v in state["groups"].items()}))
                time.sleep(period)
    finally:
        dds.close()
    result = dict(domain_id=args.domain_id, messages_received=counts, missing_topics=[t for t in AURORA_TOPICS
        if t not in latest], parse_errors=errors, snapshot=snapshot, expected_names_unverified=expected_names(args.config))
    if samples:
        result["sampling"] = summarize_samples(out, samples)
    return result


def ros(out, args):
    import rclpy
    from rclpy.action import get_action_server_names_and_types_by_node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    env = {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI")}
    rclpy.init()
    node_name = "gr3_diagnose_readonly"
    # No rosout publisher or parameter services. rclpy always adds a /parameter_events publisher,
    # which only publishes when this node's own parameters change (never here).
    node = rclpy.create_node(node_name, enable_rosout=False, start_parameter_services=False)
    latest, streams, modules = {}, {}, {}

    def joint_states(message):
        now = time.monotonic()
        for i, name in enumerate(message.name):
            latest[name] = (now, message.position[i], message.velocity[i] if i < len(message.velocity) else None,
                            message.effort[i] if i < len(message.effort) else None)
        stream = streams.setdefault(tuple(message.name), [now, now, 0])
        stream[1], stream[2] = now, stream[2]+1

    try:
        node.create_subscription(JointState, args.joint_states_topic, joint_states, qos_profile_sensor_data)
        try:
            from fourier_control_msgs.msg import ModuleState
            node.create_subscription(ModuleState, "/module_states",
                lambda m: modules.update(zip(m.name, [int(s) for s in m.state])), 10)
        except ImportError as exc:
            modules["error"] = f"fourier_control_msgs not importable ({exc}); source the fourier_control install"
        end = time.monotonic() + args.ros_listen_s
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.02)
        samples = []
        if args.sample_s > 0 and latest:
            print(f"Sampling joint state for {args.sample_s:.1f} s; move joints now ...", flush=True)
            end, next_sample = time.monotonic() + args.sample_s, time.monotonic()
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=.005)
                if time.monotonic() >= next_sample:
                    samples.append((time.monotonic(), {n: v[1] for n, v in latest.items()}))
                    next_sample += 1/args.sample_hz
        nodes = node.get_node_names_and_namespaces()
        graph = dict(nodes=sorted(f"{ns.rstrip('/')}/{n}" for n, ns in nodes),
            topics={t: types for t, types in node.get_topic_names_and_types()},
            action_servers={f"{ns.rstrip('/')}/{n}": dict(get_action_server_names_and_types_by_node(node, n, ns))
                            for n, ns in nodes if n != node_name},
            joint_states_publishers=[i.node_name for i in node.get_publishers_info_by_topic(args.joint_states_topic)])
        own_publishers = node.get_publisher_names_and_types_by_node(node_name, "/")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    rates = [dict(joints=len(names), first=names[0] if names else None, messages=count,
                  rate_hz=round((count-1)/(t1-t0), 1) if t1 > t0 else None) for names, (t0, t1, count) in streams.items()]
    result = dict(env=env, graph=graph, own_publishers=own_publishers, module_states=modules,
        module_state_codes={0: "DISABLED", 1: "ENABLED", 2: "FAILED", 3: "IDLE"}, joint_state_streams=rates,
        joints={n: dict(position=v[1], velocity=v[2], effort=v[3], age_s=round(time.monotonic()-v[0], 3))
                for n, v in latest.items()},
        expected_names_unverified=expected_ros_names(args.config, latest))
    if samples:
        result["sampling"] = summarize_ros_samples(out, samples)
    return result


def expected_ros_names(config_path, latest):
    try:
        import yaml
        ros_config = yaml.safe_load(Path(config_path).read_text())["ros"]
    except (OSError, KeyError, ImportError) as exc:
        return dict(error=str(exc))
    hand = list(ros_config["hand_gripper"]["closed_state"])
    return dict(arm=ros_config["arm_joint_names"], hand=hand,
                missing=[n for n in ros_config["arm_joint_names"] + hand if n not in latest])


def summarize_ros_samples(out, samples):
    names = sorted({n for _, values in samples for n in values})
    t = np.array([s[0] for s in samples])
    q = np.array([[values.get(n, np.nan) for n in names] for _, values in samples], dtype=float)
    np.savez_compressed(out/"ros_samples.npz", t=t-t[0], q=q, names=np.array(names))
    span, change = np.nanmax(q, 0)-np.nanmin(q, 0), q[-1]-q[0]
    moved = {n: dict(range=round(float(r), 4), net_change=round(float(c), 4))
             for n, r, c in zip(names, span, change) if r > .01}
    return dict(file="ros_samples.npz", count=len(samples), duration_s=float(t[-1]-t[0]), moved_over_0_01_rad=moved)


def expected_names(config_path):
    """Joint names the config assumes for each group, for side-by-side comparison only."""
    try:
        import yaml
        aurora_config = yaml.safe_load(Path(config_path).read_text())["aurora"]
    except (OSError, KeyError, ImportError) as exc:
        return dict(error=str(exc))
    return dict(aurora_config["groups"], right_hand_sdk_order=aurora_config["hand"]["sdk_joint_order"])


def summarize_samples(out, samples):
    t = np.array([s[0] for s in samples])
    arrays = {}
    for group in samples[0][1]:
        rows = [s[1].get(group) for s in samples]
        if all(r is not None and len(r) == len(rows[0]) for r in rows):
            arrays[group] = np.array(rows, dtype=float)
    np.savez_compressed(out/"aurora_samples.npz", t=t-t[0], **arrays)
    return dict(file="aurora_samples.npz", count=len(samples), duration_s=float(t[-1]-t[0]),
        per_group={g: dict(range=(q.max(0)-q.min(0)).round(4).tolist(), net_change=(q[-1]-q[0]).round(4).tolist())
                   for g, q in arrays.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", choices=("camera", "ros", "aurora"), default=["camera", "ros"])
    parser.add_argument("--output", type=Path, default=None, help="Default: outputs/diagnose/<timestamp>")
    parser.add_argument("--config", type=Path, default=Path("configs/deployment_mvp.yaml"))
    parser.add_argument("--serial", help="RealSense serial; default first device")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--domain-id", type=int, default=123)
    parser.add_argument("--match-timeout-s", type=float, default=8)
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--ros-listen-s", type=float, default=3)
    parser.add_argument("--sample-s", type=float, default=0, help="Record joint positions this long")
    parser.add_argument("--sample-hz", type=float, default=20)
    args = parser.parse_args()
    out = args.output or Path("outputs/diagnose")/time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report = dict(host=platform.node(), machine=platform.machine(), python=sys.version.split()[0],
                  time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), read_only=True, command_publishers_created=0)
    for section in args.sections:
        try:
            report[section] = dict(camera=camera, ros=ros, aurora=aurora)[section](out, args)
        except Exception as exc:
            report[section] = dict(error=f"{type(exc).__name__}: {exc}")
    (out/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
