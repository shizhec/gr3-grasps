"""ROS 2 bridge adapter logic and stage-executing adapters in the executor; no ROS required."""
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from gr3_grasping.deployment import protocol as p
from gr3_grasping.deployment.adapters import MockRobot
from gr3_grasping.deployment.executor import execute
from gr3_grasping.deployment.inference import TransportDemo
from gr3_grasping.deployment.ros_adapter import GripperModel, JointStateCache, RosRobot, chunks, resample

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT/"configs/deployment_mvp.yaml").read_text())


@pytest.fixture
def observation():
    positions = dict(zip(p.COMMAND_NAMES, [-.1,0,0,-1.7,0,0,0,-1.4,.1,-.1,-.1,-.1,-.1]))
    positions.update(head_yaw_joint=0., head_pitch_joint=.14)
    return dict(version=1, observation_id="test_observation", object_id="coffee_mug", source="test",
        joint_positions_rad=positions, T_world_base=np.eye(4), table_top_m=1., table_center_xy=[.6,0],
        table_half_size_xy=[.3,.4], calibration_verified=False, state_mapping_verified=False,
        frames=[dict(rgb=np.zeros((48,64,3),np.uint8), depth_m=np.ones((48,64),np.float32),
            K=np.array([[100.,0,32],[0,100.,24],[0,0,1]]), T_world_camera=np.eye(4), timestamp_s=1., source="test")])


@pytest.fixture
def plan(observation, tmp_path):
    return TransportDemo().plan(observation, tmp_path)


def test_resample_matches_bridge_tick_and_chunks_cover_stage():
    t, q = resample([0., .5, 1.234], np.array([[0.], [1.], [2.]]), 100)
    assert len(t) == 125 and t[0] == 0 and t[-1] == pytest.approx(1.234)
    np.testing.assert_allclose(np.diff(t)[:-1], .01)  # The bridge ignores time_from_start.
    assert q[50, 0] == pytest.approx(1.) and q[-1, 0] == pytest.approx(2.)
    spans = chunks(len(t), 50)
    assert spans[0] == (0, 50) and spans[-1] == (100, 125)
    assert sum(b-a for a, b in spans) == len(t)


def test_gripper_model_reproduces_gripper_shapes_and_rejects_others(plan):
    model = GripperModel(CONFIG["ros"]["hand_gripper"])
    command, shape = model.fit(dict.fromkeys(p.HAND_NAMES, 0.))
    assert command == pytest.approx(0.) and model.error(shape) == pytest.approx(0, abs=1e-9)
    half = {n: v/2 for n, v in model.closed.items()}
    half["R_thumb_proximal_yaw_joint"] = model.closed["R_thumb_proximal_yaw_joint"]
    command, shape = model.fit(half)
    assert command == pytest.approx(-.8) and model.error(half) == pytest.approx(0, abs=1e-9)
    # Thumb yaw is fixed by hand_control; a different yaw is not reproducible.
    robot = ros_robot(hand_mode="fit")
    with pytest.raises(RuntimeError, match="GripperCommand"):
        robot.validate_plan(plan)  # Test plan holds thumb yaw at -0.1, the gripper forces -1.6.
    for stage in plan["stages"]:
        stage["positions_rad"][:, 7:] = [shape[n] for n in p.HAND_NAMES]
    robot.validate_plan(plan)


def test_joint_state_cache_drops_stale_values():
    cache = JointStateCache()
    cache.update(["a", "b"], [1., np.nan], stamp=10.)
    cache.update(["c"], [3.], stamp=11.)
    assert cache.fresh(.5, now=11.2) == {"c": 3.}
    assert cache.fresh(2., now=11.2) == {"a": 1., "c": 3.}


def test_ros_adapter_refuses_motion_and_streaming_without_enabled_module():
    robot = RosRobot.__new__(RosRobot)
    robot.config = dict(CONFIG["ros"], motion_enabled=True, mapping_verified=True, startup_timeout_s=.05)
    robot.modules = (time.monotonic(), {"right_arm_trajectory": 0})
    with pytest.raises(RuntimeError, match="never publishes /enable_modules"):
        robot.prepare_motion()
    robot.modules = None
    with pytest.raises(RuntimeError, match="unavailable"):
        robot.prepare_motion()
    robot.prepared = False
    with pytest.raises(RuntimeError, match="not prepared"):
        robot.execute_stage(dict(times_s=[0., .1], positions_rad=np.zeros((2, 13))), lambda q: None)
    with pytest.raises(RuntimeError, match="whole stages"):
        robot.send(np.zeros(13))


class StageRobot(MockRobot):
    """Executes stages itself, like RosRobot, with fingers lagging (untracked) behind commands."""
    tracked_names = p.ARM_NAMES

    def __init__(self, positions, chunk=2):
        super().__init__(positions)
        self.chunk, self.chunks_sent, self.held = chunk, 0, False

    def execute_stage(self, stage, check):
        q = np.asarray(stage["positions_rad"])
        for start in range(0, len(q), self.chunk):
            check(q[start])
            self.chunks_sent += 1
            for row in q[start:start+self.chunk]:
                self.positions.update(zip(p.COMMAND_NAMES, row.tolist()))
                for name in p.HAND_NAMES[2:]:
                    self.positions[name] = row[p.COMMAND_NAMES.index(name)] - .3
                check(row)

    def send(self, q):
        raise AssertionError("stage adapters must not be streamed")

    def hold(self):
        self.held = True


def test_stage_adapter_tracks_arm_only_and_stop_file_halts_between_chunks(plan, observation, tmp_path):
    policy = CONFIG["execution"]
    robot = StageRobot(observation["joint_positions_rad"])
    report = execute(plan, robot, policy, tmp_path/"ok", observation["observation_id"])
    assert report["trajectory_completed"] and robot.chunks_sent == len(plan["stages"])
    robot = StageRobot(observation["joint_positions_rad"])
    robot.tracked_names = p.COMMAND_NAMES  # Lagging fingers fail once every joint is tracked.
    with pytest.raises(RuntimeError, match="tracking"):
        execute(plan, robot, policy, tmp_path/"all", observation["observation_id"])
    assert robot.held
    stop = tmp_path/"stop"
    stop.touch()
    robot = StageRobot(observation["joint_positions_rad"])
    with pytest.raises(RuntimeError, match="stop file"):
        execute(plan, robot, policy, tmp_path/"stop_trial", observation["observation_id"], stop_file=stop)
    assert robot.chunks_sent == 0 and robot.held


def ros_robot(**overrides):
    robot = RosRobot.__new__(RosRobot)
    robot.config = dict(CONFIG["ros"], **overrides)
    robot.gripper = GripperModel(robot.config["hand_gripper"])
    robot.hand_mode, robot.hand_command = robot.config["hand_mode"], None
    return robot


def test_open_close_mode_opens_before_contact_and_closes_from_grasp(plan):
    robot = ros_robot()
    commands = {s["name"]: robot.stage_hand_command(s) for s in plan["stages"]}
    assert commands == dict(preform=0., pregrasp=0., approach=0., grasp=-1.6, squeeze=-1.6, lift=-1.6, hold=-1.6)
    robot.validate_plan(plan)  # Planned finger shapes are ignored in this mode.
    with pytest.raises(RuntimeError, match="gripper range"):
        ros_robot(hand_open_close=dict(CONFIG["ros"]["hand_open_close"], close_command=-2.)).validate_plan(plan)
    plan["stages"][0]["name"] = "unknown"
    with pytest.raises(RuntimeError, match="no open/close"):
        robot.validate_plan(plan)


class Done:
    """Already-completed rclpy future stand-in."""
    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


def test_stage_execution_chunks_arm_goals_and_sends_hand_only_on_change(plan):
    pytest.importorskip("action_msgs")
    pytest.importorskip("control_msgs")
    from types import SimpleNamespace
    sent = dict(arm=[], hand=[])
    ok = SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))  # GoalStatus.STATUS_SUCCEEDED

    class Client:
        def __init__(self, kind):
            self.kind = kind

        def send_goal_async(self, goal, feedback_callback=None):
            sent[self.kind].append(goal)
            return Done(SimpleNamespace(accepted=True, get_result_async=lambda: Done(ok)))

    robot = ros_robot(chunk_s=.05)
    robot.prepared, robot.feedback, robot.active = True, None, None
    robot.arm_client, robot.hand_client = Client("arm"), Client("hand")
    checks = []
    for stage in plan["stages"]:
        robot.execute_stage(stage, checks.append)
    assert [g.command.position for g in sent["hand"]] == [0., -1.6]  # preform opens, grasp closes
    arm = [g.trajectory for g in sent["arm"]]
    assert all(t.joint_names == p.ARM_NAMES and 1 <= len(t.points) <= 5 for t in arm)
    expected = sum(len(resample(s["times_s"], s["positions_rad"], 100)[0]) for s in plan["stages"])
    assert sum(len(t.points) for t in arm) == expected and len(checks) >= len(arm)


def test_ros_image_decoding_and_frame_matching():
    from types import SimpleNamespace as NS
    from gr3_grasping.deployment.ros_camera import decode_color, decode_depth, matched_pair, transform_change
    rgb = np.arange(2*3*3, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.zeros((2, 12), np.uint8)
    padded[:, :9] = rgb.reshape(2, 9)
    color = NS(width=3, height=2, step=12, encoding="rgb8", data=padded.tobytes(), is_bigendian=0)
    np.testing.assert_array_equal(decode_color(color), rgb)
    np.testing.assert_array_equal(decode_color(NS(**{**vars(color), "encoding": "bgr8"})), rgb[..., ::-1])
    mm = np.array([[0, 1000], [1500, 65535]], np.uint16)
    depth = decode_depth(NS(width=2, height=2, step=4, encoding="16UC1", data=mm.tobytes(), is_bigendian=0), .001)
    np.testing.assert_allclose(depth, mm*.001, rtol=1e-6)
    big = decode_depth(NS(width=2, height=2, step=4, encoding="16UC1", data=mm.astype(">u2").tobytes(), is_bigendian=1), .001)
    np.testing.assert_allclose(big, depth)
    metres = np.array([[np.nan, 1.25]], np.float32)
    np.testing.assert_array_equal(decode_depth(NS(width=2, height=1, step=8, encoding="32FC1", data=metres.tobytes(),
        is_bigendian=0), .001), [[0, 1.25]])
    with pytest.raises(ValueError, match="encoding"):
        decode_color(NS(**{**vars(color), "encoding": "yuv422"}))
    msg = lambda t: NS(header=NS(stamp=NS(sec=int(t), nanosec=int(round((t % 1)*1e9)))))
    colors, depths = [msg(1.0), msg(1.1), msg(1.2)], [msg(1.0), msg(1.1003)]
    assert matched_pair(colors, depths, .005) == (colors[1], depths[1])
    assert matched_pair(colors[2:], depths, .005) is None
    T = np.eye(4)
    T[:3, 3] = [.003, 0, .004]
    assert transform_change(np.eye(4), T) == pytest.approx((.005, 0.))


def test_live_sources_include_ros_record():
    assert "ros_live" in p.LIVE_SOURCES and "oak_live" in p.LIVE_SOURCES


def test_observation_workspace_is_validated(observation):
    p.validate_observation(observation)  # Optional field.
    observation["perception_workspace"] = dict(workspace_min=[.39, -.08, .995], workspace_max=[.53, .11, 1.15])
    p.validate_observation(observation)
    for bad in (dict(workspace_min=[.5, 0, 1], workspace_max=[.4, 1, 2]), dict(workspace_min=[0, 0], workspace_max=[1, 1]),
                dict(workspace_min=[0, 0, np.nan], workspace_max=[1, 1, 1]), dict(workspace_min=[0, 0, 0], workspace_max=[1, 1, 1], extra=1)):
        observation["perception_workspace"] = bad
        with pytest.raises(ValueError, match="perception workspace"):
            p.validate_observation(observation)
