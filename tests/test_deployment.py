"""Numeric wire protocol, real TCP exchange and robot-local execution guards."""
import json
import socket
import sys
import threading
import time

import msgpack
import numpy as np
import pytest
import yaml

from gr3_grasping.deployment import protocol as p
from gr3_grasping.deployment.adapters import MockRobot, AuroraRobot, Fdh6DDS
from gr3_grasping.deployment.executor import execute
from gr3_grasping.deployment.inference import TransportDemo
from gr3_grasping.deployment.transport import Client, Service
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-token-not-for-deployment"


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
def policy():
    return yaml.safe_load((ROOT/"configs/deployment_mvp.yaml").read_text())["execution"]


@pytest.fixture
def plan(observation, tmp_path):
    return TransportDemo().plan(observation, tmp_path)


@pytest.fixture
def tcp_service(tmp_path):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1",0))
        endpoint = f"tcp://127.0.0.1:{socket_.getsockname()[1]}"
    stop, ready = threading.Event(), threading.Event()
    service = Service(TransportDemo(), TOKEN, tmp_path/"server")
    thread = threading.Thread(target=service.serve, args=(endpoint,stop,ready))
    thread.start()
    assert ready.wait(3)
    yield Client(endpoint,TOKEN,2), tmp_path/"server", endpoint
    stop.set()
    thread.join(3)
    assert not thread.is_alive()


def test_numeric_codec_and_invalid_array_payload(observation):
    decoded = p.unpack(p.pack(observation))
    np.testing.assert_array_equal(decoded["frames"][0]["depth_m"], observation["frames"][0]["depth_m"])
    p.validate_observation(decoded)
    with pytest.raises(ValueError):
        p.pack(np.array([object()], dtype=object))
    malicious = msgpack.packb(dict(__array__=True, dtype="O", shape=[1], data=b"payload"), use_bin_type=True)
    with pytest.raises(ValueError):
        p.unpack(malicious)
    malformed = msgpack.packb(dict(__array__=True, dtype="<f8", shape=[2], data=b"x"), use_bin_type=True)
    with pytest.raises(ValueError):
        p.unpack(malformed)


def test_tcp_observation_plan_execution_report_roundtrip(tcp_service, observation, policy, tmp_path):
    client, output, _ = tcp_service
    assert client.call("health")["hardware_control"] is False
    planned = client.call("plan", observation=observation)["plan"]
    assert planned["observation_digest"] == p.digest(observation)
    robot = MockRobot(observation["joint_positions_rad"])
    report = execute(planned,robot,policy,tmp_path/"robot",observation["observation_id"],time_scale=.02)
    client.call("report",report=report)
    assert robot.command_count > 0 and report["trajectory_completed"]
    assert report["grasp_success"] is None and not report["hardware"]
    assert json.loads((output/observation["observation_id"]/"robot_report.json").read_text()) == report
    with pytest.raises(RuntimeError, match="FileExistsError"):
        client.call("plan",observation=observation)
    count = robot.command_count
    with pytest.raises(FileExistsError):
        execute(planned,robot,policy,tmp_path/"robot",observation["observation_id"])
    assert robot.command_count == count


def test_authentication_and_timeout_leave_server_usable(tcp_service):
    client, _, endpoint = tcp_service
    with pytest.raises(RuntimeError,match="Authentication"):
        Client(endpoint,"wrong",1).call("health")
    assert client.call("health")["ok"]
    with socket.socket() as port:
        port.bind(("127.0.0.1",0))
        absent = f"tcp://127.0.0.1:{port.getsockname()[1]}"
        with pytest.raises(TimeoutError):
            Client(absent,TOKEN,.05).call("health")


@pytest.mark.parametrize("fault", ["id", "bounds", "speed", "nan", "discontinuity", "phases"])
def test_bad_plan_rejected_before_any_command(plan,observation,policy,tmp_path,fault):
    if fault == "id":
        plan["observation_id"] = "wrong"
    elif fault == "bounds":
        plan["stages"][0]["positions_rad"][0,0] = 100
    elif fault == "speed":
        plan["stages"][0]["positions_rad"][1,0] += .2
    elif fault == "nan":
        plan["stages"][0]["positions_rad"][0,0] = np.nan
    elif fault == "discontinuity":
        plan["stages"][1]["positions_rad"][:,0] += .1
    elif fault == "phases":
        plan["stages"] = plan["stages"][1:]
    robot = MockRobot(observation["joint_positions_rad"])
    with pytest.raises(ValueError):
        execute(plan,robot,policy,tmp_path,observation["observation_id"])
    assert robot.command_count == 0


def test_start_state_changed_and_unverified_hardware_never_command(plan,observation,policy,tmp_path):
    robot = MockRobot(observation["joint_positions_rad"])
    robot.positions[p.ARM_NAMES[0]] += .03
    with pytest.raises(RuntimeError,match="start state"):
        execute(plan,robot,policy,tmp_path/"changed",observation["observation_id"])
    assert robot.command_count == 0
    robot.hardware = True
    with pytest.raises(RuntimeError,match="disabled"):
        execute(plan,robot,policy,tmp_path/"hardware",observation["observation_id"],execute_motion=True)
    assert robot.command_count == 0


def test_local_stop_and_feedback_failure_hold_without_opening_hand(plan,observation,policy,tmp_path):
    class FeedbackFailure(MockRobot):
        held = False
        def hold(self): self.held = True
        def send(self,q):
            super().send(q)
            self.positions[p.ARM_NAMES[0]] += .3
    robot = FeedbackFailure(observation["joint_positions_rad"])
    with pytest.raises(RuntimeError,match="tracking"):
        execute(plan,robot,policy,tmp_path/"tracking",observation["observation_id"],time_scale=.02)
    assert robot.held
    np.testing.assert_allclose([robot.positions[n] for n in p.HAND_NAMES],
                               [observation["joint_positions_rad"][n] for n in p.HAND_NAMES])
    stop = tmp_path/"stop"
    stop.touch()
    robot = MockRobot(observation["joint_positions_rad"])
    with pytest.raises(RuntimeError,match="stop file"):
        execute(plan,robot,policy,tmp_path/"stop_trial",observation["observation_id"],stop_file=stop)
    assert robot.command_count == 0


def test_all_hand_setpoints_preflight_before_arm_commands(plan):
    hand = Fdh6DDS.__new__(Fdh6DDS)
    hand.names = p.HAND_NAMES
    hand.scale, hand.offset = np.ones(6), np.zeros(6)
    hand.config = dict(sdk_lower=[-2]*6,sdk_upper=[2]*6)
    robot = AuroraRobot.__new__(AuroraRobot)
    robot.hand = hand
    robot.validate_plan(plan)
    hand.config["sdk_lower"] = [0]*6  # Incorrect calibration of negative URDF bends.
    with pytest.raises(RuntimeError,match="SDK bounds"):
        robot.validate_plan(plan)


def test_cli_unverified_live_plan_does_not_open_sdk(tcp_service,observation,tmp_path,monkeypatch):
    _, _, endpoint = tcp_service
    import gr3_grasping.deployment.adapters as adapters
    from gr3_grasping.deployment.__main__ import main
    def forbidden(*args,**kwargs):
        raise AssertionError("Vendor SDK must not be opened")
    monkeypatch.setattr(adapters,"AuroraRobot",forbidden)
    monkeypatch.setenv("GR3_DEPLOY_TOKEN",TOKEN)
    observation.update(source="oak_live",recorded_unix_s=time.time(),calibration_verified=True,state_mapping_verified=True)
    file = tmp_path/"live.msgpack"
    p.save(file,observation)
    output = tmp_path/"client"
    monkeypatch.setattr(sys,"argv",["deployment","client","--endpoint",endpoint,"--observation",str(file),
        "--output",str(output),"--robot","aurora","--execute"])
    with pytest.raises(SystemExit) as status:
        main()
    assert status.value.code == 1
    report = json.loads((output/"robot_report.json").read_text())
    assert not report["trajectory_completed"] and "Live execution disabled" in report["error"]


def test_measured_geometry_is_recompiled_for_collision_queries(observation,tmp_path):
    import mujoco
    from gr3_grasping.deployment.inference import configure_scene
    file = tmp_path/"twin.xml"
    file.write_text('''<mujoco><worldbody>
      <body name="base_link"><geom type="sphere" size=".03"/></body>
      <geom name="table" type="box" size=".1 .1 .025" pos="0 0 -.025"/>
      <body name="target" pos=".6 .1 .848"><freejoint/><geom type="sphere" size=".05"/></body>
    </worldbody></mujoco>''')
    observation.update(table_center_xy=[.6,.1],table_half_size_xy=[.2,.3],table_top_m=.8)
    observation["T_world_base"][0,3] = -1
    configure_scene(file,observation)
    model = mujoco.MjModel.from_xml_path(str(file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.xpos[model.body("base_link").id],[-1,0,0])
    assert any(model.geom("table").id in c.geom and c.dist < -.001 for c in data.contact)


def test_head_motion_invalidates_frozen_planning_context(plan,observation,policy,tmp_path):
    robot = MockRobot(observation["joint_positions_rad"])
    robot.positions["head_pitch_joint"] += .1
    with pytest.raises(RuntimeError,match="Frozen body/head"):
        execute(plan,robot,policy,tmp_path,observation["observation_id"])
    assert robot.command_count == 0


def test_aurora_cannot_send_before_preparation(observation):
    robot = AuroraRobot.__new__(AuroraRobot)
    robot.prepared = False
    robot.config = dict(motion_enabled=True,mapping_verified=True)
    with pytest.raises(RuntimeError,match="disabled"):
        robot.send([observation["joint_positions_rad"][n] for n in p.COMMAND_NAMES])


def test_hardware_task_ledger_survives_different_output_directory(plan,observation,policy,tmp_path):
    robot = MockRobot(observation["joint_positions_rad"])
    robot.hardware = True  # Test double, no SDK or device access.
    plan["hardware_eligible"] = True
    policy["task_ledger_dir"] = str(tmp_path/"persistent_ledger")
    result = execute(plan,robot,policy,tmp_path/"first",observation["observation_id"],execute_motion=True)
    assert result["trajectory_completed"]
    count = robot.command_count
    with pytest.raises(FileExistsError):
        execute(plan,robot,policy,tmp_path/"second",observation["observation_id"],execute_motion=True)
    assert robot.command_count == count
