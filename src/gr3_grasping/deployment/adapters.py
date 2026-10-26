"""Robot-local adapters. Imports of vendor modules occur only when selected."""
import importlib
import importlib.metadata
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from .protocol import ARM_NAMES, HAND_NAMES, COMMAND_NAMES


class MockRobot:
    """Kinematic command tracking only; never evidence of physical grasp success."""
    hardware = False

    def __init__(self, positions):
        self.positions = dict(positions)
        self.command_count = 0

    def read_state(self):
        return dict(self.positions)

    def prepare_motion(self):
        pass

    def validate_plan(self, plan):
        pass

    def send(self, q):
        self.positions.update(zip(COMMAND_NAMES, np.asarray(q).tolist(), strict=True))
        self.command_count += 1

    def hold(self):
        pass

    def close(self):
        pass


class Fdh6DDS:
    """Public EndEffector DDS contract; channel calibration must be supplied on site."""
    def __init__(self, config):
        self.config, self.latest, self.publisher = config, None, None
        if not config["mapping_verified"]:
            raise RuntimeError("FDH-6 mapping is unverified; fill channel order, scale, offset and state offset first")
        self.names = config["sdk_joint_order"]
        self.scale, self.offset = np.array(config["scale"], dtype=float), np.array(config["offset"], dtype=float)
        if set(self.names) != set(HAND_NAMES) or self.scale.shape != (6,) or self.offset.shape != (6,):
            raise ValueError("Expected six calibrated FDH-6 channels")
        if not np.isfinite(self.scale).all() or not np.isfinite(self.offset).all() or np.any(abs(self.scale) < 1e-8):
            raise ValueError("Invalid hand calibration")
        from fourierdds_py import DDSInterface, SubscriberQosProfile
        state_module = importlib.import_module("fourier_msgs_pod.msg.RobotJointState")
        self.dds = DDSInterface(domain_id=config["domain_id"])
        self.subscription = self.dds.create_subscription(state_module.EndEffectorStatePubSubType,
            "end_effector_state", self._state, qos_profile=SubscriberQosProfile.default())

    def _state(self, message):
        start = self.config["state_offset"]
        if message.state_size() >= start + 6:
            self.latest = (time.monotonic(), np.array([message.position()[i] for i in range(start, start+6)]))

    def read_state(self):
        deadline = time.monotonic() + 4
        while self.latest is None and time.monotonic() < deadline:
            time.sleep(.01)
        if self.latest is None or time.monotonic()-self.latest[0] > .5:
            raise RuntimeError("FDH-6 state unavailable or stale")
        q = (self.latest[1]-self.offset)/self.scale
        if not np.isfinite(q).all():
            raise RuntimeError("Invalid FDH-6 state")
        return dict(zip(self.names, q.tolist(), strict=True))

    def prepare_motion(self):
        from fourierdds_py import PublisherQosProfile
        self.command_module = importlib.import_module("fourier_msgs_pod.msg.RobotJointCmd")
        self.publisher = self.dds.create_publisher(self.command_module.EndEffectorGroupCmdPubSubType,
            "end_effector_cmd", qos_profile=PublisherQosProfile.default())
        time.sleep(3)  # DDS discovery, before starting the timed trajectory.

    def validate_command(self, positions):
        values = np.array([positions[n] for n in self.names])*self.scale + self.offset
        if not np.isfinite(values).all() or np.any(values < self.config["sdk_lower"]) or np.any(values > self.config["sdk_upper"]):
            raise RuntimeError("Mapped hand command outside verified SDK bounds")
        return values

    def send(self, positions):
        values = self.validate_command(positions)
        msg = self.command_module.EndEffectorGroupCmd()
        for i in (0, 1):
            chars = msg.group_names()[i].content()
            for j in range(len(chars)):
                chars[j] = 0
        for i, value in enumerate(self.config["group"].encode()):
            msg.group_names()[0].content()[i] = value
        for i, value in enumerate(values):
            cmd = msg.commands()[0]
            cmd.position()[i], cmd.velocity()[i], cmd.effort()[i] = float(value), 0., 0.
        self.publisher.publish(msg)


class AuroraRobot:
    """Existing controller state must be selected externally; never changes FSM or gains."""
    hardware = True

    def __init__(self, config):
        os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
        from fourier_aurora_client import AuroraClient
        self.config, self.hand, self.last, self.prepared = config, None, None, False
        self.client = AuroraClient.get_instance(domain_id=config["domain_id"], robot_name=config["robot_name"])
        time.sleep(1)

    def _hand(self):
        if self.hand is None:
            factory = self.config.get("hand_factory")
            if factory:
                module, name = factory.split(":", 1)
                self.hand = getattr(importlib.import_module(module), name)(self.config["hand"])
            else:
                self.hand = Fdh6DDS(self.config["hand"])
        return self.hand

    def read_state(self):
        result = {}
        for group, names in self.config["groups"].items():
            q = self.client.get_group_state(group, "position")
            if len(q) != len(names) or not np.isfinite(q).all():
                raise RuntimeError(f"Invalid measured group {group}")
            result.update(zip(names, [float(v) for v in q], strict=True))
        result.update(self._hand().read_state())
        return result

    def prepare_motion(self):
        if not self.config["motion_enabled"] or not self.config["mapping_verified"]:
            raise RuntimeError("Hardware motion is disabled or joint mapping unverified")
        expected = self.config["required_fsm_state"]
        if expected is None or self.client.get_fsm_state() != expected:
            raise RuntimeError("Robot is not in the locally verified control state")
        if set(self.config["groups"]["right_manipulator"]) != set(ARM_NAMES):
            raise RuntimeError("Invalid right arm mapping")
        self._hand().prepare_motion()
        self.prepared = True

    def validate_plan(self, plan):
        # Validate every SDK-mapped finger setpoint before any arm command.
        hand = self._hand()
        for stage in plan["stages"]:
            for q in stage["positions_rad"]:
                hand.validate_command(dict(zip(COMMAND_NAMES, q, strict=True)))

    def send(self, q):
        if not self.prepared or not self.config["motion_enabled"] or not self.config["mapping_verified"]:
            raise RuntimeError("Hardware motion disabled")
        if self.client.get_fsm_state() != self.config["required_fsm_state"]:
            raise RuntimeError("Controller state changed")
        positions = dict(zip(COMMAND_NAMES, np.asarray(q).tolist(), strict=True))
        self.hand.validate_command(positions)
        self.client.set_group_cmd(position_cmd={"right_manipulator": [
            positions[n] for n in self.config["groups"]["right_manipulator"]]})
        self.hand.send(positions)
        self.last = positions

    def hold(self):
        # Keep existing hand target; never open/drop an object on an exception.
        if self.last is not None:
            q = self.read_state()
            self.client.set_group_cmd(position_cmd={"right_manipulator": [
                q[n] for n in self.config["groups"]["right_manipulator"]]})

    def close(self):
        self.client.close()


def capture_oak(calibration_path, timeout_s=25):
    """Reuse the aligned DepthAI 2.x capture; no camera probing on the inference server."""
    from gr3_grasping.geometry import RGBDFrame
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="gr3_oak_") as directory:
        path = Path(directory)/"frame.npz"
        subprocess.run([sys.executable, str(root/"scripts/capture_oak.py"), "--extrinsics",
            str(calibration_path), "--output", str(path), "--timeout-s", str(timeout_s-2)],
            check=True, timeout=timeout_s)
        return RGBDFrame.load(path)


def probe():
    """Package/device information only: no SDK connection, publisher or motion."""
    versions = {}
    for name in ("fourier-aurora-client", "depthai", "pyzmq", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, packages=versions, vendor_modules={n: importlib.util.find_spec(n) is not None
        for n in ("fourierdds_py", "fourier_msgs_pod")}, motion_attempted=False)
