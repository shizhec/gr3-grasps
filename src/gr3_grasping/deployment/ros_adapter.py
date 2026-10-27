"""ROS 2 adapter for the fourier_control bridge (robot_control + hand_control on the robot host).

Construction and read_state() create subscriptions only (rosout and parameter services are
disabled; rclpy's built-in /parameter_events publisher stays idle). Action clients are created
in prepare_motion(), which requires motion_enabled, mapping_verified and the trajectory module
already enabled: this adapter never publishes /enable_modules and never changes the robot's
controller state.

Bridge behaviour this adapter is written against (fourier_grx_modules/joint_control.py and
fourier_grx_control/hand_control.py):
- FollowJointTrajectory sends one point to Aurora per robot_control tick and ignores
  time_from_start, so stages are resampled at exactly that tick rate.
- Goals cannot be cancelled and a new goal silently replaces an active one, so stages go out
  as short chunks, each awaited; checks between chunks bound the stop latency to one chunk.
- The hand is a GripperCommand: one scalar for the four fingers and thumb pitch, thumb yaw
  always at its closed_state. hand_mode "open_close" ignores the planned finger shapes and opens
  the hand for the stages before contact and closes it from "grasp" on (an approximation of the
  Dexonomy grasp); hand_mode "fit" rejects plans whose hand shapes the gripper cannot reproduce.
"""
import threading
import time

import numpy as np

from .protocol import ARM_NAMES, HAND_NAMES, COMMAND_NAMES

FINGERS = HAND_NAMES[2:]
THUMB_PITCH, THUMB_YAW = "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint"
MODULE_ENABLED = 1  # fourier_control_msgs/ModuleState.ENABLED


def resample(times, positions, rate_hz):
    """Points spaced exactly 1/rate_hz apart, ending on the stage's last point."""
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    count = int(np.ceil(times[-1]*rate_hz - 1e-9)) + 1
    sample_t = np.minimum(np.arange(count)/rate_hz, times[-1])
    return sample_t, np.stack([np.interp(sample_t, times, positions[:, j]) for j in range(positions.shape[1])], 1)


def chunks(count, size):
    return [(start, min(start+size, count)) for start in range(0, count, size)]


class GripperModel:
    """hand_control maps SDK p (1 open, 0 closed) to joint radians as closed_state * (1 - p)."""
    def __init__(self, config):
        self.closed = {name: float(value) for name, value in config["closed_state"].items()}
        if set(self.closed) != set(HAND_NAMES) or any(abs(v) < 1e-6 for v in self.closed.values()):
            raise ValueError("closed_state needs a non-zero value for each of the six hand joints")
        self.reference = config["command_reference_joint"]
        self.tolerance = float(config["tolerance_rad"])

    def fit(self, target):
        """Return (GripperCommand position, resulting hand shape) for a six-joint target."""
        p = float(np.clip(np.mean([1 - target[n]/self.closed[n] for n in FINGERS]), 0, 1))
        shape = {n: self.closed[n]*(1-p) for n in FINGERS + [THUMB_PITCH]}
        shape[THUMB_YAW] = self.closed[THUMB_YAW]  # hand_control always sends SDK 0 for thumb yaw
        return self.closed[self.reference]*(1-p), shape

    def error(self, target):
        return max(abs(value-target[name]) for name, value in self.fit(target)[1].items())


class JointStateCache:
    """Latest position per joint, stamped with local receipt time (robot clocks may differ)."""
    def __init__(self):
        self._lock, self._values = threading.Lock(), {}

    def update(self, names, positions, stamp=None):
        stamp = time.monotonic() if stamp is None else stamp
        with self._lock:
            for name, value in zip(names, positions):
                self._values[name] = (stamp, float(value))

    def fresh(self, max_age_s, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            return {n: v for n, (t, v) in self._values.items() if now-t <= max_age_s and np.isfinite(v)}


class RosRobot:
    hardware = True
    # Hand feedback is 1 Hz and gripper-approximated; only arm joints are tracked tick by tick.
    tracked_names = ARM_NAMES

    def __init__(self, config):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        self.config, self.prepared = config, False
        if list(config["arm_joint_names"]) != ARM_NAMES:
            raise ValueError("arm_joint_names must match the planner's right arm order")
        self.gripper = GripperModel(config["hand_gripper"])
        self.hand_mode, self.hand_command = config["hand_mode"], None
        if self.hand_mode not in ("open_close", "fit"):
            raise ValueError("hand_mode must be open_close or fit")
        self.state, self.modules, self.feedback = JointStateCache(), None, None
        self.arm_client = self.hand_client = self.active = None
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = rclpy.create_node("gr3_grasping_robot", enable_rosout=False, start_parameter_services=False)
        self.node.create_subscription(JointState, config["joint_states_topic"],
            lambda m: self.state.update(m.name, m.position), qos_profile_sensor_data)
        try:
            from fourier_control_msgs.msg import ModuleState
            self.node.create_subscription(ModuleState, config["module_states_topic"], self._module_states, 10)
        except ImportError:
            self.node.get_logger().warning("fourier_control_msgs not importable; motion will be refused")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _module_states(self, message):
        self.modules = (time.monotonic(), dict(zip(message.name, message.state)))

    def read_state(self):
        deadline = time.monotonic() + self.config["startup_timeout_s"]
        while True:
            body = self.state.fresh(self.config["arm_max_age_s"])
            hands = self.state.fresh(self.config["hand_max_age_s"])
            state = {**{n: v for n, v in hands.items() if n in HAND_NAMES}, **body}
            missing = [n for n in COMMAND_NAMES if n not in state]
            if not missing:
                return state
            if time.monotonic() > deadline:
                raise RuntimeError(f"Joint state unavailable or stale: {missing}")
            time.sleep(.01)

    def prepare_motion(self):
        if not self.config["motion_enabled"] or not self.config["mapping_verified"]:
            raise RuntimeError("ROS motion is disabled or joint/hand mapping unverified")
        deadline = time.monotonic() + self.config["startup_timeout_s"]
        while self.modules is None and time.monotonic() < deadline:
            time.sleep(.01)
        if self.modules is None or time.monotonic()-self.modules[0] > self.config["module_states_max_age_s"]:
            raise RuntimeError("Module states unavailable or stale")
        missing = [m for m in self.config["required_modules"] if self.modules[1].get(m) != MODULE_ENABLED]
        if missing:
            raise RuntimeError(f"Modules not enabled: {missing}. Enable them on the robot side; "
                               "this adapter never publishes /enable_modules")
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        self.arm_client = ActionClient(self.node, FollowJointTrajectory, self.config["arm_action"])
        self.hand_client = ActionClient(self.node, GripperCommand, self.config["hand_action"])
        for client in (self.arm_client, self.hand_client):
            if not client.wait_for_server(timeout_sec=self.config["action_timeout_s"]):
                raise RuntimeError("Trajectory or gripper action server unavailable")
        self.prepared = True

    def stage_hand_command(self, stage):
        """GripperCommand position for a stage: open/close by stage name, or the fitted final shape."""
        if self.hand_mode == "fit":
            return self.gripper.fit(dict(zip(COMMAND_NAMES, np.asarray(stage["positions_rad"])[-1])))[0]
        settings = self.config["hand_open_close"]
        if stage["name"] in settings["open_stages"]:
            return float(settings["open_command"])
        if stage["name"] in settings["close_stages"]:
            return float(settings["close_command"])
        raise RuntimeError(f"Stage {stage['name']} has no open/close hand assignment")

    def validate_plan(self, plan):
        if self.hand_mode == "open_close":
            reference = self.gripper.closed[self.gripper.reference]
            for stage in plan["stages"]:
                command = self.stage_hand_command(stage)
                if not min(reference, 0) <= command <= max(reference, 0):
                    raise RuntimeError("Open/close hand command outside the gripper range")
            return
        for stage in plan["stages"]:
            for q in stage["positions_rad"]:
                target = dict(zip(COMMAND_NAMES, q, strict=True))
                if self.gripper.error(target) > self.gripper.tolerance:
                    raise RuntimeError("Planned hand shape is not reproducible by the GripperCommand interface")

    def send(self, q):
        raise RuntimeError("RosRobot executes whole stages via execute_stage, not streamed commands")

    def _wait(self, future, timeout_s, what):
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for {what}")
            time.sleep(.005)
        return future.result()

    def _send_hand(self, position):
        from control_msgs.action import GripperCommand
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(position), float(self.config["hand_max_effort"])
        handle = self._wait(self.hand_client.send_goal_async(goal), self.config["action_timeout_s"], "gripper goal")
        if not handle.accepted:
            raise RuntimeError("Gripper goal rejected")
        return handle.get_result_async()

    def _arm_goal(self, t, q):
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_NAMES)
        for elapsed, row in zip(t-t[0], q):
            # The bridge ignores time_from_start; spacing comes from resampling at its tick rate.
            sec = int(elapsed)
            goal.trajectory.points.append(JointTrajectoryPoint(positions=[float(v) for v in row[:7]],
                time_from_start=Duration(sec=sec, nanosec=int(round((elapsed-sec)*1e9)))))
        return goal

    def _feedback(self, message):
        self.feedback = (time.monotonic(), dict(zip(message.feedback.joint_names, message.feedback.desired.positions)))

    def execute_stage(self, stage, check):
        """Run one stage as awaited chunks; check(command) raises to stop before the next chunk."""
        if not self.prepared:
            raise RuntimeError("Hardware motion not prepared")
        from action_msgs.msg import GoalStatus
        rate = self.config["server_rate_hz"]
        t, q = resample(stage["times_s"], stage["positions_rad"], rate)
        hand_command, hand_result = self.stage_hand_command(stage), None
        # The first stage always sends, putting the hand in a known state; later only on change.
        if self.hand_command is None or abs(hand_command-self.hand_command) > 1e-6:
            hand_result, self.hand_command = self._send_hand(hand_command), hand_command
        for start, end in chunks(len(t), max(1, round(self.config["chunk_s"]*rate))):
            check(q[start])
            self.feedback = None
            handle = self._wait(self.arm_client.send_goal_async(self._arm_goal(t[start:end], q[start:end]),
                feedback_callback=self._feedback), self.config["action_timeout_s"], "trajectory goal")
            if not handle.accepted:
                raise RuntimeError("Trajectory goal rejected")
            accepted, self.active = time.monotonic(), handle.get_result_async()
            deadline = accepted + (end-start)/rate + self.config["action_timeout_s"]
            while not self.active.done():
                if time.monotonic() > deadline:
                    raise RuntimeError("Trajectory chunk did not finish in time")
                index = min(end-1, start+int((time.monotonic()-accepted)*rate))
                command = q[index].copy()
                if self.feedback is not None and self.feedback[0] >= accepted:
                    command[:7] = [self.feedback[1][n] for n in ARM_NAMES]
                check(command)
                time.sleep(1/self.config["check_hz"])
            result, self.active = self.active.result(), None
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
                raise RuntimeError(f"Trajectory chunk failed: status {result.status}, error {result.result.error_code}")
        if hand_result is not None:
            result = self._wait(hand_result, self.config["action_timeout_s"], "gripper result")
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"Gripper goal failed: status {result.status}")
        check(q[-1])

    def hold(self):
        # Goals cannot be cancelled: let the active chunk finish (bounded by chunk_s). Aurora keeps
        # the last target afterwards. The hand target is never changed here.
        if self.active is not None:
            self._wait(self.active, self.config["chunk_s"]+self.config["action_timeout_s"], "active chunk")

    def close(self):
        import rclpy
        self.executor.shutdown()
        self.node.destroy_node()
        if self.owns_context and rclpy.ok():
            rclpy.shutdown()
        self.thread.join(2)
