"""Local clock, local bounds, state checks and a persistent execution journal."""
import json
from pathlib import Path
import time

import numpy as np

from .protocol import COMMAND_NAMES, json_value, validate_plan


def execute(plan, robot, policy, directory, observation_id, execute_motion=False, stop_file=None,
            time_scale=1.):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_plan(plan, observation_id, policy)
    journal = directory / f"{plan['task_id']}.jsonl"
    # Exclusive creation is an at-most-once guard, including aborted tasks.
    with journal.open("x") as log:
        def event(kind, **fields):
            value = dict(event=kind, local_time_s=time.monotonic(), task_id=plan["task_id"], **fields)
            log.write(json.dumps(json_value(value), allow_nan=False)+"\n")
            log.flush()

        event("received", hardware=robot.hardware)
        if robot.hardware and (not execute_motion or not plan["hardware_eligible"]):
            event("rejected", reason="Hardware requires --execute plus verified observation/configuration")
            raise RuntimeError("Hardware execution is disabled or observation is uncalibrated/unverified")
        if robot.hardware and time_scale != 1:
            raise ValueError("Hardware timing cannot be accelerated")
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Invalid time scale")
        if robot.hardware:
            # A different --output directory must never bypass hardware replay protection.
            ledger = Path(policy["task_ledger_dir"]).expanduser()
            ledger.mkdir(parents=True, exist_ok=True)
            with (ledger/f"{plan['task_id']}.claim").open("x") as claim:
                claim.write(json.dumps(dict(task_id=plan["task_id"], observation_id=observation_id)))
        def measured():
            state = robot.read_state()
            for name, reference in plan.get("context_joint_positions_rad", {}).items():
                if name not in state or not np.isfinite(state[name]) or abs(state[name]-reference) > policy["context_tolerance_rad"]:
                    raise RuntimeError("Frozen body/head joints moved; reobserve and replan")
            q = np.array([state[n] for n in COMMAND_NAMES])
            if q.shape != (13,) or not np.isfinite(q).all():
                raise RuntimeError("Invalid robot feedback")
            if np.any(q < np.asarray(policy["lower_rad"])-.01) or np.any(q > np.asarray(policy["upper_rad"])+.01):
                raise RuntimeError("Measured joints outside local bounds")
            return q

        # Adapters with slow or approximate feedback (e.g. a 1 Hz gripper) track a subset tick by tick.
        tracked = [COMMAND_NAMES.index(n) for n in getattr(robot, "tracked_names", COMMAND_NAMES)]

        def check(command, phase, record=True):
            if stop_file and Path(stop_file).exists():
                raise RuntimeError("Local stop file requested")
            actual = measured()
            if np.max(abs(actual-command)[tracked]) > policy["tracking_tolerance_rad"]:
                raise RuntimeError("Joint tracking exceeds local tolerance")
            if record:
                event("sample", stage=phase, command_rad=command, actual_rad=actual)

        initial = np.asarray(plan["stages"][0]["positions_rad"])[0]
        if np.max(abs(measured()-initial)) > policy["start_tolerance_rad"]:
            event("rejected", reason="Measured start state differs from plan")
            raise RuntimeError("Measured start state differs from plan; reobserve and replan")
        phase = "prepare"
        try:
            robot.validate_plan(plan)
            robot.prepare_motion()
            if np.max(abs(measured()-initial)) > policy["start_tolerance_rad"]:
                raise RuntimeError("Robot moved during adapter preparation")
            for stage in plan["stages"]:
                phase = stage["name"]
                event("stage_started", stage=phase)
                if hasattr(robot, "execute_stage"):
                    # The adapter owns stage timing (e.g. trajectory goals) and calls back between steps.
                    robot.execute_stage(stage, lambda command: check(np.asarray(command, float), phase))
                    event("stage_completed", stage=phase)
                    continue
                t = np.asarray(stage["times_s"])
                q = np.asarray(stage["positions_rad"])
                start = time.monotonic()
                period = 1 / policy["command_hz"]
                steps = int(np.ceil(t[-1]*time_scale/period))
                for step in range(steps+1):
                    deadline = start+min(step*period, t[-1]*time_scale)
                    remaining = deadline-time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    if time.monotonic()-deadline > policy["max_tick_lateness_s"]:
                        raise RuntimeError("Local command scheduling deadline missed")
                    if stop_file and Path(stop_file).exists():
                        raise RuntimeError("Local stop file requested")
                    elapsed = min(step*period/time_scale, t[-1])
                    command = np.array([np.interp(elapsed, t, q[:, j]) for j in range(13)])
                    robot.send(command)
                    check(command, phase, step % max(1, round(policy["command_hz"]/10)) == 0 or step == steps)
                event("stage_completed", stage=phase)
            event("completed", grasp_success=None)
            return dict(observation_id=observation_id, task_id=plan["task_id"], trajectory_completed=True,
                grasp_success=None, hardware=robot.hardware,
                scope="Kinematic mock transport/execution check" if not robot.hardware else "Hardware trajectory completion; grasp needs sensor verification")
        except BaseException as error:
            try:
                robot.hold()
            except Exception as hold_error:
                event("hold_failed", error=str(hold_error))
            event("aborted", stage=phase, error=f"{type(error).__name__}: {error}")
            raise
