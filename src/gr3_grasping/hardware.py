"""Read-only robot interfaces. Motion deployment requires hardware calibration."""
import os
import time

import numpy as np

from .model import arm_names


def read_aurora_state(domain_id=123):
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    from fourier_aurora_client import AuroraClient

    # Official documentation currently requires this client on the robot's host.
    client = AuroraClient.get_instance(domain_id=domain_id, robot_name="gr3")
    try:
        time.sleep(1)
        groups = {}
        for side in ("left", "right"):
            q = np.asarray(client.get_group_state(f"{side}_manipulator", "position"), dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise RuntimeError(f"Invalid {side} arm state: {q}")
            groups[f"{side}_manipulator"] = {"joint_names": arm_names(side), "position_rad": q.tolist()}
        return {"timestamp_s": time.time(), "domain_id": domain_id, "groups": groups,
                "joint_order_requires_hardware_verification": True, "read_only": True}
    finally:
        client.close()
