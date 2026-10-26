"""Versioned numeric messages; no pickle or executable wire payloads."""
import hashlib
import json
from pathlib import Path
import re

import msgpack
import numpy as np

VERSION = 1
MAX_BYTES = 64 * 1024 * 1024
DTYPES = {"uint8", "uint16", "int32", "int64", "float32", "float64", "bool"}
ARM_NAMES = [f"right_{n}_joint" for n in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll")]
HAND_NAMES = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint"] + [
    f"R_{n}_proximal_joint" for n in ("index", "middle", "ring", "pinky")]
COMMAND_NAMES = ARM_NAMES + HAND_NAMES


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError("Identifier must contain 1..80 letters/digits/dashes/underscores")
    return value


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.name not in DTYPES or value.nbytes > MAX_BYTES:
            raise ValueError("Unsupported numeric array")
        return {"__array__": True, "dtype": value.dtype.str, "shape": list(value.shape),
                "data": np.ascontiguousarray(value).tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if value.get("__array__") is True:
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if dtype.name not in DTYPES or not isinstance(shape, list) or len(shape) > 4:
            raise ValueError("Invalid array type or shape")
        count = 1
        for dimension in shape:
            if type(dimension) is not int or dimension < 0 or dimension > MAX_BYTES:
                raise ValueError("Invalid dimension")
            count *= dimension
        if count * dtype.itemsize > MAX_BYTES or count * dtype.itemsize != len(value["data"]):
            raise ValueError("Array size mismatch")
        return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()
    return value


def pack(value):
    raw = msgpack.packb(_encode(value), use_bin_type=True)
    if len(raw) > MAX_BYTES:
        raise ValueError("Message too large")
    return raw


def unpack(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("Message too large")
    return msgpack.unpackb(raw, raw=False, object_hook=_decode, strict_map_key=True)


def save(path, value):
    Path(path).write_bytes(pack(value))


def load(path):
    return unpack(Path(path).read_bytes())


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(pack(value)).hexdigest()


def validate_observation(obs):
    if obs.get("version") != VERSION:
        raise ValueError("Unsupported observation version")
    identifier(obs["observation_id"])
    state = obs["joint_positions_rad"]
    if any(n not in state for n in COMMAND_NAMES + ["head_yaw_joint", "head_pitch_joint"]):
        raise ValueError("Missing measured arm/hand/head positions")
    if not state or not all(np.isscalar(v) and np.isfinite(v) for v in state.values()):
        raise ValueError("Invalid joint positions")
    from gr3_grasping.geometry import RGBDFrame, validate_transform
    validate_transform(obs["T_world_base"])
    if not 1 <= len(obs["frames"]) <= 4:
        raise ValueError("Expected 1..4 RGB-D views")
    for record in obs["frames"]:
        frame = RGBDFrame(**record)
        if frame.rgb.dtype != np.uint8 or not np.isfinite(frame.depth_m).all() or np.any(frame.depth_m < 0):
            raise ValueError("Invalid RGB or metric depth")
        if not np.isfinite(frame.timestamp_s) or not np.any(frame.depth_m > 0):
            raise ValueError("Invalid timestamp or empty depth")
    if not np.isfinite(obs["table_top_m"]):
        raise ValueError("Invalid table height")
    for key in ("table_center_xy", "table_half_size_xy"):
        v = np.asarray(obs[key])
        if v.shape != (2,) or not np.isfinite(v).all() or (key == "table_half_size_xy" and np.any(v <= 0)):
            raise ValueError("Invalid observed table geometry")


def validate_plan(plan, observation_id, policy):
    if plan.get("version") != VERSION or plan.get("observation_id") != observation_id:
        raise ValueError("Plan/observation mismatch")
    identifier(plan["task_id"])
    if plan["joint_names"] != COMMAND_NAMES:
        raise ValueError("Unsupported command joint order")
    lower, upper = np.asarray(policy["lower_rad"]), np.asarray(policy["upper_rad"])
    speeds = np.asarray(policy["max_speed_rad_s"])
    if (lower.shape != (13,) or upper.shape != (13,) or speeds.shape != (13,) or
        not np.isfinite(np.r_[lower,upper,speeds]).all() or np.any(lower > upper) or not np.all(speeds > 0)):
        raise ValueError("Invalid local joint policy")
    previous, duration = None, 0.
    phases = [s["name"] for s in plan["stages"]]
    if phases != ["preform", "pregrasp", "approach", "grasp", "squeeze", "lift", "hold"]:
        raise ValueError("Incomplete or reordered grasp stages")
    for stage in plan["stages"]:
        q, t = np.asarray(stage["positions_rad"]), np.asarray(stage["times_s"])
        if q.ndim != 2 or q.shape[1] != 13 or not 2 <= len(q) <= 20000 or t.shape != (len(q),):
            raise ValueError("Invalid trajectory dimensions")
        if not np.isfinite(q).all() or not np.isfinite(t).all() or t[0] != 0 or not np.all(np.diff(t) > 0):
            raise ValueError("Nonfinite or nonmonotonic trajectory")
        if np.any(q < lower-1e-7) or np.any(q > upper+1e-7):
            raise ValueError("Trajectory exceeds local joint bounds")
        if np.any(abs(np.diff(q, axis=0)) / np.diff(t)[:, None] > speeds+1e-5):
            raise ValueError("Trajectory exceeds local speed bounds")
        if previous is not None and np.max(abs(q[0]-previous)) > policy["continuity_tolerance_rad"]:
            raise ValueError("Discontinuous trajectory")
        previous, duration = q[-1], duration + float(t[-1])
    if duration > policy["max_duration_s"]:
        raise ValueError("Trajectory too long")
    return duration
