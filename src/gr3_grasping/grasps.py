from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform, validate_transform


@dataclass
class Grasp:
    # TCP frame, not source gripper base. Default: z approaches, x closes.
    # FDH-6 uses its calibrated hand site and explicit approach_direction_world.
    T_world_tcp: np.ndarray
    score: float
    source: str
    approach_direction_world: np.ndarray | None = None

    def pregrasp(self, distance):
        T = self.T_world_tcp.copy()
        direction = T[:3, 2] if self.approach_direction_world is None else self.approach_direction_world
        T[:3, 3] -= distance * direction
        return T


def geometric_grasps(cloud):
    """Top/oblique tabletop proposals, without learned quality or force-closure claims."""
    center = np.median(cloud.points_world, axis=0)
    # Visible surface median is biased; use top surface only as an approach target.
    center[2] = np.quantile(cloud.points_world[:, 2], 0.80)
    result = []
    for tilt in [0, -25, -45, -60]:
        for angle in [0, np.pi / 2, np.pi, -np.pi / 2, np.pi / 4, -np.pi / 4]:
            rot = (Rotation.from_euler("y", tilt, degrees=True).as_matrix()
                   @ Rotation.from_euler("z", angle).as_matrix() @ np.diag([1, -1, -1]))
            result.append(Grasp(transform(rot, center), 0.0, "geometry_unscored"))
    return result


def fdh6_grasps(cloud, table_top_m, side="right"):
    """Opposed thumb/finger proposals from observed extents and a calibrated table.

    This assumes an upright, separated tabletop object. It is a contact baseline,
    not a learned dexterous grasp model. No simulator object pose is used.
    """
    low, high = np.quantile(cloud.points_world, [0.02, 0.98], axis=0)
    center = (low + high) / 2
    center[2] = (table_top_m + high[2]) / 2
    result = []
    # An oblique palm orientation gives the extended fingers table clearance.
    for tilt in [-60, -45, -25, 0]:
        for angle in [np.pi, np.pi / 2, 0, -np.pi / 2, np.pi / 4, -np.pi / 4]:
            rot = (Rotation.from_euler("y", tilt, degrees=True).as_matrix()
                   @ Rotation.from_euler("z", angle).as_matrix() @ np.diag([1, -1, -1]))
            if side == "left":
                # Left hand meshes are mirrored in local Y. Mirror the world
                # task and closing axis together, retaining a proper rotation.
                rot = np.diag([1, -1, 1]) @ rot @ np.diag([-1, 1, 1])
            # Enter through the open palm side (+hand Y), not along the long
            # fingers. The opposed thumb sits above the object's top in this
            # preshape; palm and extended fingers stay behind the object.
            direction = rot[:, 0] * (-1 if side == "left" else 1)
            result.append(Grasp(transform(rot, center), 0.0, "fdh6_tabletop_geometry", direction.copy()))
    return result


class GraspGen:
    """Client for the official msgpack/ZMQ protocol. No CUDA required here."""

    def __init__(self, config):
        import msgpack
        import msgpack_numpy
        import zmq

        self.config = config
        self.msgpack, self.codec, self.zmq = msgpack, msgpack_numpy, zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, config["timeout_ms"])
        self.socket.setsockopt(zmq.SNDTIMEO, config["timeout_ms"])
        self.socket.connect(f"tcp://{config['host']}:{config['port']}")
        try:
            self.metadata = self.request({"action": "metadata"})
            name = self.metadata["gripper_name"]
            # Official base-to-contact TCP depths; preserve the source embodiment.
            depths = {"franka_panda": 0.10527314, "robotiq_2f_140": 0.195,
                      "single_suction_cup_30mm": 0.069}
            if name not in depths:
                raise ValueError(f"Unknown gripper {name!r}; supply its verified base-to-TCP transform first")
            self.T_base_tcp = transform(translation=[0, 0, depths[name]])
        except BaseException:
            self.close()
            raise

    def request(self, payload):
        try:
            self.socket.send(self.msgpack.packb(payload, default=self.codec.encode, use_bin_type=True))
            response = self.msgpack.unpackb(self.socket.recv(), object_hook=self.codec.decode, raw=False)
        except self.zmq.Again as error:
            raise TimeoutError("GraspGen server timed out; start the official GPU server or use --backend geometry") from error
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def infer(self, points_camera, T_world_camera):
        points = np.asarray(points_camera, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16 or not np.isfinite(points).all():
            raise ValueError("Expected at least 16 finite metric 3D points")
        response = self.request({"action": "infer", "point_cloud": points,
                                 "num_grasps": self.config["num_grasps"],
                                 "topk_num_grasps": self.config["topk_num_grasps"],
                                 "grasp_threshold": -1.0, "min_grasps": 10, "max_tries": 2,
                                 "remove_outliers": True})
        poses, scores = np.asarray(response["grasps"]), np.asarray(response["confidences"])
        if poses.shape != (len(scores), 4, 4) or not np.isfinite(scores).all():
            raise ValueError("Malformed GraspGen response")
        result = []
        for pose, score in zip(poses, scores):
            validate_transform(pose)
            # Output is in the same metric frame as the input cloud. Do not recenter it twice.
            T = T_world_camera @ pose @ self.T_base_tcp
            result.append(Grasp(validate_transform(T), float(score), f"graspgen:{self.metadata['gripper_name']}"))
        return sorted(result, key=lambda grasp: -grasp.score)

    def close(self):
        self.socket.close()
        self.context.term()
