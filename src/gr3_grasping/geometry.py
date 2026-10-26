from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def validate_transform(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 4x4 transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous transform")
    rot = value[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rot), 1, atol=1e-4):
        raise ValueError("Rotation must belong to SO(3)")
    return value


def transform_points(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


@dataclass
class RGBDFrame:
    """Aligned RGB/depth. Optical frame: x right, y down, z forward. Units: m."""

    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_world_camera: np.ndarray
    timestamp_s: float
    source: str

    def __post_init__(self):
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("RGB and aligned depth dimensions do not match")
        self.K = np.asarray(self.K, dtype=float)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(self.K[2], [0, 0, 1]) or not np.isclose(self.K[0, 1], 0):
            raise ValueError("Expected rectified zero-skew intrinsics")
        self.T_world_camera = validate_transform(self.T_world_camera)

    def points_camera(self):
        v, u = np.indices(self.depth_m.shape)
        z = self.depth_m
        return np.stack(((u - self.K[0, 2]) * z / self.K[0, 0],
                         (v - self.K[1, 2]) * z / self.K[1, 1], z), axis=-1)

    def points_world(self):
        return transform_points(self.T_world_camera, self.points_camera())

    def save(self, path):
        np.savez_compressed(path, rgb=self.rgb, depth_m=self.depth_m, K=self.K,
                            T_world_camera=self.T_world_camera, timestamp_s=self.timestamp_s,
                            source=self.source)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["rgb"], data["depth_m"], data["K"], data["T_world_camera"],
                       float(data["timestamp_s"]), str(data["source"]))


def rotation_error(target, actual):
    return Rotation.from_matrix(target @ actual.T).as_rotvec()
