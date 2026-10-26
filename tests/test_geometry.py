import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gr3_grasping.geometry import RGBDFrame, transform, validate_transform
from gr3_grasping.perception import segment_tabletop


def test_optical_backprojection_roundtrip_and_storage(tmp_path):
    K = np.array([[200, 0, 2], [0, 220, 1], [0, 0, 1]])
    depth = np.full((3, 5), 0.8, dtype=np.float32)
    T = transform(Rotation.from_euler("y", 25, degrees=True).as_matrix(), [0.1, 0, 1.5])
    frame = RGBDFrame(np.zeros((3, 5, 3), dtype=np.uint8), depth, K, T, 1.0, "test")
    points = frame.points_camera()
    np.testing.assert_allclose(points[1, 2], [0, 0, 0.8], atol=1e-7)
    np.testing.assert_allclose(frame.points_world()[1, 2], T[:3, 3] + 0.8 * T[:3, 2], atol=1e-7)
    projected = points[..., :2] / points[..., 2:] * np.array([200, 220]) + [2, 1]
    v, u = np.indices(depth.shape)
    np.testing.assert_allclose(projected, np.stack([u, v], axis=-1), atol=1e-6)
    path = tmp_path / "frame.npz"
    frame.save(path)
    np.testing.assert_equal(RGBDFrame.load(path).rgb, frame.rgb)
    np.testing.assert_allclose(RGBDFrame.load(path).T_world_camera, T)


def test_reject_improper_rotation():
    T = np.eye(4)
    T[0, 0] = -1
    with pytest.raises(ValueError, match="SO"):
        validate_transform(T)


def test_depth_foreground_ignores_invalid_depth_and_needs_no_simulator():
    depth = np.full((24, 32), 2.0, dtype=np.float32)
    depth[4:10, 4:10] = 0.8
    depth[12:19, 22:29] = 0.7
    depth[5, 5] = 0
    depth[13, 23] = np.nan
    frame = RGBDFrame(np.zeros((24, 32, 3), dtype=np.uint8), depth,
                      np.array([[100, 0, 16], [0, 100, 12], [0, 0, 1]]), np.eye(4), 0, "synthetic")
    config = {"workspace_min": [-1, -1, 0.5], "workspace_max": [1, 1, 1.0], "min_pixels": 10}
    objects = segment_tabletop(frame, config)
    assert len(objects) == 2
    assert all(np.isfinite(cloud.points_camera).all() for cloud in objects)
    cropped = segment_tabletop(frame, config, box=[1, 1, 14, 12])
    assert len(cropped) == 1
