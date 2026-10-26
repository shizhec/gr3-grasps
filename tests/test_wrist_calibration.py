"""Check offset frame/sign conventions and calibration/validation separation."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gr3_grasping.geometry import transform
from gr3_grasping.grasps import Grasp
from gr3_grasping.m2t2_backend import retarget_fdh6
from gr3_grasping.wrist_calibration import BASE_OFFSET, calibration_score, load_offset_profile


def test_offset_correction_moves_wrist_in_its_own_rotated_frame():
    source = Grasp(transform(Rotation.from_euler("xyz", [.4, -.2, 1.1]).as_matrix(), [.5, -.2, 1.1]), .8, "m2t2")
    correction = np.array([.01, -.005, .007])
    old = retarget_fdh6([source], "right", BASE_OFFSET)
    new = retarget_fdh6([source], "right", BASE_OFFSET + correction)
    for before, after in zip(old, new):
        R = before.T_world_tcp[:3, :3]
        np.testing.assert_allclose(after.T_world_tcp[:3, 3] - before.T_world_tcp[:3, 3], -R @ correction)
        np.testing.assert_allclose(after.T_world_tcp[:3, :3], R)
        np.testing.assert_allclose(after.approach_direction_world, before.approach_direction_world)
        assert after.score == before.score == .8


@pytest.mark.parametrize("value", [[0, 0], [float("nan"), 0, 0], [.2, 0, 0]])
def test_invalid_or_excessive_offsets_are_rejected(tmp_path, value):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": "gr3_fdh6_wrist_offset_v1", "side": "right", "contact_offset_tcp_m": value}))
    with pytest.raises(ValueError):
        load_offset_profile(path)


def test_calibration_success_count_takes_priority_over_contact_or_height():
    success = [{"grasp_success": True, "hold_opposed_contact_fraction": .91, "hold_min_lift_m": .101}]
    near_misses = [{"grasp_success": False, "hold_opposed_contact_fraction": 1., "hold_min_lift_m": .099}] * 10
    assert calibration_score(success, BASE_OFFSET) > calibration_score(near_misses, BASE_OFFSET)
