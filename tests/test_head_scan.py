from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gr3_grasping.cli import write_json  # Configure EGL before importing MuJoCo.
from gr3_grasping.geometry import RGBDFrame, transform, transform_points
from gr3_grasping.head_scan import DEFAULTS, associate_target, capture_head_scan, voxel_fuse
from gr3_grasping.model import ROOT, build_scene
from gr3_grasping.perception import segment_tabletop
from gr3_grasping.simulation import Simulation


def test_fusion_aligns_two_camera_origins_and_removes_duplicate_observations():
    K = np.array([[100, 0, 2], [0, 100, 2], [0, 0, 1]])
    frames = [RGBDFrame(np.full((5, 5, 3), 100, np.uint8), np.ones((5, 5), np.float32),
                        K, transform(translation=[shift, 0, 0]), shift, "test") for shift in [0, .02]]
    # Quantize cell centers away from floating-point voxel boundaries.
    points = np.concatenate([frame.points_world().reshape(-1, 3) for frame in frames]) + .001
    colors = np.concatenate([frame.rgb.reshape(-1, 3) for frame in frames])
    fused, rgb = voxel_fuse(points, colors, .005)
    assert len(fused) == 35  # 7 columns, 5 rows; 15 overlapping observations.
    np.testing.assert_allclose(fused[:, 2], 1.001)
    np.testing.assert_array_equal(rgb, 100)
    local = transform_points(np.linalg.inv(frames[1].T_world_camera), fused)
    np.testing.assert_allclose(transform_points(frames[1].T_world_camera, local), fused)


def test_view_association_rejects_larger_unrelated_foreground():
    rng = np.random.default_rng(7)
    points = rng.normal(0, .01, (200, 3))
    matching = SimpleNamespace(points_world=points + [.001, 0, 0])
    hand = SimpleNamespace(points_world=rng.normal(.2, .01, (3000, 3)))
    selected, overlap = associate_target([hand, matching], points, DEFAULTS)
    assert selected is matching and overlap > .9
    assert associate_target([hand], points, DEFAULTS)[0] is None


def test_target_crop_preserves_observed_target_and_nearby_table_context():
    from gr3_grasping.m2t2_backend import crop_target_context

    target = np.array([[.40, -.20, 1.05], [.45, -.15, 1.12]])
    support = np.array([[.38, -.22, 1.0], [.48, -.12, 1.0]])
    distant = np.array([[.7, .3, 1.0]])
    scene = np.concatenate([target, support, distant])
    np.testing.assert_array_equal(crop_target_context(scene, target, .10), np.concatenate([target, support]))
    np.testing.assert_array_equal(crop_target_context(scene, target, None), scene)
    with pytest.raises(ValueError):
        crop_target_context(scene, target, -.1)


def test_native_head_scan_moves_servos_updates_extrinsics_and_restores_head(tmp_path):
    config = yaml.safe_load((ROOT / "configs/scanned_candy.yaml").read_text())
    config["head_scan"] = {"enabled": True}
    if not (ROOT / config["robot"]["urdf"]).is_file():
        pytest.skip("Fetch official robot and object assets first")
    scene = build_scene(config, tmp_path / "model")
    with Simulation(scene, config) as sim:
        sim.settle()
        frame = sim.capture()
        target = segment_tabletop(frame, config["perception"])[0]
        initial_joints = sim.data.qpos.copy()
        points, fused_target, summary = capture_head_scan(sim, frame, target, config, tmp_path)
        assert summary["accepted_views"] == 4
        assert summary["fused_target_voxels"] > summary["initial_target_voxels"]
        assert .03 < summary["max_camera_baseline_m"] < .12
        assert summary["head_restoration"]["tracking_error_rad"] < .015
        for name in ["head_yaw_joint", "head_pitch_joint"]:
            qid = sim.model.joint(name).qposadr[0]
            assert sim.data.qpos[qid] == pytest.approx(initial_joints[qid], abs=.015)
        for view in summary["views"][1:]:
            assert view["max_measured_speed_rad_s"] <= .37
            assert view["timestamp_s"] > frame.timestamp_s
        assert not np.allclose(summary["views"][0]["T_world_camera"], summary["views"][1]["T_world_camera"])
        np.testing.assert_allclose(transform_points(frame.T_world_camera, fused_target.points_camera),
                                   fused_target.points_world, atol=1e-7)
        assert (tmp_path / "head_scan/views.png").is_file()
        assert len(points) > len(fused_target.points_world)


class UpstreamShapedModel:
    def infer(self, score, mask, cfg):
        import torch
        return torch.where((score > cfg.object_thresh) & mask.sum(dim=1) > 0)[0]


def test_m2t2_fix_keeps_even_and_odd_masks_but_rejects_empty_and_low_confidence():
    torch = pytest.importorskip("torch")
    from gr3_grasping.m2t2_compat import correct_object_mask_filter

    model = UpstreamShapedModel()
    score = torch.tensor([.99, .99, .99, .99, .99, .1])
    mask = torch.arange(4)[None] < torch.tensor([0, 1, 2, 3, 4, 4])[:, None]
    cfg = SimpleNamespace(object_thresh=.4)
    assert model.infer(score, mask, cfg).tolist() == [1, 3]
    metadata = correct_object_mask_filter(model)
    assert model.infer(score, mask, cfg).tolist() == [1, 2, 3, 4]
    assert metadata["name"] == "object_mask_filter_parentheses"
