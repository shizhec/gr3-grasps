"""Hard GR3 right-arm pose prior for offline Dexonomy synthesis."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from gr3_grasping.dexonomy_backend import ROOT, arm_prior_mask, arm_prior_metrics

PRIOR = ROOT / "configs/dexonomy_arm_prior.yaml"


def base_prior(**extra):
    """Configured placement and thumb rule, without the generated reachability map."""
    prior = yaml.safe_load(PRIOR.read_text())
    for key in ("reachability_map", "reach_position_tol_m", "reach_rotation_tol_rad", "lift_check_m"):
        prior.pop(key)
    return {**prior, **extra}


def scene_pose(position, thumb, palm):
    """Wrist pose (x, y, z, qw, qx, qy, qz) from FDH-6 thumb (+X) and palm (+Y) axes."""
    x, y = np.asarray(thumb, float), np.asarray(palm, float)
    R = Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))
    return np.r_[position, R.as_quat()[[3, 0, 1, 2]]]


def test_side_grasp_from_robot_is_accepted_and_reversed_wraps_rejected():
    prior = base_prior(min_approach_alignment=.34, min_finger_forward=-.5)
    natural = scene_pose([-.10, 0, .08], thumb=[0, 0, 1], palm=[1, 0, 0])
    # Thumb-down wrap of outputs/dexonomy_gr3_arm (wrap_wide/coffee_mug/0_3).
    reversed_wrap = scene_pose([-.109, .051, .167], thumb=[-.62, .18, -.76], palm=[.06, -.96, -.28])
    # Thumb-up, but the hand sits behind the mug with the palm facing the robot.
    behind = scene_pose([.127, -.026, .083], thumb=[0, 0, 1], palm=[-.72, -.69, 0])
    np.testing.assert_array_equal(arm_prior_mask([natural, reversed_wrap, behind], prior), [True, False, False])
    metrics = arm_prior_metrics(reversed_wrap, prior)
    assert metrics["thumb_up"][0] < -.7


def test_object_yaw_rotates_the_scene_hand_with_the_object():
    prior = base_prior(min_approach_alignment=.34, min_finger_forward=-.5)
    # Rotating the object by +90 deg turns a scene palm along -Y into world +X.
    turned = scene_pose([0, .10, .08], thumb=[0, 0, 1], palm=[0, -1, 0])
    assert not arm_prior_mask(turned, prior)[0]
    assert arm_prior_mask(turned, {**prior, "object_yaw_deg": 90})[0]


def test_prior_shoulder_matches_gr3_right_shoulder_joint(tmp_path):
    from gr3_grasping.model import build_scene
    from gr3_grasping.simulation import Simulation

    config = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
    if not (ROOT / config["robot"]["urdf"]).is_file():
        pytest.skip("Fetch official assets before the integration tests")
    with Simulation(build_scene(config, tmp_path), config, render=False) as sim:
        anchor = sim.data.xanchor[sim.model.joint("right_shoulder_pitch_joint").id]
    np.testing.assert_allclose(base_prior()["shoulder_world_m"], anchor, atol=1e-3)


def test_reachability_map_needs_both_grasp_and_lifted_wrist_poses(tmp_path):
    natural = scene_pose([-.10, 0, .08], thumb=[0, 0, 1], palm=[1, 0, 0])
    prior = base_prior(reachability_map=str(tmp_path / "map.npz"), reach_position_tol_m=.03,
                       reach_rotation_tol_rad=.26, lift_check_m=.16)
    world = natural[:3] + [*prior["object_xy_world_m"], prior["table_top_world_m"]]
    lifted = world + [0, 0, .16]
    # Store the sign-flipped quaternion: q and -q are the same rotation.
    np.savez(tmp_path / "map.npz", position_world_m=np.stack([world, lifted]),
             quat_world_wxyz=-np.stack([natural[3:], natural[3:]]))
    assert arm_prior_mask(natural, prior)[0]
    shifted = natural + np.r_[.04, 0, 0, 0, 0, 0, 0]
    turned = scene_pose([-.10, 0, .08], thumb=[0, np.sin(.4), np.cos(.4)], palm=[1, 0, 0])
    assert not arm_prior_mask(np.stack([shifted, turned]), prior).any()
    # Without the lifted sample the grasp itself is reachable, but not the lift.
    np.savez(tmp_path / "map.npz", position_world_m=world[None], quat_world_wxyz=natural[None, 3:])
    m = arm_prior_metrics(natural, {**prior, "reachability_map": str(tmp_path / "map.npz"), "lift_check_m": .1601})
    assert m["reach_distance"][0] < 1e-6 and m["lift_reach_distance"][0] > 1


def test_installed_prior_removes_rows_before_soft_upstream_filters():
    torch = pytest.importorskip("torch")
    pytest.importorskip("dexonomy")
    from omegaconf import OmegaConf
    import dexonomy.op.gen_init as upstream
    from gr3_grasping.dexonomy_backend import install_arm_prior

    prior = base_prior()
    good = scene_pose([-.10, 0, .08], thumb=[0, 0, 1], palm=[1, 0, 0])
    bad = scene_pose([-.10, 0, .08], thumb=[0, 0, -1], palm=[1, 0, 0])
    data = dict(grasp_pose=torch.tensor(np.stack([bad, good, bad]), dtype=torch.float32),
                obj_id=torch.zeros(3, dtype=torch.long), marker=torch.arange(3))
    no_filters = upstream.InitFilter(OmegaConf.create({"general": {"n_final": 4}}), "cpu", None)
    original, records = upstream.InitFilter.forward, []
    restore = install_arm_prior(prior, records)
    try:
        kept = no_filters.forward(dict(data))
        empty = no_filters.forward({k: v[[0, 2]] for k, v in data.items()})
    finally:
        restore()
    assert kept["marker"].tolist() == [1]
    assert all(len(v) == 0 for v in empty.values())
    assert records == [dict(sampled=3, accepted=1), dict(sampled=2, accepted=0)]
    assert upstream.InitFilter.forward is original
