import numpy as np
import pytest
from types import SimpleNamespace
from copy import deepcopy

from gr3_grasping.cli import write_json  # Configure EGL before MuJoCo imports.

torch = pytest.importorskip("torch")
from gr3_grasping.pointnet_cpu import ball_query, furthest_point_sampling, three_nn, three_interpolate
from gr3_grasping.m2t2_backend import M2T2Backend, base_to_tcp, retarget_fdh6
from gr3_grasping.geometry import transform
from gr3_grasping.grasps import Grasp


def test_ball_query_uses_input_order_strict_radius_and_first_point_padding():
    xyz = torch.tensor([[[0.2, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [0.3, 0, 0]]])
    centers = torch.tensor([[[0., 0, 0], [2., 0, 0]]])
    indices = ball_query(centers, xyz, 0.3, 4)
    assert indices.tolist() == [[[0, 1, 0, 0], [0, 0, 0, 0]]]


def test_fps_and_three_neighbor_interpolation():
    xyz = torch.tensor([[[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0., 0, 0]]])
    assert furthest_point_sampling(xyz, 3).tolist() == [[0, 2, 1]]
    query = torch.tensor([[[0.12, 0, 0]]])
    d2, idx = three_nn(query, xyz)
    np.testing.assert_allclose(d2, [[[0.02 ** 2, 0.08 ** 2, 0.12 ** 2]]], atol=1e-7)
    inv = 1 / d2.sqrt()
    weights = inv / inv.sum(-1, keepdim=True)
    values = torch.tensor([[[10., 20., 30., 0.]]])
    expected = 10 * weights[0, 0, 0] + 20 * weights[0, 0, 1]
    assert three_interpolate(values, idx, weights)[0, 0, 0] == pytest.approx(expected)


def test_m2t2_contact_offset_and_fdh6_transfer_keep_learned_center_and_score():
    pose = transform(np.diag([1, -1, -1]), [0.5, -0.2, 1.2])
    tcp = base_to_tcp(pose)
    np.testing.assert_allclose(tcp[:3, 3], [0.5, -0.2, 1.0966])
    candidates = retarget_fdh6([Grasp(tcp, 0.85, "m2t2:franka_panda")], "right")
    for candidate in candidates:
        np.testing.assert_allclose(candidate.T_world_tcp[:3, 3] + candidate.T_world_tcp[:3, :3] @ [-.012, 0, -.045], tcp[:3, 3])
        assert candidate.score == 0.85
        np.testing.assert_allclose(candidate.pregrasp(.08)[:3, 3], candidate.T_world_tcp[:3, 3] - .08 * tcp[:3, 2])
    assert np.linalg.det(candidates[1].T_world_tcp[:3, :3]) == pytest.approx(1)


@pytest.fixture(scope="module")
def learned_scan_scene(tmp_path_factory):
    pytest.importorskip("omegaconf")
    pytest.importorskip("trimesh")
    import yaml
    from gr3_grasping.model import ROOT, build_scene
    from gr3_grasping.perception import segment_tabletop
    from gr3_grasping.simulation import Simulation

    config = yaml.safe_load((ROOT / "configs/scanned_candy.yaml").read_text())
    required = [ROOT / config["robot"]["urdf"], ROOT / "third_party/M2T2Models/m2t2.pth",
                ROOT / "third_party/M2T2/config.yaml"]
    required.extend(ROOT / obj["mjcf"] for obj in config["scene"]["objects"])
    if not all(path.is_file() for path in required):
        pytest.skip("Fetch official robot/scanned assets and M2T2 checkpoint first")
    directory = tmp_path_factory.mktemp("m2t2_scan")
    scene = build_scene(config, directory / "model")
    with Simulation(scene, config) as sim:
        sim.settle()
        frame = sim.capture()
        target = segment_tabletop(frame, config["perception"])[0]
    backend = M2T2Backend(config["m2t2"])
    source = backend.infer(frame, target, config["scene"], config["perception"], directory)
    assert backend.metadata["raw_grasps"] > 0
    assert all(grasp.source == "m2t2:franka_panda" for grasp in source)
    proposals = retarget_fdh6(source, "right", config["m2t2"]["fdh6_contact_offset_m"])
    return config, scene, proposals, target.centroid_world


@pytest.mark.parametrize("no_close,height_reference", [(False, "body_origin"), (True, "body_origin"), (False, "center_of_mass")])
def test_official_m2t2_weights_to_real_scan_contact_grasp(learned_scan_scene, tmp_path, no_close, height_reference):
    from gr3_grasping.grasping import run_grasp
    from gr3_grasping.simulation import Simulation

    config, scene, proposals, centroid = learned_scan_scene
    config = deepcopy(config)
    config["hand"]["evaluation_height_reference"] = height_reference
    with Simulation(scene, config, render=False) as sim:
        sim.settle()
        body = sim.model.body("scanned_candy").id
        initial_com_z = float(sim.data.xipos[body, 2])
        report = {"arm": "right", "hand": "fdh6", "selected_centroid_world_m": centroid}
        result = run_grasp(sim, proposals, report, SimpleNamespace(plan_only=False, no_close=no_close), tmp_path, write_json)
        if height_reference == "center_of_mass":
            assert result["final_lift_m"] == pytest.approx(float(sim.data.xipos[body, 2]) - initial_com_z)
    assert result["object_height_reference"] == height_reference
    assert result["grasp_success"] is (not no_close)
    if no_close:
        assert abs(result["final_lift_m"]) < .005
        assert result["hold_opposed_contact_fraction"] == 0
    else:
        assert result["hold_min_lift_m"] >= .10
        assert result["hold_opposed_contact_fraction"] >= .90
        assert result["max_target_finger_penetration_m"] < .003
