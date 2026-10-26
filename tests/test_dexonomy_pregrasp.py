"""Palm-normal retreat pregrasp replacing upstream's 40 mm whole-hand clearance."""
import importlib.util
import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from gr3_grasping.dexonomy_backend import (
    HAND_SOURCE, PALM_RETREAT, ROOT, expand_hand_q, export_hand, export_object,
)


@pytest.fixture(scope="module")
def candy(tmp_path_factory):
    pytest.importorskip("dexonomy")
    pytest.importorskip("trimesh")
    item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"]
                if i["id"] == "candy_box")
    if not HAND_SOURCE.is_file() or not (ROOT / "third_party/mujoco_scanned_objects/models" / item["folder"]).is_dir():
        pytest.skip("Fetch official hand and GSO assets")
    assets = tmp_path_factory.mktemp("dexonomy_pregrasp")
    _, meta = export_hand(assets / "assets/hand/fdh6")
    scene = export_object(item, assets / "assets/object/candy_box")
    fixture = json.loads((ROOT / "tests/fixtures/dexonomy_candy_pose.json").read_text())
    payload = {key: np.asarray(fixture[key]) for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")}
    payload.update(tmpl_name=fixture["tmpl_name"], scene_path=str(scene))
    return assets, meta, payload


def make_output(tmp_path, candy, names=("pose",)):
    assets, _, payload = candy
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets").symlink_to(assets / "assets")
    directory = tmp_path / "run/grasp_data/wrap_medium/candy_box"
    directory.mkdir(parents=True)
    for name in names:
        np.save(directory / f"{name}.npy", payload)
    return directory


def env_for(candy):
    from dexonomy.sim import HandCfg, MuJoCo_OptEnv, MuJoCo_OptCfg
    from dexonomy.util.file_util import load_scene_cfg

    assets, _, payload = candy
    return MuJoCo_OptEnv(HandCfg(str(assets / "assets/hand/fdh6/hand.xml"), freejoint=True),
                         load_scene_cfg(payload["scene_path"]), MuJoCo_OptCfg(obj_margin=0))


def test_retreat_backs_off_along_palm_normal_with_opened_coupled_fingers(candy):
    from gr3_grasping.dexonomy_backend import hand_clearances, palm_retreat_pregrasp

    _, meta, payload = candy
    env = env_for(candy)
    grasp = payload["grasp_qpos"][-1].astype(float)
    path, info = palm_retreat_pregrasp(env, grasp, meta)
    assert path is not None, info
    # Near end: exact grasp wrist; orientation is held along the whole path.
    np.testing.assert_allclose(path[-1][:7], grasp[:7])
    np.testing.assert_allclose(path[:, 3:7], np.tile(grasp[3:7], (len(path), 1)))
    palm = Rotation.from_quat(grasp[[4, 5, 6, 3]]).as_matrix()[:, 1]
    offset = path[0][:3] - grasp[:3]
    assert np.linalg.norm(offset) == pytest.approx(info["retreat_m"])
    assert offset @ -palm > .7 * info["retreat_m"]  # away from the object, at most a lift bias
    active = [meta["joint_order"].index(n) for n in meta["active_joints"]]
    closed, opened = grasp[7:][active], path[0][7:][active]
    assert opened[0] == pytest.approx(closed[0])  # thumb yaw keeps its grasp value
    np.testing.assert_allclose(opened[1:], (1 - info["open_ratio"]) * closed[1:], atol=1e-12)
    np.testing.assert_allclose(path[0][7:], expand_hand_q(opened, meta), atol=1e-12)
    target, table = np.array([hand_clearances(env, q) for q in path]).T
    assert target[0] >= PALM_RETREAT["clearance_m"]
    assert target.min() >= -PALM_RETREAT["approach_penetration_m"]
    assert table.min() >= PALM_RETREAT["table_clearance_m"]


def test_retreat_reports_failed_checks_instead_of_relaxing_them(candy):
    from gr3_grasping.dexonomy_backend import palm_retreat_pregrasp

    _, meta, payload = candy
    grasp = payload["grasp_qpos"][-1]
    path, info = palm_retreat_pregrasp(env_for(candy), grasp, meta, {**PALM_RETREAT, "table_clearance_m": 1.})
    assert path is None
    assert {a["reason"] for a in info["attempts"]} == {"table_clearance"}
    assert len(info["attempts"]) == 2 * 2 * 3
    path, info = palm_retreat_pregrasp(env_for(candy), grasp, meta, reachable=lambda wrist: False)
    assert path is None and {a["reason"] for a in info["attempts"]} == {"pregrasp_wrist_unreachable"}


def test_apply_rewrites_accepted_and_moves_rejected_candidates(candy, tmp_path):
    from gr3_grasping.dexonomy_backend import apply_palm_retreat

    directory = make_output(tmp_path / "ok", candy)
    report = apply_palm_retreat(tmp_path / "ok", ["candy_box"])
    grasp = np.load(directory / "pose.npy", allow_pickle=True).item()
    assert report["wrap_medium/candy_box/pose.npy"]["accepted"]
    assert grasp["pregrasp_source"] == "palm_retreat"
    np.testing.assert_array_equal(grasp["upstream_pregrasp_qpos"], candy[2]["pregrasp_qpos"])
    # A resumed pass keeps the first decision and does not retreat twice.
    apply_palm_retreat(tmp_path / "ok", ["candy_box"])
    np.testing.assert_array_equal(np.load(directory / "pose.npy", allow_pickle=True).item()["pregrasp_qpos"],
                                  grasp["pregrasp_qpos"])

    make_output(tmp_path / "bad", candy)
    report = apply_palm_retreat(tmp_path / "bad", ["candy_box"], settings={**PALM_RETREAT, "clearance_m": 1.})
    assert not report["wrap_medium/candy_box/pose.npy"]["accepted"]
    assert not list((tmp_path / "bad/run/grasp_data").rglob("*.npy"))
    assert (tmp_path / "bad/run/pregrasp_rejected/wrap_medium/candy_box/pose.npy").is_file()


def test_retreated_candy_grasp_still_lifts_under_gravity(candy, tmp_path):
    from gr3_grasping.dexonomy_backend import apply_palm_retreat

    if mujoco.__version__ != "3.16.0":
        pytest.skip("Pinned discrete dynamics regression requires MuJoCo 3.16.0")
    directory = make_output(tmp_path, candy)
    apply_palm_retreat(tmp_path, ["candy_box"])
    spec = importlib.util.spec_from_file_location("validate_dexonomy_lift", ROOT / "scripts/validate_dexonomy_lift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.validate(directory / "pose.npy", tmp_path)
    assert report["success"], report["failure"]
    assert report["max_penetration_m"] < .003
