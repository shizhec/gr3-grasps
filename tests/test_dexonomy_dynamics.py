"""Frozen FDH-6 candidate: gravity lift, open-hand control and integrator A/B."""
import importlib.util
import json

import mujoco
import numpy as np
import pytest
import yaml

from gr3_grasping.dexonomy_backend import HAND_SOURCE, ROOT, export_hand, export_object


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    pytest.importorskip("dexonomy")
    if mujoco.__version__ != "3.16.0":
        pytest.skip("Pinned discrete dynamics regression requires MuJoCo 3.16.0")
    item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"]
                if i["id"] == "candy_box")
    if not HAND_SOURCE.is_file() or not (ROOT / "third_party/mujoco_scanned_objects/models" / item["folder"]).is_dir():
        pytest.skip("Fetch official hand and GSO assets")
    output = tmp_path_factory.mktemp("dexonomy_dynamics")
    export_hand(output / "assets/hand/fdh6")
    scene = export_object(item, output / "assets/object/candy_box")
    fixture = json.loads((ROOT / "tests/fixtures/dexonomy_candy_pose.json").read_text())
    payload = {key: np.asarray(fixture[key]) for key in
               ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")}
    payload.update(tmpl_name=fixture["tmpl_name"], scene_path=str(scene))
    path = output / "run/grasp_data/wrap_medium/candy_box/pose.npy"
    path.parent.mkdir(parents=True)
    np.save(path, payload)
    spec = importlib.util.spec_from_file_location("validate_dexonomy_lift", ROOT / "scripts/validate_dexonomy_lift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, output, module.validate


@pytest.mark.parametrize("integrator,no_close,expected", [
    ("discrete", False, True), ("discrete", True, False), ("implicitfast", False, False),
])
def test_coupled_contact_lift_and_controls(case, integrator, no_close, expected):
    path, output, validate = case
    report = validate(path, output, no_close=no_close, integrator=integrator)
    assert report["success"] is expected
    assert report["mass_kg"] == pytest.approx(.13)
    assert report["max_penetration_m"] < .003
    if expected:
        assert report["hold_min_lift_m"] >= .1
        assert report["hold_duration_s"] == pytest.approx(2)
        assert report["opposed_fraction"] >= .9
        assert report["max_mimic_error_rad"] < .001
        assert report["max_wrist_position_error_m"] <= .015
        assert report["max_wrist_rotation_error_rad"] <= .1
    elif no_close:
        assert report["peak_lift_m"] < .01
        assert report["opposed_fraction"] == 0
    else:
        assert report["failure"] == "mimic_error_exceeds_0.01rad"
