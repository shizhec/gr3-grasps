import json

from gr3_grasping.cli import write_json  # Configure EGL before MuJoCo.
import mujoco
import numpy as np
import pytest
import yaml

from gr3_grasping.model import ROOT, build_scene


def test_two_scans_preserve_scale_texture_convex_parts_and_free_dynamics(tmp_path):
    config = yaml.safe_load((ROOT / "configs/scanned.yaml").read_text())
    if not (ROOT / config["scene"]["objects"][0]["mjcf"]).is_file():
        pytest.skip("Fetch scanned assets first")
    path = build_scene(config, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.neq == 10  # Only robot linkage constraints, no object welds.
    for spec in config["scene"]["objects"]:
        body = model.body(spec["name"]).id
        assert model.joint(spec["name"] + "_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
        geoms = np.flatnonzero(model.geom_bodyid == body)
        assert len(geoms) == 33
        visual = geoms[model.geom_contype[geoms] == 0]
        assert len(visual) == 1 and model.geom_matid[visual[0]] >= 0
        assert sum(model.geom_contype[geoms] == 2) == 32
        assert model.body_mass[body] == pytest.approx(spec["mass"])
        assert model.body_gravcomp[body] == 0
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    dimensions = provenance["scanned_objects"][0]["dimensions_m"]
    np.testing.assert_allclose(dimensions, [0.053116, 0.052494, 0.088823], atol=1e-5)
