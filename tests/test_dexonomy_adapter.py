"""Check FDH-6 reduced coordinates and object mass for the offline adapter."""
import json

import mujoco
import numpy as np
import pytest
import yaml

from gr3_grasping.dexonomy_backend import (
    HAND_SOURCE, ROOT, expand_hand_q, export_hand, export_object, export_grasp_commands,
)


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    if not HAND_SOURCE.exists():
        pytest.skip("Official FDH-6 assets not downloaded")
    return export_hand(tmp_path_factory.mktemp("dexonomy_hand"))


def test_reduced_hand_has_only_six_motors_and_five_joint_constraints(hand):
    path, meta = hand
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nu, model.neq) == (11, 6, 5)
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    rng = np.random.default_rng(91)
    ids = [model.joint(n).id for n in meta["active_joints"]]
    for _ in range(20):
        active = rng.uniform(model.jnt_range[ids, 0], model.jnt_range[ids, 1])
        q = expand_hand_q(active, meta)
        for eq in range(model.neq):
            a, b = model.eq_obj1id[eq], model.eq_obj2id[eq]
            assert abs(q[model.jnt_qposadr[a]] - model.eq_data[eq, 1] * q[model.jnt_qposadr[b]]) < 1e-12
        assert np.all(q >= model.jnt_range[:, 0])
        assert np.all(q <= model.jnt_range[:, 1])
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-.25, .25], (6, 1)))


def test_mimic_force_mapping_preserves_virtual_work(hand):
    _, meta = hand
    jac = np.column_stack([expand_hand_q(row, meta) for row in np.eye(6)])
    rng = np.random.default_rng(2)
    torques, displacement = rng.normal(size=11), rng.normal(size=6)
    assert (jac.T @ torques) @ displacement == pytest.approx(torques @ (jac @ displacement))
    # A passive-only load must affect its driving actuator.
    thumb_load = np.zeros(11)
    thumb_load[meta["joint_order"].index("R_thumb_distal_joint")] = 1
    assert (jac.T @ thumb_load)[1] == pytest.approx(1.06)


def test_scanned_visual_has_zero_mass_and_collision_mass_scales(tmp_path):
    pytest.importorskip("trimesh")
    item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"] if i["id"] == "medicine_bottle")
    if not (ROOT / "third_party/mujoco_scanned_objects/models" / item["folder"]).exists():
        pytest.skip("GSO assets not downloaded")
    scene_path = export_object(item, tmp_path)
    scene = np.load(scene_path, allow_pickle=True).item()
    spec = mujoco.MjSpec.from_file(scene["scene"][item["id"]]["xml_path"])
    info = json.loads((tmp_path / "info.json").read_text())
    density = item["mass_kg"] / (info["mass"] / info["density"])
    for geom in spec.geoms:
        assert geom.name, "Dexonomy margin updates require named geometry"
        if geom.contype:
            geom.density = density
        else:
            assert geom.mass == 0
    model = spec.compile()
    assert model.body_mass.sum() == pytest.approx(item["mass_kg"], rel=1e-6)


def test_dexonomy_runtime_margin_updates_all_scanned_geoms(hand, tmp_path):
    pytest.importorskip("dexonomy")
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"] if i["id"] == "medicine_bottle")
    path = export_object(item, tmp_path)
    scene = np.load(path, allow_pickle=True).item()
    env = MuJoCo_OptEnv(HandCfg(str(hand[0]), freejoint=True), scene,
                       MuJoCo_OptCfg(obj_margin=.045))
    ids = [i for i in range(env._model.ngeom) if env._model.geom(i).name.startswith("obj-")]
    assert len(ids) > 1
    collision_ids = [i for i in ids if env._model.geom_contype[i]]
    np.testing.assert_allclose(env._model.geom_margin[collision_ids], .045)
    env.set_obj_margin(.001)
    np.testing.assert_allclose(env._model.geom_margin[ids], .001)


def test_handoff_exports_named_motor_commands_and_wxyz_wrist_frame(hand, tmp_path):
    _, meta = hand
    annotation = tmp_path / "assets/hand/fdh6"
    annotation.mkdir(parents=True)
    (annotation / "hand_meta.json").write_text(json.dumps(meta))
    directory = tmp_path / "run/grasp_data/template/object"
    directory.mkdir(parents=True)
    active = np.array([-1.3, .42, -.31, -.52, -.64, -.75])
    q = np.r_[.1, .2, .3, np.sqrt(.5), 0, 0, np.sqrt(.5), expand_hand_q(active, meta)]
    np.save(directory / "pose.npy", dict(tmpl_name="template", pregrasp_qpos=np.stack([q, q]),
        grasp_qpos=q[None], squeeze_qpos=q[None]))
    exported = export_grasp_commands(tmp_path)
    assert exported["active_joint_names"] == meta["active_joints"]
    commands = exported["grasps"][0]["phases"]["pregrasp"]
    assert len(commands) == 2
    np.testing.assert_allclose(commands[0]["active_joint_rad"], active)
    transform = np.array(commands[0]["T_scene_hand_base"])
    np.testing.assert_allclose(transform[:3, 3], [.1, .2, .3])
    np.testing.assert_allclose(transform[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert exported["grasps"][0]["official_force_closure_pass"] is False
