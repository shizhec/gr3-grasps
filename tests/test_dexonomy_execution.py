"""Whole-hand contact policy for the GR3 Dexonomy executor."""
import mujoco
import numpy as np
import json
import subprocess
import sys
from dataclasses import asdict
import pytest
import yaml
from gr3_grasping.dexonomy_execution import (forbidden_contacts, interpolate_hand_pose, pose,
    build_experiment, DexterousPlan, MotorExecution)
from gr3_grasping.dexonomy_backend import ROOT, HAND_SOURCE, export_hand, export_object
from gr3_grasping.simulation import Simulation
from gr3_grasping.cli import write_json


def contact_scene(other_name):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <body name="{other_name}" pos="0 0 1"><freejoint/><geom name="robot" type="sphere" size=".05"/></body>
      <body name="target" pos=".098 0 1"><freejoint/><geom name="object" type="sphere" size=".05"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, model.body(other_name).id, model.body("target").id


def test_palm_contact_is_allowed_only_in_contact_phases_and_bounded():
    model, data, hand, target = contact_scene("R_hand_base_link")
    assert forbidden_contacts(model, data, {hand}, {hand}, target, False)
    assert not forbidden_contacts(model, data, {hand}, {hand}, target, True)
    data.qpos[7] = .096  # 4 mm overlap remains forbidden for intended contacts.
    mujoco.mj_forward(model, data)
    assert forbidden_contacts(model, data, {hand}, {hand}, target, True)


def test_contact_permission_never_allows_object_forearm_or_fixed_torso_collision():
    for name, moving in (("right_forearm", True), ("torso", False)):
        model, data, robot, target = contact_scene(name)
        result = forbidden_contacts(model, data, {robot} if moving else set(), set(), target, True)
        assert result and result[0]["penetration_m"] > .001
    # A hypothetically carried target must also be checked against world/table.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="1 1 .1"/>
      <body name="target" pos="0 0 .046"><freejoint/><geom type="sphere" size=".05"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert forbidden_contacts(model, data, set(), set(), model.body("target").id, True)


def test_hand_interpolation_uses_shortest_quaternion_path():
    a = np.r_[0., 0, 0, 1, 0, 0, 0, -.4, .2]
    b = np.r_[1., 2, 3, -1, 0, 0, 0, -.8, .6]
    q = interpolate_hand_pose(a, b, .5)
    np.testing.assert_allclose(q[:3], [.5, 1, 1.5])
    np.testing.assert_allclose(q[7:], [-.6, .4])
    np.testing.assert_allclose(pose(q)[:3, :3], np.eye(3), atol=1e-12)


@pytest.fixture(scope="module")
def full_arm_case(tmp_path_factory):
    if mujoco.__version__ != "3.16.0":
        pytest.skip("Recorded full arm dynamics require MuJoCo 3.16.0")
    pytest.importorskip("trimesh")
    settings = yaml.safe_load((ROOT / "configs/dexonomy_gr3_mug.yaml").read_text())
    if not HAND_SOURCE.exists() or not (ROOT / "third_party/mujoco_scanned_objects/models" / settings["object_folder"]).exists():
        pytest.skip("Fetch official GR3/FDH-6 and scanned cup assets")
    directory = tmp_path_factory.mktemp("gr3_motor_case")
    _, meta = export_hand(directory / "hand")
    item = next(i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"] if i["id"] == "coffee_mug")
    scene = export_object(item, directory / "object")
    fixture = json.loads((ROOT / "tests/fixtures/dexonomy_gr3_mug.json").read_text())
    source = np.load(scene, allow_pickle=True).item()["scene"]["coffee_mug"]
    np.testing.assert_allclose(source["pose"], fixture["object_pose"], atol=1e-12)
    candidate = directory / "grasp.npy"
    payload = {key: np.asarray(fixture[key]) for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")}
    payload.update(scene_path=str(scene))
    np.save(candidate, payload)
    settings["candidate"] = str(candidate)
    settings["hand_metadata"] = str(directory / "hand/hand_meta.json")
    path, cfg, grasp, _, provenance = build_experiment(settings, directory / "model")
    plans = {k: DexterousPlan(np.array(v["arm"]), np.array(v["hand"]), np.array(v["times"]),
        np.array(v["target_tcp"]), v["source"]) for k,v in fixture["plans"].items()}
    return path, cfg, settings, meta, grasp, plans, provenance


@pytest.mark.parametrize("no_close,expected", [(False, True), (True, False)])
def test_full_gr3_motor_lift_and_same_path_open_control(full_arm_case, tmp_path, no_close, expected):
    path, cfg, settings, meta, grasp, plans, _ = full_arm_case
    with Simulation(path, cfg, render=False) as sim:
        assert sim.model.nmocap == 0
        assert sim.model.neq == 10 and np.all(sim.model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
        assert sim.model.body_mass[sim.model.body("target").id] == pytest.approx(.3)
        executor = MotorExecution(sim, meta, tmp_path, settings, no_close=no_close)
        np.testing.assert_allclose(sim.model.actuator_forcerange[executor.hand_motors], np.tile([-.25,.25], (6,1)))
        sim.settle()
        active = [meta["joint_order"].index(n) for n in meta["active_joints"]]
        executor.preform(grasp["pregrasp_qpos"][0][7:][active])
        executor.execute(plans)
        result = executor.finish()
    assert result["success"] is expected
    assert result["hold_duration_s"] == pytest.approx(2.)
    assert result["max_target_penetration_m"] < .003
    assert result["max_mimic_error_rad"] < .001
    if expected:
        assert result["hold_min_lift_m"] >= .1
        assert result["hold_opposed_fraction"] >= .9
        assert not result["hold_non_hand_support"]
    else:
        assert result["peak_lift_m"] < .01
        assert result["hold_opposed_fraction"] == 0


def test_cli_replay_verifies_initial_state_and_prints_physics_arrays(full_arm_case, tmp_path):
    path, cfg, settings, meta, grasp, plans, provenance = full_arm_case
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    write_json(recorded / "plans.json", {k: asdict(v) for k,v in plans.items()})
    write_json(recorded / "provenance.json", provenance)
    with Simulation(path, cfg, render=False) as sim:
        executor = MotorExecution(sim, meta, recorded, settings)
        sim.settle()
        active = [meta["joint_order"].index(n) for n in meta["active_joints"]]
        executor.preform(grasp["pregrasp_qpos"][0][7:][active])
        np.savez(recorded / "planning_state.npz", qpos=sim.data.qpos, qvel=sim.data.qvel,
                 ctrl=sim.data.ctrl, time_s=float(sim.data.time))
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(settings))
    output = tmp_path / "cli"
    process = subprocess.run([sys.executable, str(ROOT / "scripts/run_dexonomy_gr3.py"),
        "--config", str(config), "--output", str(output), "--replay-plan", str(recorded), "--plan-only"],
        cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "execution_report.json").read_text())
    assert not result["success"] and result["failure"] == "plan_only_no_execution"
    assert result["physics"]["hand_motor_torque_range_nm"] == [[-.25,.25]]*6
    assert '"physics"' in process.stdout
    replay = json.loads((output / "replay_metadata.json").read_text())
    assert replay["initial_state_verified"] and replay["no_state_reset"]
    assert replay["compiled_model_sha256"] == provenance["compiled_model_sha256"]
    # Altering physical mass must reject replay before the arm moves.
    changed = dict(settings, mass_kg=.31)
    config.write_text(yaml.safe_dump(changed))
    rejected = tmp_path / "rejected"
    process = subprocess.run([sys.executable, str(ROOT / "scripts/run_dexonomy_gr3.py"),
        "--config", str(config), "--output", str(rejected), "--replay-plan", str(recorded), "--plan-only"],
        cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert process.returncode == 1
    result = json.loads((rejected / "execution_report.json").read_text())
    assert "Replay model or candidate differs" in result["failure"]
