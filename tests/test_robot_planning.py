from pathlib import Path

import numpy as np
import pytest
import yaml

from gr3_grasping.model import ROOT, build_scene
from gr3_grasping.planning import Planner
from gr3_grasping.simulation import Simulation


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    config = yaml.safe_load((ROOT / "configs/dummy.yaml").read_text())
    if not (ROOT / config["robot"]["urdf"]).is_file():
        pytest.skip("Fetch official assets before the integration tests")
    scene = build_scene(config, tmp_path_factory.mktemp("model"))
    with Simulation(scene, config, render=False) as sim:
        sim.settle()
        yield sim, Planner(sim.model, sim.data, config["planning"]), config


def test_import_preserves_native_camera_and_31_body_joints(robot):
    sim, planner, _ = robot
    assert sim.model.nv == 31 + 3 * 6
    assert sim.model.body("camera_link").id > 0
    assert sim.model.camera("native_camera").id >= 0
    assert not planner.collisions(sim.data.qpos[planner.qids])


def test_bounded_ik_recovers_reachable_pose_and_rejects_far_target(robot):
    sim, planner, _ = robot
    start = sim.data.qpos[planner.qids].copy()
    target = planner.pose(start + [0.03, 0, 0.03, -0.08, 0, 0, 0]).copy()
    q, position_error, rotation_error, success = planner.ik(target, start)
    assert success and position_error < 0.003 and rotation_error < 0.04
    assert np.all(q >= planner.limits[:, 0]) and np.all(q <= planner.limits[:, 1])
    target[:3, 3] = [5, 5, 5]
    assert not planner.ik(target, start)[3]


def test_path_checker_detects_hand_penetrating_table(robot):
    sim, planner, _ = robot
    start = sim.data.qpos[planner.qids].copy()
    assert not planner.collisions(start)
    colliding = start.copy()
    colliding[[0, 3]] = [-0.2, -0.6]
    contacts = planner.collisions(colliding)
    assert contacts
    assert any("table" in (contact["geom1"], contact["geom2"]) for contact in contacts)
