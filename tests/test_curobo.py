"""Validate the frozen URDF and, when installed, GPU/MuJoCo frame agreement."""
import xml.etree.ElementTree as ET

from gr3_grasping.cli import write_json  # Configure headless rendering.
import numpy as np
import pytest
import yaml

from gr3_grasping.curobo_backend import export_robot
from gr3_grasping.geometry import rotation_error, transform
from gr3_grasping.model import ROOT, arm_names, build_scene
from gr3_grasping.planning import Planner
from gr3_grasping.simulation import Simulation


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    config = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
    if not (ROOT / config["robot"]["urdf"]).exists():
        pytest.skip("Official robot assets required")
    scene = build_scene(config, tmp_path_factory.mktemp("curobo_scene"))
    with Simulation(scene, config, render=False) as sim:
        sim.settle()
        yield config, sim


@pytest.mark.parametrize("side", ["right", "left"])
def test_export_preserves_only_seven_arm_drives_and_tcp(robot, tmp_path, side):
    config, sim = robot
    result = export_robot(sim.model, sim.data, config, side, tmp_path)["robot_cfg"]["kinematics"]
    root = ET.parse(result["urdf_path"]).getroot()
    drives = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    assert set(drives) == set(arm_names(side))
    assert root.find(f"link[@name='{side}_tcp']") is not None
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            assert joint.find("mimic") is None
    assert result["collision_spheres"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_gpu_fk_agrees_with_mujoco_across_joint_configurations(robot, tmp_path, side):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("Use .venv-curobo for GPU integration tests")
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from scipy.spatial.transform import Rotation

    config, sim = robot
    result = export_robot(sim.model, sim.data, config, side, tmp_path)
    gpu = Kinematics(KinematicsCfg.from_data_dict(result["robot_cfg"]["kinematics"]))
    cpu = Planner(sim.model, sim.data, config["planning"], side)
    rng = np.random.default_rng(72)
    base = sim.model.body("base_link").id
    base_pose = transform(sim.data.xmat[base].reshape(3, 3), sim.data.xpos[base])
    for q in rng.uniform(cpu.limits[:, 0], cpu.limits[:, 1], size=(12, 7)):
        tensor = torch.tensor([[q[cpu.names.index(n)] for n in gpu.joint_names]], device="cuda", dtype=torch.float32)
        state = gpu.compute_kinematics(JointState.from_position(tensor, joint_names=gpu.joint_names))
        pose = state.tool_poses.get_link_pose(f"{side}_tcp")
        p = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        observed = base_pose @ transform(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(), p)
        expected = cpu.pose(q)
        assert np.linalg.norm(expected[:3, 3] - observed[:3, 3]) < 1e-5
        assert np.linalg.norm(rotation_error(expected[:3, :3], observed[:3, :3])) < 1e-4


def test_gpu_motion_path_is_rechecked_and_obeys_shared_speed_bound(robot, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("Use .venv-curobo for GPU integration tests")
    from gr3_grasping.curobo_backend import CuroboPlanner
    from gr3_grasping.grasps import Grasp

    config, sim = robot
    planner = CuroboPlanner(sim.model, sim.data, config, directory=tmp_path)
    start = sim.data.qpos[planner.qids].copy()
    # A nearby reachable target exercises IK, TrajOpt, interpolation and checks.
    q_goal = start.copy()
    q_goal[0] += .15
    target = planner.pose(q_goal)
    direction = target[:3, 2]
    contact = target.copy()
    contact[:3, 3] += config["planning"]["pregrasp_distance_m"] * direction
    plan = planner.plan([Grasp(contact, 1., "reachable_test")], start)
    assert plan.checked_samples == len(plan.waypoints)
    assert plan.ik_position_error_m < config["planning"]["position_tolerance_m"]
    assert all(not planner.collisions(q) for q in plan.waypoints)
    speed = np.abs(np.diff(plan.waypoints, axis=0)) / np.diff(plan.timestamps_s)[:, None]
    assert speed.max() <= config["planning"]["max_joint_speed_rad_s"] + 1e-6
