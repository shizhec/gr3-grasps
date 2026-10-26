"""Physics regression: RGB-D -> FDH-6 grasp, plus a fingers-open control."""
from types import SimpleNamespace

# Configure headless EGL before importing MuJoCo.
from gr3_grasping.cli import write_json
import mujoco
import numpy as np
import pytest
import yaml

from gr3_grasping.grasping import FDH6, run_grasp
from gr3_grasping.grasps import fdh6_grasps
from gr3_grasping.model import ROOT, build_scene
from gr3_grasping.perception import segment_tabletop
from gr3_grasping.simulation import Simulation


@pytest.fixture(scope="module")
def fdh6_scene(tmp_path_factory):
    config = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
    if not (ROOT / config["robot"]["urdf"]).exists():
        pytest.skip("Fetch official assets before the integration tests")
    scene = build_scene(config, tmp_path_factory.mktemp("fdh6_model"))
    with Simulation(scene, config) as sim:
        sim.settle()
        clouds = segment_tabletop(sim.capture(), config["perception"])
        assert len(clouds) == 3
        target = clouds[0]
        proposals = fdh6_grasps(target, config["scene"]["table_top_m"])
    return config, scene, proposals, target.centroid_world


def test_official_fdh6_has_six_drives_five_mimics_and_free_objects(fdh6_scene):
    config, scene, _, _ = fdh6_scene
    with Simulation(scene, config, render=False) as sim:
        model = sim.model
        assert model.nv == 31 + 2 * 11 + 3 * 6
        assert model.nu == 31 + 2 * 6
        assert model.neq == 10
        # No weld/connect/equality that can hold an object to the robot.
        assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
        for obj in config["scene"]["objects"]:
            assert model.joint(f"{obj['name']}_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
        for side in ("right", "left"):
            hand = FDH6(model, side)
            assert len(hand.actuators) == 6
            np.testing.assert_allclose(model.actuator_forcerange[hand.actuators, 1], config["hand"]["force_limit_nm"])


@pytest.mark.parametrize("no_close", [False, True])
def test_contact_lift_and_fingers_open_negative_control(fdh6_scene, tmp_path, no_close):
    config, scene, proposals, centroid = fdh6_scene
    with Simulation(scene, config, render=False) as sim:
        sim.settle()
        report = {"arm": "right", "hand": "fdh6", "selected_centroid_world_m": centroid}
        args = SimpleNamespace(plan_only=False, no_close=no_close)
        result = run_grasp(sim, proposals, report, args, tmp_path, write_json)
        assert result["grasp_executed"]
        assert result["grasp_success"] is (not no_close)
        if no_close:
            assert abs(result["final_lift_m"]) < 0.005
            assert result["hold_opposed_contact_fraction"] == 0
        else:
            assert result["hold_min_lift_m"] >= 0.10
            assert result["hold_duration_s"] >= 2
            assert result["hold_opposed_contact_fraction"] >= 0.9
            assert result["max_target_finger_penetration_m"] < 0.003
            # Mimic joints must still follow their source under contact loads.
            for eq in range(sim.model.neq):
                j1, j2 = sim.model.eq_obj1id[eq], sim.model.eq_obj2id[eq]
                q1, q2 = sim.data.qpos[sim.model.jnt_qposadr[[j1, j2]]]
                assert abs(q1 - sim.model.eq_data[eq, 1] * q2) < 0.01
