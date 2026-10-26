from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gr3_grasping.benchmark import select_target_index, summarize, trial_config
from gr3_grasping.model import ROOT


def test_success_rate_includes_unexecuted_and_incomplete_trials():
    rows = [
        {"object_id": "a", "label": "A", "yaw_deg": 0, "grasp_success": True,
         "grasp_executed": True, "failure_category": "success"},
        {"object_id": "a", "label": "A", "yaw_deg": 45, "grasp_success": False,
         "grasp_executed": False, "failure_category": "planning_failure"},
        {"object_id": "b", "label": "B", "yaw_deg": 0, "grasp_success": False,
         "grasp_executed": False, "failure_category": "timeout"},
    ]
    summary = summarize(rows, expected_trials=4)
    assert summary["success_rate"] == .25  # Neither 1/1 executed nor 1/3 completed.
    assert summary["complete"] is False
    assert summary["executed_trials"] == 1
    assert summary["objects"][0]["successes"] == 1
    assert summary["outcomes"]["planning_failure"] == 1


def test_fixed_target_region_rejects_larger_robot_foreground():
    hand = SimpleNamespace(centroid_world=np.array([.31, -.39, 1.1]), points_world=np.zeros((5000, 3)))
    target = SimpleNamespace(centroid_world=np.array([.45, -.19, 1.1]), points_world=np.zeros((300, 3)))
    assert select_target_index([hand, target], [.46, -.20]) == 1
    with pytest.raises(RuntimeError, match="fixed target region"):
        select_target_index([hand], [.46, -.20])


def test_ten_distinct_scans_share_hand_camera_and_planning_settings():
    manifest = yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())
    assert len({obj["folder"] for obj in manifest["objects"]}) == 10
    assert manifest["yaws_deg"] == [0, 45, 90]
    first = trial_config(manifest, manifest["objects"][0], 0)
    last = trial_config(manifest, manifest["objects"][-1], 90)
    for key in ("robot", "hand", "camera", "planning", "m2t2", "perception"):
        assert first[key] == last[key]
    assert len(first["scene"]["objects"]) == len(last["scene"]["objects"]) == 1
    assert first["scene"]["objects"][0]["rotation_rpy_deg"] == [0, 0, 0]
    assert last["scene"]["objects"][0]["rotation_rpy_deg"] == [0, 0, 90]
    assert first["hand"]["evaluation_height_reference"] == "center_of_mass"


def test_single_object_evaluation_allows_offset_scan_origin():
    import mujoco
    from gr3_grasping.grasping import associate_evaluation_body

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="hammer" pos="0 0 1"><freejoint/>
        <geom type="box" size=".15 .02 .02"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = SimpleNamespace(model=model, data=data, config={"scene": {"objects": [{"name": "hammer"}]}})
    # The visible head may be more than the previous 8 cm gate from its origin.
    assert associate_evaluation_body(sim, np.array([.14, 0, 1])) == model.body("hammer").id


def test_planner_and_offset_overrides_preserve_limits_and_isolate_trials():
    manifest = yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())
    manifest["planning"] = {"backend": "curobo"}
    manifest["fdh6_contact_offset_m"] = [-.022, 0, -.05]
    first = trial_config(manifest, manifest["objects"][0], 0)
    second = trial_config(manifest, manifest["objects"][1], 45)
    assert first["planning"]["backend"] == second["planning"]["backend"] == "curobo"
    assert first["planning"]["max_joint_speed_rad_s"] == .35
    assert first["planning"]["collision_penetration_m"] == .001
    first["m2t2"]["fdh6_contact_offset_m"][0] = 99
    assert second["m2t2"]["fdh6_contact_offset_m"] == [-.022, 0, -.05]
    assert manifest["fdh6_contact_offset_m"] == [-.022, 0, -.05]
