"""Paired replay of observed M2T2 candidates to calibrate an FDH-6 offset."""
from pathlib import Path
import hashlib
import json
import time
from types import SimpleNamespace

from .cli import write_json  # Configure headless MuJoCo before imports.
import mujoco
import numpy as np
import yaml

from .benchmark import failure_category
from .grasps import Grasp
from .grasping import run_grasp
from .head_scan import move_head
from .m2t2_backend import retarget_fdh6
from .model import build_scene
from .simulation import Simulation

BASE_OFFSET = np.array([-.012, 0., -.045])


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offset_grid(center=BASE_OFFSET, step=.01):
    """A fixed six-axis neighbourhood, including its unmodified center."""
    return [np.asarray(center).copy()] + [np.asarray(center) + sign * step * axis
                                         for axis in np.eye(3) for sign in (-1, 1)]


def load_offset_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile.get("schema") != "gr3_fdh6_wrist_offset_v1" or profile.get("side") != "right":
        raise ValueError("Expected a right-hand gr3_fdh6_wrist_offset_v1 calibration profile")
    value = np.asarray(profile["contact_offset_tcp_m"], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value - BASE_OFFSET) > .04:
        raise ValueError("Invalid FDH-6 contact offset or correction exceeds the 4 cm calibration bound")
    return value.tolist()


def prepare_replay(source, output):
    """Restore the physical scan motion; model predictions remain byte-identical."""
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [source / name for name in ("config.yaml", "target_cloud.npz", "m2t2_source_grasps.json")]
    identity = {str(p): file_sha(p) for p in paths}
    input_state = source / "planning_start_state.npz"
    scan_path = source / "head_scan/scan.json"
    if input_state.is_file():
        identity[str(input_state)] = file_sha(input_state)
    elif scan_path.is_file():
        identity[str(scan_path)] = file_sha(scan_path)
    if (output / "replay.json").is_file():
        previous = json.loads((output / "replay.json").read_text())
        if previous["input_sha256"] != identity:
            raise ValueError(f"Replay input changed: {source}")
        return previous
    config = yaml.safe_load((source / "config.yaml").read_text())
    scene = build_scene(config, output / "model")
    with Simulation(scene, config, render=False) as sim:
        if input_state.is_file():
            state = np.load(input_state)["integration_state"]
            mujoco.mj_setState(sim.model, sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(sim.model, sim.data)
            restoration = "saved integration state"
        else:
            sim.settle()
            if scan_path.is_file():
                scan = json.loads(scan_path.read_text())
                for view in scan["views"][1:]:
                    move_head(sim, view["command_rad"], scan["settings"])
                move_head(sim, scan["head_restoration"]["command_rad"], scan["settings"])
            restoration = "replayed recorded head servo commands; no rerender or network inference"
        state = np.zeros(mujoco.mj_stateSize(sim.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(sim.model, sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(output / "start_state.npz", integration_state=state)
    record = {"source": str(source), "scene": str(scene), "directory": str(output),
              "input_sha256": identity, "state_restoration": restoration,
              "initial_state_sha256": file_sha(output / "start_state.npz")}
    write_json(output / "replay.json", record)
    return record


def run_replay(replay, offset, output, planner="scipy", render=False, no_close=False):
    """Execute exactly one first-feasible candidate, with no physical retry."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(replay["source"])
    config = yaml.safe_load((source / "config.yaml").read_text())
    config["m2t2"]["fdh6_contact_offset_m"] = list(offset)
    config["planning"]["backend"] = planner
    cloud = np.load(source / "target_cloud.npz")["points_world"]
    np.savez_compressed(output / "target_cloud.npz", points_world=cloud)
    source_grasps = json.loads((source / "m2t2_source_grasps.json").read_text())
    grasps = [Grasp(np.asarray(g["T_world_tcp"]), g["score"], g["source"]) for g in source_grasps]
    proposals = retarget_fdh6(grasps, "right", offset)
    report = {"stage": "perception_complete", "arm": "right", "hand": "fdh6",
              "grasp_executed": False, "grasp_success": False, "selected_centroid_world_m": np.median(cloud, axis=0),
              "contact_offset_tcp_m": list(offset), "source_trial": str(source), "planning_backend": planner,
              "input_sha256": replay["input_sha256"], "initial_state_sha256": replay["initial_state_sha256"]}
    write_json(output / "run_config.json", config)
    write_json(output / "grasps.json", [vars(g) for g in proposals])
    try:
        with Simulation(replay["scene"], config, render=render) as sim:
            state = np.load(Path(replay["directory"]) / "start_state.npz")["integration_state"]
            mujoco.mj_setState(sim.model, sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(sim.model, sim.data)
            try:
                args = SimpleNamespace(plan_only=False, no_close=no_close)
                if render:
                    import imageio.v2 as imageio
                    with imageio.get_writer(output / "grasp.mp4", fps=15, macro_block_size=1) as video:
                        run_grasp(sim, proposals, report, args, output, write_json, video)
                else:
                    run_grasp(sim, proposals, report, args, output, write_json)
            finally:
                if render:
                    from PIL import Image
                    Image.fromarray(sim.grasp_view(sim.model.body("benchmark_target").id, report["stage"])).save(output / "final_grasp.png")
    except Exception as error:
        report.update(failed_stage=report["stage"], stage="failed", error=str(error), grasp_success=False)
    report["wall_time_s"] = time.perf_counter() - started
    report["failure_category"] = failure_category(report)
    write_json(output / "report.json", report)
    return report


def calibration_score(reports, offset):
    """Select on calibration outcomes only, then prefer smaller corrections."""
    successes = sum(bool(r.get("grasp_success")) for r in reports)
    # Holding contact breaks success-count ties; clamped height penalizes drops.
    contact = sum(r.get("hold_opposed_contact_fraction", 0.) for r in reports)
    height = sum(np.clip(r.get("hold_min_lift_m", -.1), -.1, .16) for r in reports)
    return successes, contact, height, -float(np.linalg.norm(np.asarray(offset) - BASE_OFFSET))
