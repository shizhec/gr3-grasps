#!/usr/bin/env python3
"""Offline oracle registration and GR3 endpoint IK diagnostic, without motion."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from gr3_grasping.geometry import transform
from gr3_grasping.model import ROOT, build_scene
from gr3_grasping.planning import Planner
from gr3_grasping.simulation import Simulation


def pose(q):
    return transform(Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix(), q[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/dexonomy_fdh6_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dexonomy_dynamics_audit/gr3_ik")
    args = parser.parse_args()
    root, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())
    base = yaml.safe_load((ROOT / "configs/scanned_candy.yaml").read_text())
    meta = json.loads((root / "assets/hand/fdh6/hand_meta.json").read_text())
    results = []
    for item in manifest["objects"]:
        paths = sorted((root / "run/grasp_data").glob(f"*/{item['id']}/*.npy"))
        if not paths:
            continue
        cfg = deepcopy(base)
        cfg["scene"]["objects"] = [dict(name="target", type="mjcf",
            mjcf=f"third_party/mujoco_scanned_objects/models/{item['folder']}/model.xml",
            placement_xy=manifest["placement_xy"], rotation_rpy_deg=[0, 0, 0], mass=item["mass_kg"])]
        scene = build_scene(cfg, output / item["id"] / "model")
        with Simulation(scene, cfg, render=False) as sim:
            sim.settle()
            model, data = sim.model, sim.data
            body = model.body("target").id
            base_id, tcp = model.body("R_hand_base_link").id, model.site("right_tcp").id
            T_world_base = transform(data.xmat[base_id].reshape(3, 3), data.xpos[base_id])
            T_world_tcp = transform(data.site_xmat[tcp].reshape(3, 3), data.site_xpos[tcp])
            T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp
            T_world_object = transform(data.xmat[body].reshape(3, 3), data.xpos[body])
            hand_ids = [model.joint(n).qposadr[0] for n in meta["joint_order"]]
            planner = Planner(model, data, cfg["planning"], "right")
            initial = data.qpos[planner.qids].copy()
            for path in paths:
                grasp = np.load(path, allow_pickle=True).item()
                object_pose = np.load(grasp["scene_path"], allow_pickle=True).item()["scene"][item["id"]]["pose"]
                T_world_scene = T_world_object @ np.linalg.inv(pose(np.asarray(object_pose)))
                stages, seed = {}, initial.copy()
                for phase in ("pregrasp", "grasp", "squeeze", "lift"):
                    q = grasp["pregrasp_qpos"][0].copy() if phase == "pregrasp" else grasp["grasp_qpos"][0].copy() if phase == "grasp" else grasp["squeeze_qpos"][0].copy()
                    if phase == "lift":
                        q[2] += .16
                    planner.data.qpos[hand_ids] = q[7:]
                    target = T_world_scene @ pose(q) @ T_base_tcp
                    best = None
                    for attempt in range(4):
                        guess = seed if attempt == 0 else initial + planner.rng.normal(0, .7, 7)
                        solution, pe, re, valid = planner.ik(target, guess)
                        result = dict(reachable=bool(valid), position_error_m=pe, rotation_error_rad=re,
                            arm_q_rad=solution.tolist(), T_world_tcp=target.tolist())
                        if best is None or pe + .2*re < best["position_error_m"] + .2*best["rotation_error_rad"]:
                            best = result
                        if valid:
                            seed, best = solution, result
                            best["legacy_collision_rejections"] = planner.collisions(solution, body if phase != "pregrasp" else None)
                            break
                    stages[phase] = best
                results.append(dict(source=str(path), object_id=item["id"], stages=stages,
                    all_endpoints_reachable=all(s["reachable"] for s in stages.values())))
    report = dict(scope="Offline object-pose oracle registration; bounded SciPy endpoint IK diagnostic only. No cuRobo path planning, arm execution or camera input.",
        placement_xy=manifest["placement_xy"], yaw_deg=0, attempts_per_endpoint=4, results=results)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(total=len(results), all_endpoints_reachable=sum(r["all_endpoints_reachable"] for r in results)), indent=2))


if __name__ == "__main__":
    main()
