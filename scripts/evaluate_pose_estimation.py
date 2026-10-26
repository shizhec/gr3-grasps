#!/usr/bin/env python3
"""Head-camera known-mesh pose estimation error on the Dexonomy scene set (no arm motion)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from pathlib import Path

import numpy as np
import yaml
from gr3_grasping.cli import write_json
from gr3_grasping.dexonomy_execution import build_experiment
from gr3_grasping.geometry import transform
from gr3_grasping.model import ROOT
from gr3_grasping.pose_estimation import DEFAULTS, model_points, observe_object_pose, pose_errors
from gr3_grasping.simulation import Simulation


def any_candidate(manifest, object_id):
    synthesis = ROOT / manifest["pools"]["0"][0]["synthesis"]
    for entry in manifest["pools"]["0"]:
        found = sorted((ROOT / entry["synthesis"] / "run/grasp_data").glob(f"*/{object_id}/*_grasp.npy"))
        if found:
            return found[0]
    raise FileNotFoundError(f"No candidate for {object_id} under {synthesis}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/dexonomy_scenes6.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-noise", type=float, default=0.0)
    parser.add_argument("--depth-dropout", type=float, default=0.0)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    settings = yaml.safe_load((ROOT / manifest["arm_config"]).read_text())
    items = {i["id"]: i for i in yaml.safe_load((ROOT / "configs/benchmark10.yaml").read_text())["objects"]}
    rows = []
    for object_id in manifest["objects"]:
        for yaw in manifest["yaws_deg"]:
            directory = args.output / f"{object_id}_yaw{yaw:03d}"
            directory.mkdir(parents=True, exist_ok=False)
            item = items[object_id]
            s = {**settings, "candidate": str(any_candidate(manifest, object_id)), "object_id": object_id,
                 "object_folder": item["folder"], "mass_kg": item["mass_kg"], "yaw_deg": yaw}
            path, cfg, *_ = build_experiment(s, directory / "model")
            cfg["camera"].update(depth_noise_std_m=args.depth_noise, depth_dropout=args.depth_dropout)
            with Simulation(path, cfg, render=True) as sim:
                sim.settle()
                body = sim.model.body("target").id
                T_true = transform(sim.data.xmat[body].reshape(3, 3), sim.data.xpos[body])
                T_est, diag = observe_object_pose(sim, cfg, directory)
                errors = pose_errors(T_est, T_true, model_points(sim.model, body, DEFAULTS["model_voxel_m"]))
            row = dict(object_id=object_id, yaw_deg=yaw, **errors, best=diag["best"],
                       feasible=len(diag["feasible"]),
                       fused_target_points=diag["fused_target_points"], accepted_views=diag["accepted_views"])
            write_json(directory / "errors.json", dict(T_true=T_true, T_est=T_est, **row))
            rows.append(row)
            print(f"{object_id:16s} {yaw:3d}  xy {errors['translation_xy_m']*1000:5.1f} mm  "
                  f"yaw {errors['yaw_error_deg']:7.1f}  ADD {errors['add_m']*1000:5.1f}  "
                  f"ADD-S {errors['add_s_m']*1000:5.1f} mm  color {diag['best']['color_error']:.3f}  "
                  f"feasible {len(diag['feasible']):2d}", flush=True)
    write_json(args.output / "summary.json", dict(depth_noise_std_m=args.depth_noise,
               depth_dropout=args.depth_dropout, rows=rows,
               median_add_s_m=float(np.median([r["add_s_m"] for r in rows])),
               max_add_s_m=float(np.max([r["add_s_m"] for r in rows]))))


if __name__ == "__main__":
    main()
