#!/usr/bin/env python3
"""Sample GR3 right-arm forward kinematics into a hand-base reachability map.

Uniform joint samples within official limits; the rest of the robot keeps the
configs/sim.yaml initial pose. Only poses near the object placement are kept.
Self/table collisions are not checked here: the arm planner still does that.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from multiprocessing import Pool
from pathlib import Path
import tempfile

import mujoco
import numpy as np
import yaml

from gr3_grasping.model import ROOT, arm_names, build_scene

_model = None


def _init(scene_path, initial_joints):
    global _model, _data, _qids, _limits, _body
    _model = mujoco.MjModel.from_xml_path(scene_path)
    _data = mujoco.MjData(_model)
    for name, value in initial_joints.items():
        _data.qpos[_model.joint(name).qposadr[0]] = value
    names = arm_names("right")
    _qids = np.array([_model.joint(n).qposadr[0] for n in names])
    _limits = _model.jnt_range[[_model.joint(n).id for n in names]]
    _body = _model.body("R_hand_base_link").id


def _sample(job):
    seed, count, center, radius = job
    rng = np.random.default_rng(seed)
    q = rng.uniform(_limits[:, 0], _limits[:, 1], (count, 7))
    result = []
    for row in q:
        _data.qpos[_qids] = row
        mujoco.mj_kinematics(_model, _data)
        position = _data.xpos[_body]
        if np.linalg.norm(position - center) <= radius:
            result.append(np.r_[position, _data.xquat[_body], row])
    return np.asarray(result).reshape(-1, 14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=ROOT / "configs/dexonomy_arm_prior.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8_000_000)
    parser.add_argument("--radius", type=float, default=0.30, help="Kept hand-base distance from the object anchor")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    prior = yaml.safe_load(args.prior.read_text())
    config = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
    config["scene"]["objects"] = []
    center = np.array([*prior["object_xy_world_m"], prior["table_top_world_m"] + 0.15])
    with tempfile.TemporaryDirectory() as directory:
        scene = str(build_scene(config, directory))
        chunk = 100_000
        jobs = [(seed, chunk, center, args.radius) for seed in range(int(np.ceil(args.samples / chunk)))]
        with Pool(args.jobs, _init, (scene, config["robot"]["initial_joints"])) as pool:
            rows = np.concatenate(pool.map(_sample, jobs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, position_world_m=rows[:, :3], quat_world_wxyz=rows[:, 3:7],
                        arm_q_rad=rows[:, 7:], anchor_world_m=center, radius_m=args.radius,
                        joint_samples=len(jobs) * chunk, body="R_hand_base_link",
                        scope="uniform right-arm joint samples; fixed base/waist; no collision check")
    print(f"kept {len(rows)} of {len(jobs) * chunk} samples within {args.radius} m -> {args.output}")


if __name__ == "__main__":
    main()
