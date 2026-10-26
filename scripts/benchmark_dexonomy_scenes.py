#!/usr/bin/env python3
"""Scene benchmark: per object/yaw, plan candidates in a fixed order and execute the first plannable one once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml
from gr3_grasping.cli import write_json
from gr3_grasping.dexonomy_backend import arm_prior_mask, arm_prior_metrics, load_arm_prior
from gr3_grasping.model import ROOT

RUNNER = [str(ROOT / ".venv-curobo/bin/python"), str(ROOT / "scripts/run_dexonomy_gr3.py")]


def candidate_pool(manifest, object_id, yaw):
    """Prior-accepted (at this yaw) candidates that pass the closed-hand floating lift audit."""
    pool = []
    for entry in manifest["pools"][str(yaw)]:
        synthesis, audit = ROOT / entry["synthesis"], ROOT / entry["audit"]
        prior = json.loads((synthesis / "arm_prior_final.json").read_text())
        experiment = json.loads((synthesis / "experiment.json").read_text())
        if experiment["signature"]["arm_prior"]["object_yaw_deg"] != yaw:
            raise RuntimeError(f"{synthesis} was synthesized for another yaw")
        lifted = {Path(r["source"]).resolve() for r in json.loads((audit / manifest["audit_summary"]).read_text())
                  if r["success"] and not r["no_close"]}
        for key, metrics in prior.items():
            path = (synthesis / "run/grasp_data" / key).resolve()
            if key.split("/")[1] == object_id and metrics["accepted"] and path in lifted:
                pool.append(dict(candidate=str(path), key=key,
                                 reach=max(metrics["reach_distance"], metrics["lift_reach_distance"])))
    return sorted(pool, key=lambda c: (c["reach"], c["candidate"]))


def camera_pool(manifest, object_id, world_scene):
    """Candidates from every synthesized yaw, re-checked against the prior at the estimated pose."""
    # The sampled reachability map depends on placement only, not on object yaw.
    prior = load_arm_prior(ROOT / manifest["camera"]["arm_prior"])
    prior.update(object_xy_world_m=np.asarray(world_scene)[:2, 3].tolist(),
                 object_yaw_deg=float(np.degrees(np.arctan2(world_scene[1][0], world_scene[0][0]))))
    pool, keys = [], []
    for yaw in manifest["pools"]:
        for entry in manifest["pools"][yaw]:
            synthesis, audit = ROOT / entry["synthesis"], ROOT / entry["audit"]
            lifted = {Path(r["source"]).resolve() for r in json.loads((audit / manifest["audit_summary"]).read_text())
                      if r["success"] and not r["no_close"]}
            for path in sorted((synthesis / "run/grasp_data").glob(f"*/{object_id}/*_grasp.npy")):
                if path.resolve() in lifted:
                    pool.append(path.resolve())
                    keys.append(f"yaw{yaw}/" + str(path.relative_to(synthesis / "run/grasp_data")))
    if not pool:
        return []
    q = np.array([np.load(p, allow_pickle=True).item()["grasp_qpos"][-1][:7] for p in pool])
    accepted, metrics = arm_prior_mask(q, prior), arm_prior_metrics(q, prior)
    reach = np.maximum(metrics["reach_distance"], metrics["lift_reach_distance"])
    result = [dict(candidate=str(p), key=k, reach=float(r)) for p, k, r, a in zip(pool, keys, reach, accepted) if a]
    return sorted(result, key=lambda c: (c["reach"], c["candidate"]))


def run(args, log, timeout):
    with open(log, "w") as stream:
        try:
            return subprocess.run(RUNNER + args, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                                  env={**os.environ, "MUJOCO_GL": "egl"}).returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def scene(manifest, output, object_id, yaw, camera=None):
    directory = output / f"{object_id}_yaw{yaw:03d}"
    if (directory / "scene.json").is_file():
        return json.loads((directory / "scene.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    common = ["--config", str(ROOT / manifest["arm_config"]), "--yaw-deg", str(yaw)]
    start = time.time()
    registration = None
    if camera is None:
        pool = candidate_pool(manifest, object_id, yaw)
    else:
        common += ["--registration", "camera", "--depth-noise", str(camera["depth_noise_std_m"]),
                   "--depth-dropout", str(camera["depth_dropout"])]
        # Any candidate of this object builds the identical scene; only the object model matters.
        anchor = next(c for e in manifest["pools"]["0"] for c in
                      sorted((ROOT / e["synthesis"] / "run/grasp_data").glob(f"*/{object_id}/*_grasp.npy")))
        run(common + ["--candidate", str(anchor), "--perceive-only", "--output", str(directory / "perceive")],
            directory / "perceive.log", manifest["timeout_s"])
        report = directory / "perceive/camera_registration.json"
        if not report.is_file():
            result = dict(object_id=object_id, yaw_deg=yaw, pool_size=0, attempts=[], executed=None,
                          success=False, failure="perception_failed", open_control=None, wall_time_s=time.time() - start)
            write_json(directory / "scene.json", result)
            return result
        registration = json.loads(report.read_text())
        pool = camera_pool(manifest, object_id, registration["T_world_scene"])
    result = dict(object_id=object_id, yaw_deg=yaw, pool_size=len(pool), attempts=[], executed=None,
                  success=False, failure=None, open_control=None,
                  pose_errors=registration and registration["evaluation_only_errors"])
    for i, c in enumerate(pool[:manifest["max_plan_attempts"]]):
        name = f"plan{i:02d}__" + c["key"].replace("/", "__").removesuffix("_grasp.npy")
        code = run(common + ["--candidate", c["candidate"], "--plan-only", "--output", str(directory / name)],
                   directory / f"{name}.log", manifest["timeout_s"])
        report = directory / name / "execution_report.json"
        failure = json.loads(report.read_text())["failure"] if report.is_file() else None
        planned = code == 0 and (directory / name / "plans.json").is_file()
        result["attempts"].append(dict(candidate=c["key"], reach=c["reach"], planned=planned, exit=code,
                                       failure=None if planned else failure))
        if not planned:
            continue
        for mode, extra in (("execute", []), ("open", ["--no-close"])):
            run(common + ["--candidate", c["candidate"], "--replay-plan", str(directory / name),
                          "--output", str(directory / mode)] + extra,
                directory / f"{mode}.log", manifest["timeout_s"])
        reports = {m: directory / m / "execution_report.json" for m in ("execute", "open")}
        closed = json.loads(reports["execute"].read_text()) if reports["execute"].is_file() else {}
        opened = json.loads(reports["open"].read_text()) if reports["open"].is_file() else {}
        result.update(executed=c["key"], success=bool(closed.get("success")),
                      failure=closed.get("failure", "missing_execution_report"),
                      hold_min_lift_m=closed.get("hold_min_lift_m"), peak_lift_m=closed.get("peak_lift_m"),
                      hold_opposed_fraction=closed.get("hold_opposed_fraction"),
                      max_target_penetration_m=closed.get("max_target_penetration_m"),
                      open_control=dict(success=opened.get("success"), peak_lift_m=opened.get("peak_lift_m")))
        break
    else:
        result["failure"] = "no_candidates" if not pool else f"no_plan_in_first_{len(result['attempts'])}"
    result["wall_time_s"] = time.time() - start
    write_json(directory / "scene.json", result)
    return result


def category(r):
    if r["success"]:
        return "success"
    if r["executed"] is None:
        return r["failure"]
    f = r["failure"] or ""
    if "Physical collision" in f:
        return "collision_during_execution"
    if "insufficient" in f:
        return "insufficient_lift_or_hold"
    return "execution_error"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/dexonomy_scenes6.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--yaws", type=int, nargs="+")
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--registration", choices=["oracle", "camera"], default="oracle")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    jobs = [(o, y) for y in (args.yaws or manifest["yaws_deg"]) for o in (args.objects or manifest["objects"])]
    with ThreadPoolExecutor(args.jobs) as pool:
        camera = manifest["camera"] if args.registration == "camera" else None
        results = list(pool.map(lambda job: scene(manifest, output, *job, camera), jobs))
    for r in results:
        r["category"] = category(r)
        print(f"{r['object_id']:16s} yaw {r['yaw_deg']:3d}  pool {r['pool_size']:3d}  "
              f"tries {len(r['attempts']):2d}  {r['category']}", flush=True)
    write_json(output / "summary.json", dict(
        success=sum(r["success"] for r in results), scenes=len(results),
        open_control_success=sum(bool(r["open_control"] and r["open_control"]["success"]) for r in results),
        registration=args.registration,
        scope=("One physical execution per scene; known mesh, fixed base; " + (
            "head RGB-D known-mesh pose estimate (true pose for evaluation only)" if args.registration == "camera"
            else "simulator object pose, no camera")),
        results=results))
    columns = ["object_id", "yaw_deg", "pool_size", "category", "executed", "hold_min_lift_m",
               "hold_opposed_fraction", "max_target_penetration_m"]
    with open(output / "results.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns + ["plan_attempts", "open_peak_lift_m"])
        for r in results:
            writer.writerow([r.get(k) for k in columns] + [len(r["attempts"]),
                            r["open_control"]["peak_lift_m"] if r["open_control"] else None])


if __name__ == "__main__":
    main()
