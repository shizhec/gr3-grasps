#!/usr/bin/env python3
"""Tune one shared offset on yaw=0, then evaluate untouched yaw=45/90."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gr3_grasping.cli import write_json
from gr3_grasping.wrist_calibration import (BASE_OFFSET, calibration_score, offset_grid,
                                           prepare_replay, run_replay)


def evaluate(jobs, workers):
    result = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for key, replay, offset, output in jobs:
            path = output / "report.json"
            if path.is_file():
                report = json.loads(path.read_text())
                if report["initial_state_sha256"] != replay["initial_state_sha256"] or report["contact_offset_tcp_m"] != list(offset):
                    raise ValueError(f"Cached calibration report differs: {path}")
                result[key] = report
            else:
                futures[pool.submit(run_replay, replay, offset, output)] = key
        for future in as_completed(futures):
            key = futures[future]
            report = future.result()
            result[key] = report
            print(key, report["failure_category"], f"{report['wall_time_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/benchmark10_head_scan_context"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wrist_calibration"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trials = sorted((args.source / "trials").glob("*/m2t2_source_grasps.json"))
    if len(trials) != 30:
        raise ValueError("Expected the fixed 10-object x 3-yaw benchmark")
    replays = {p.parent.name: prepare_replay(p.parent, args.output / "inputs" / p.parent.name) for p in trials}
    calibration = [key for key in replays if key.endswith("yaw000")]
    heldout = [key for key in replays if key not in calibration]
    offsets = offset_grid()
    protocol = {"calibration_trials": calibration, "heldout_trials": heldout,
                "coarse_offsets_tcp_m": offsets, "refine_step_m": None if args.no_refine else .005,
                "selection": "lexicographic successes, holding contacts, clamped holding height, smaller correction",
                "unchanged": ["saved point clouds", "M2T2 candidate order", "hand closing pose", "SciPy planner", "success criteria"],
                "physical_attempts_per_trial_and_offset": 1, "unseen_object_evaluation": False}
    write_json(args.output / "protocol.json", protocol)
    reports = evaluate([(f"p{i:02d}/{key}", replays[key], offset, args.output / "calibration" / f"p{i:02d}" / key)
                        for i, offset in enumerate(offsets) for key in calibration], args.workers)

    def choose():
        return max(range(len(offsets)), key=lambda i: calibration_score([reports[f"p{i:02d}/{k}"] for k in calibration], offsets[i]))

    coarse_best = choose()
    if not args.no_refine:
        refinements = offset_grid(offsets[coarse_best], .005)[1:]
        begin = len(offsets)
        offsets.extend(refinements)
        write_json(args.output / "refinement.json", {"coarse_best_index": coarse_best, "offsets_tcp_m": refinements})
        reports.update(evaluate([(f"p{i:02d}/{key}", replays[key], offset, args.output / "calibration" / f"p{i:02d}" / key)
                                 for i, offset in enumerate(offsets) if i >= begin for key in calibration], args.workers))
    best = choose()
    profile = {"schema": "gr3_fdh6_wrist_offset_v1", "side": "right", "contact_offset_tcp_m": offsets[best],
               "baseline_contact_offset_tcp_m": BASE_OFFSET, "correction_tcp_m": offsets[best] - BASE_OFFSET,
               "selected_profile_index": best, "calibration_trials": calibration, "heldout_trials": heldout,
               "simulation_only": True, "hardware_calibrated": False,
               "definition": "p_world_tcp = p_world_source_contact - R_world_tcp @ contact_offset_tcp_m"}
    # Freeze the profile before running any held-out case.
    write_json(args.output / "offset_profile.json", profile)
    validation = evaluate([(f"{label}/{key}", replays[key], offset, args.output / "heldout" / label / key)
                           for label, offset in (("baseline", BASE_OFFSET), ("calibrated", offsets[best]))
                           for key in heldout], args.workers)
    summary = {"profile": profile, "calibration_candidates": [
        {"index": i, "offset_tcp_m": offset, "score": calibration_score([reports[f"p{i:02d}/{k}"] for k in calibration], offset)}
        for i, offset in enumerate(offsets)],
        "heldout": {label: {"successes": sum(validation[f"{label}/{k}"]["grasp_success"] for k in heldout),
                            "trials": len(heldout), "outcomes": {category: sum(validation[f"{label}/{k}"]["failure_category"] == category for k in heldout)
                                for category in sorted({validation[f"{label}/{k}"]["failure_category"] for k in heldout})}}
                    for label in ("baseline", "calibrated")}}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, default=lambda x: x.tolist(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
