#!/usr/bin/env python3
"""Evaluate a planner on frozen observed candidates and physical start states."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gr3_grasping.cli import write_json
from gr3_grasping.wrist_calibration import BASE_OFFSET, load_offset_profile, run_replay


def execute_trial(args, path):
    name = path.parent.name
    directory = args.output / "trials" / name
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, __file__, "--worker", "--trial", name, "--inputs", str(args.inputs),
               "--output", str(directory), "--planner", args.planner]
    if args.offset_profile:
        command += ["--offset-profile", str(args.offset_profile)]
    if args.video:
        command.append("--video")
    if args.no_close:
        command.append("--no-close")
    try:
        with (directory / "process.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        report = json.loads((directory / "report.json").read_text())
    except Exception as error:
        report = {"grasp_success": False, "grasp_executed": False,
                  "failure_category": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "pipeline_error",
                  "error": str(error)}
        write_json(directory / "report.json", report)
    report["trial"] = name
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/wrist_calibration/inputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner", choices=["scipy", "curobo"], default="curobo")
    parser.add_argument("--offset-profile", type=Path)
    parser.add_argument("--trial", help="Run one trial, e.g. android_toy_yaw000")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video", action="store_true", help="Render a single trial")
    parser.add_argument("--no-close", action="store_true", help="Fingers-open negative control")
    parser.add_argument("--workers", type=int, default=1, help="Independent worker processes; default one GPU planner at a time")
    args = parser.parse_args()
    offset = load_offset_profile(args.offset_profile) if args.offset_profile else BASE_OFFSET.tolist()
    sources = sorted(args.inputs.glob("*/replay.json"))
    if args.trial:
        sources = [p for p in sources if p.parent.name == args.trial]
    if not sources:
        raise ValueError("No matching replay inputs")
    if args.worker:
        if len(sources) != 1:
            raise ValueError("Worker needs exactly one trial")
        replay = json.loads(sources[0].read_text())
        run_replay(replay, offset, args.output, args.planner, render=args.video, no_close=args.no_close)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_trial, args, path) for path in sources]
        for future in as_completed(futures):
            report = future.result()
            rows.append(report)
            summary = {"planner": args.planner, "contact_offset_tcp_m": offset, "completed_trials": len(rows),
                       "expected_trials": len(sources), "successes": sum(r["grasp_success"] for r in rows),
                       "executed_trials": sum(r["grasp_executed"] for r in rows),
                       "outcomes": {category: sum(r["failure_category"] == category for r in rows)
                                    for category in sorted({r["failure_category"] for r in rows})}, "trials": rows}
            write_json(args.output / "summary.json", summary)
            print(report["trial"], report["failure_category"], f"{len(rows)}/{len(sources)}", flush=True)


if __name__ == "__main__":
    main()
