#!/usr/bin/env python3
"""Independent simulation trials of saved candidates; never commands hardware.

Each trial starts from the same scene. This is offline validation, not a real
robot retry controller. It never edits object poses or inserts attachments.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from gr3_grasping.cli import write_json
import numpy as np

from gr3_grasping.grasps import Grasp
from gr3_grasping.grasping import run_grasp
from gr3_grasping.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing run containing model/scene.xml, grasps.json and report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--stop-on-success", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.run / "run_config.json").read_text())
    original_report = json.loads((args.run / "report.json").read_text())
    items = json.loads((args.run / "grasps.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"independent_simulation_trials": True, "source_run": str(args.run.resolve()), "trials": []}
    for index, item in enumerate(items[:args.limit]):
        directory = args.output / f"candidate_{index:03d}"
        directory.mkdir(exist_ok=True)
        pose = np.asarray(item["T_world_tcp"])
        direction = item.get("approach_direction_world")
        grasp = Grasp(pose, item["score"], item["source"], None if direction is None else np.asarray(direction))
        report = {key: original_report[key] for key in ("arm", "hand", "selected_centroid_world_m")}
        report.update(candidate_index=index, grasp_success=False, grasp_executed=False)
        try:
            with Simulation(args.run / "model/scene.xml", config, render=False) as sim:
                sim.settle()
                report = run_grasp(sim, [grasp], report, SimpleNamespace(plan_only=False, no_close=False), directory, write_json)
        except RuntimeError as error:
            report.update(stage="failed", error=str(error))
        write_json(directory / "report.json", report)
        result = {key: report[key] for key in ("grasp_success", "grasp_executed", "final_lift_m", "hold_opposed_contact_fraction", "error") if key in report}
        result["candidate_index"] = index
        summary["trials"].append(result)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(result), flush=True)
        if args.stop_on_success and report["grasp_success"]:
            break
    return 0 if any(row["grasp_success"] for row in summary["trials"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
