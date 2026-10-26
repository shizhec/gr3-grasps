#!/usr/bin/env python3
"""Run isolated multi-object grasp trials and keep every failure in the rate."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gr3_grasping.benchmark import trial_config, run_trial, write_results, write_contact_sheet, failure_category
from gr3_grasping.cli import write_json
from gr3_grasping.model import ROOT
import yaml


def run_isolated(directory, timeout_s):
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("LP_NUM_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")
    with (directory / "process.log").open("w") as log:
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, env=env)
            if completed.returncode and not (directory / "report.json").exists():
                raise RuntimeError(f"Worker exited {completed.returncode}; see process.log")
            report = json.loads((directory / "report.json").read_text())
            if "failure_category" not in report:
                report.update(stage="failed", grasp_success=False,
                              error=f"Worker exited {completed.returncode} without a final trial report; see process.log")
                report["failure_category"] = failure_category(report)
                write_json(directory / "report.json", report)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError, FileNotFoundError) as error:
            report = {"grasp_success": False, "grasp_executed": False, "stage": "failed", "error": str(error)}
            if isinstance(error, subprocess.TimeoutExpired):
                previous = directory / "report.json"
                if previous.is_file():
                    report = {**json.loads(previous.read_text()), **report}
                    report["grasp_executed"] = json.loads(previous.read_text()).get("grasp_executed", False)
                report["error"] = f"Trial timed out after {timeout_s} seconds"
            report["failure_category"] = failure_category(report)
            write_json(directory / "report.json", report)
    spec = json.loads((directory / "trial_spec.json").read_text())
    metadata_path = directory / "m2t2_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    scan_path = directory / "head_scan/scan.json"
    scan = json.loads(scan_path.read_text()) if scan_path.is_file() else {}
    return {**report, **spec, "raw_grasps": metadata.get("raw_grasps"),
            "head_scan_views": scan.get("accepted_views", 1),
            "target_voxel_gain_fraction": scan.get("target_voxel_gain_fraction"),
            "max_camera_baseline_m": scan.get("max_camera_baseline_m"),
            "compatibility_fixes": metadata.get("compatibility_fixes", []),
            "inference_s": metadata.get("inference_s"), "wall_time_s": time.perf_counter() - started,
            "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/benchmark10.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark10")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--planner", choices=["scipy", "curobo"])
    parser.add_argument("--fdh6-offset-profile", type=Path)
    parser.add_argument("--head-scan", action="store_true", help="Acquire and fuse native head-camera views before inference")
    parser.add_argument("--head-context-margin-m", type=float, default=0.12,
                        help="XY context around observed scan target; 0 reproduces uncropped fusion")
    parser.add_argument("--legacy-m2t2-filter", action="store_true", help="Reproduce the original upstream object-mask filtering bug")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        spec = json.loads((args.worker / "trial_spec.json").read_text())
        config = json.loads((args.worker / "run_config.json").read_text())
        report = run_trial(config, args.worker, spec["arm"])
        return 0 if report["grasp_success"] else 1
    if args.jobs < 1:
        parser.error("jobs must be positive")
    manifest = yaml.safe_load(args.manifest.read_text())
    if args.planner:
        manifest.setdefault("planning", {})["backend"] = args.planner
    if args.fdh6_offset_profile:
        from gr3_grasping.wrist_calibration import load_offset_profile, file_sha
        if manifest["arm"] != "right":
            parser.error("The current FDH-6 offset profile applies to the right hand")
        manifest["fdh6_contact_offset_m"] = load_offset_profile(args.fdh6_offset_profile)
        manifest["fdh6_offset_profile_sha256"] = file_sha(args.fdh6_offset_profile)
    if args.head_scan:
        manifest.setdefault("head_scan", {})["enabled"] = True
        manifest["head_scan"]["model_context_margin_m"] = args.head_context_margin_m or None
    manifest["fix_object_mask_filter"] = not args.legacy_m2t2_filter
    if len({item["id"] for item in manifest["objects"]}) != len(manifest["objects"]):
        parser.error("Object IDs must be unique")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("Output already contains a benchmark; use a new output directory to preserve prior results")
    protocol = {"manifest": manifest, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "backend": "m2t2", "physical_attempts_per_trial": 1, "single_object_scenes": True,
                "target_selection": "largest observed RGB-D foreground component within 0.14 m of the fixed task XY region",
                "scale": 1.0,
                "mass_source": "declared simulation assumptions", "jobs": args.jobs}
    protocol["height_reference"] = manifest.get("evaluation_height_reference", "center_of_mass")
    protocol["effective_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write_json(output / "manifest.json", protocol)
    implementation_paths = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").glob("*.py")),
                            *sorted((ROOT / "configs").glob("*.yaml")),
                            ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "assets.lock.json"]
    write_json(output / "implementation_sha256.json", {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths})
    trials = []
    for item in manifest["objects"]:
        for yaw in manifest["yaws_deg"]:
            directory = output / "trials" / f"{item['id']}_yaw{yaw:03d}"
            directory.mkdir(parents=True)
            config = trial_config(manifest, item, yaw)
            write_json(directory / "run_config.json", config)
            (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            write_json(directory / "trial_spec.json", {"object_id": item["id"], "label": item["label"],
                                                      "yaw_deg": yaw, "arm": manifest["arm"]})
            trials.append(directory)
    rows = []
    write_results(output, rows, len(trials))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_isolated, directory, manifest["timeout_s"]): directory for directory in trials}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: trials.index(Path(r["directory"])))
            summary = write_results(output, rows, len(trials))
            print(json.dumps({"completed": len(rows), "total": len(trials), "object": row["object_id"],
                              "yaw": row["yaw_deg"], "result": row["failure_category"],
                              "successes": summary["successes"]}, ensure_ascii=False), flush=True)
    write_contact_sheet(output, rows)
    print(f"Complete. Report: {output / 'report.md'}", flush=True)
    # Grasp failures are measured results, not a failed benchmark run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
