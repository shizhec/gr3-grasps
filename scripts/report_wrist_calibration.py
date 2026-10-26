#!/usr/bin/env python3
"""Assemble calibration, held-out, and planner comparisons without dropping failures."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("outputs/wrist_calibration"))
    parser.add_argument("--curobo-baseline", type=Path, default=Path("outputs/curobo_baseline"))
    parser.add_argument("--curobo-calibrated", type=Path, default=Path("outputs/curobo_calibrated"))
    args = parser.parse_args()
    root = args.calibration
    summary = read(root / "summary.json")
    best = summary["profile"]["selected_profile_index"]
    groups = {}
    for label, profile in (("scipy_original", 0), ("scipy_calibrated", best)):
        rows = {p.parent.name: read(p) for p in (root / "calibration" / f"p{profile:02d}").glob("*/report.json")}
        variant = "baseline" if profile == 0 else "calibrated"
        rows.update({p.parent.name: read(p) for p in (root / "heldout" / variant).glob("*/report.json")})
        groups[label] = rows
    for label, path in (("curobo_original", args.curobo_baseline), ("curobo_calibrated", args.curobo_calibrated)):
        if (path / "summary.json").is_file():
            batch = read(path / "summary.json")
            if batch["completed_trials"] != batch["expected_trials"]:
                raise ValueError(f"Incomplete benchmark: {path}")
            groups[label] = {row["trial"]: row for row in batch["trials"]}
    aggregated = {}
    for label, rows in groups.items():
        if len(rows) != 30:
            raise ValueError(f"Expected all thirty trials: {label}")
        train = [r for name, r in rows.items() if name.endswith("yaw000")]
        heldout = [r for name, r in rows.items() if not name.endswith("yaw000")]
        aggregated[label] = {"successes": sum(r["grasp_success"] for r in rows.values()), "trials": len(rows),
                             "calibration_successes": sum(r["grasp_success"] for r in train), "calibration_trials": len(train),
                             "heldout_successes": sum(r["grasp_success"] for r in heldout), "heldout_trials": len(heldout),
                             "executed_trials": sum(r["grasp_executed"] for r in rows.values()),
                             "outcomes": dict(Counter(r["failure_category"] for r in rows.values()))}
    original, calibrated = groups["scipy_original"], groups["scipy_calibrated"]
    identities = all(original[k]["input_sha256"] == calibrated[k]["input_sha256"] and
                     original[k]["initial_state_sha256"] == calibrated[k]["initial_state_sha256"] for k in original)
    changed = sum(original[k].get("selected_candidate_index") != calibrated[k].get("selected_candidate_index") for k in original)
    previous = {p.parent.name: read(p)["grasp_success"] for p in Path("outputs/benchmark10_head_scan_context/trials").glob("*/report.json")}
    mismatches = [k for k in original if k in previous and original[k]["grasp_success"] != previous[k]]
    result = {"groups": aggregated, "identical_inputs_and_start_states": identities,
              "selected_candidate_changed_trials": changed, "historical_replay_outcome_mismatches": mismatches,
              "profile": summary["profile"], "unseen_objects": False}
    (root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# cuRobo 与 FDH-6 手腕偏移实验", "", "同一批头部点云与 M2T2 候选，固定起始积分状态；每场景只执行一次。",
             "10 个 0° 场景用于偏移选择；45°、90° 的 20 个场景在参数冻结后测试。物体类别在两部分重合。", "",
             f"原接触偏移：{summary['profile']['baseline_contact_offset_tcp_m']} m。",
             f"选定接触偏移：{summary['profile']['contact_offset_tcp_m']} m。", "",
             "| 组别 | 校准场景成功 / 10 | 未参与调参的朝向成功 / 20 | 总成功 / 30 | 执行次数 | 规划失败 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, row in aggregated.items():
        lines.append(f"| {label} | {row['calibration_successes']} | {row['heldout_successes']} | {row['successes']} | {row['executed_trials']} | {row['outcomes'].get('planning_failure', 0)} |")
    lines += ["", f"SciPy 偏移前后输入及起始状态完全相同：{identities}。实际选中候选编号变化：{changed}/30。",
              f"原偏移重放与历史成功/失败结论差异：{mismatches}。", "",
              "成功仍要求质心升高至少 10 cm 并保持 2 s、无桌面接触、对向承重接触占比至少 90%、穿透不超过 3 mm。",
              "总成功次数中含用于调参的 10 个场景；泛化效果应单独看另外 20 个朝向，不能视为未见物体或真机成功率。", "",
              "完整数据见 comparison.json、summary.json、protocol.json、refinement.json、逐次 planning_diagnostics.json 与 contact_trace.json。", ""]
    (root / "report.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
