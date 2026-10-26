#!/usr/bin/env python3
"""Compare the recorded motor execution and its identical-arm open control."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gr3_grasping.cli import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--planning", default="dev03")
    args = parser.parse_args()
    root = args.root.resolve()
    cases = {k: json.loads((root / k / "execution_report.json").read_text()) for k in ("closed", "open")}
    traces = {k: json.loads((root / k / "contact_trace.json").read_text()) for k in cases}
    provenance = {k: json.loads((root / k / "provenance.json").read_text()) for k in cases}
    fingerprints = [hashlib.sha256((root / k / "plans.json").read_bytes()).hexdigest() for k in cases]
    initial = [np.load(root / k / "planning_state.npz") for k in cases]
    assert fingerprints[0] == fingerprints[1], "Controls must use identical plans"
    assert all(np.array_equal(initial[0][k], initial[1][k]) for k in initial[0].files)
    assert provenance["closed"]["model_sha256"] == provenance["open"]["model_sha256"]
    assert provenance["closed"]["source_sha256"] == provenance["open"]["source_sha256"]
    arm_commands = [np.array([s["arm_target_rad"] for s in traces[k]]) for k in cases]
    assert np.array_equal(*arm_commands), "Arm commands differ between controls"
    comparison = dict(scope=cases["closed"]["scope"], cases=cases,
        identical_plan_sha256=fingerprints[0], identical_initial_state=True,
        identical_arm_commands=True, provenance=provenance["closed"],
        planner=json.loads((root / args.planning / "planner_metadata.json").read_text()),
        tracking_policy="Pregrasp tracking checked after settling; contact phases and hold checked every physics step",
        limitation="One selected development grasp of one known cup; no camera registration or hardware execution")
    write_json(root / "report.json", comparison)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    motion_start = float(initial[0]["time_s"])
    for k, label in (("closed", "FDH-6 close + squeeze"), ("open", "Same arm path, pregrasp fingers")):
        samples = [s for s in traces[k] if s["time_s"] >= motion_start]
        ax.plot([s["time_s"]-motion_start for s in samples], [s["lift_m"]*100 for s in samples], label=label, linewidth=2)
    hold = [s["time_s"]-motion_start for s in traces["closed"] if s["phase"] == "hold"]
    ax.axvspan(hold[0], hold[-1], color="green", alpha=.10, label="2 s hold")
    ax.axhline(10, linestyle="--", color="gray", label="10 cm threshold")
    ax.set(xlabel="Time since arm motion starts (s)", ylabel="Object COM lift (cm)",
        title="GR3 right arm + FDH-6: known cup / oracle registration")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(root / "com_lift.png", dpi=200)
    fig.savefig(root / "com_lift.svg")
    rows = []
    for k, r in cases.items():
        rows.append(f"| {k} | {r['success']} | {r['hold_min_lift_m']*100:.2f} | {r['hold_duration_s']:.2f} | {r['hold_opposed_fraction']:.0%} | {r['max_target_penetration_m']*1000:.3f} |")
    pre, lift = cases["closed"]["stages"]["pregrasp"], cases["closed"]["stages"]["lift"]
    text = "\n".join([
        "# GR3 + FDH-6 Dexonomy 整臂执行", "",
        "已知杯子模型、仿真真值位姿注册；固定基座和腰部，机器人采用理想重力补偿。", "",
        "| 试验 | 通过 | 保持最低 COM 提升 cm | 保持 s | 对向承力比例 | 最大目标穿透 mm |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "两次试验的模型、候选、初始物理状态、规划文件与逐采样手臂目标完全一致；仅闭手指令不同。物体始终为自由体，没有 mocap、物体绑定、合成外力或执行时关节投影。", "",
        "cuRobo 负责全局到预抓取路径及后续笛卡尔采样的 GPU IK；有效 IK 再作有界最小二乘细化。GPU 手型冻结在预成形状态；变化手型由 MuJoCo 重检。挤压阶段先用独立电机物理预演获取实际接触手型，提升规划只在副本中放置随手移动的杯子碰撞几何。正式执行逐物理步检查真实几何和接触。", "",
        f"自由空间预抓取过程最大跟踪误差约 {pre['max_position_error_m']*1000:.1f} mm / {pre['max_rotation_error_rad']:.3f} rad；该阶段在静置 0.5 s 后验收，末端误差约 {pre['endpoint_position_error_m']*1000:.2f} mm / {pre['endpoint_rotation_error_rad']:.4f} rad。接近、闭手、挤压、提升和保持逐步执行 15 mm / 0.10 rad 门限，提升最大平移误差约 {lift['max_position_error_m']*1000:.2f} mm。没有宣称全程均在 15 mm 内。", "",
        "当前为单个开发候选验证，不能换算成十物体成功率，也未接入头部点云定位或真机。", "",
        "- [闭手视频](closed/execution.mp4) / [对照视频](open/execution.mp4)",
        "- [COM 曲线](com_lift.png) / [机器可读报告](report.json)",
        "- 每次目录保存 50 Hz 接触摘要、qpos/qvel/ctrl 状态、终态及导入模型；整臂规划来源目录为 `"+args.planning+"/`。", ""])
    (root / "report.md").write_text(text)
    print(root / "report.md")


if __name__ == "__main__":
    main()
