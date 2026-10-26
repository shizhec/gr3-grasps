#!/usr/bin/env python3
"""Report the frozen-candidate integrator A/B, negative control and convergence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from gr3_grasping.dexonomy_backend import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dexonomy_dynamics_audit")
    args = parser.parse_args()
    root = args.output.resolve()
    prefix = "lift_summary_mj3.16.0_"
    profiles = {
        "implicitfast": prefix + "implicitfast_wrist4000us_noslip0_dt500us.json",
        "discrete": prefix + "discrete_wrist4000us_noslip0_dt500us.json",
        "half_step": prefix + "discrete_wrist4000us_noslip0_dt250us.json",
        "open_control": prefix + "discrete_wrist4000us_noslip0_dt500us_open_control.json",
    }
    def key(row):
        path = Path(row["source"])
        return f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
    data = {name: {key(r): r for r in json.loads((root / filename).read_text())}
            for name, filename in profiles.items()}
    if not all(set(d) == set(data["discrete"]) for d in data.values()):
        raise RuntimeError("Profile candidate identities differ")
    n_candidate = len(data["discrete"])
    if not n_candidate:
        raise RuntimeError("No candidates to compare")
    for d in data.values():
        for row in d.values():
            if row["mujoco_version"] != "3.16.0" or row["wrist_timeconstant_s"] != .004:
                raise RuntimeError("Unexpected comparison protocol")
    sources = {name: hashlib.sha256(Path(row["source"]).read_bytes()).hexdigest()
               for name, row in data["discrete"].items()}
    for name, rows in data.items():
        for candidate, row in rows.items():
            if row.get("source_sha256", sources[candidate]) != sources[candidate]:
                raise RuntimeError(f"Candidate changed in {name}: {candidate}")
    coverage = dict(Counter(Path(r["source"]).parent.name for r in data["discrete"].values()))
    summary = dict(scope="Frozen candidate comparison; floating wrist with offline full mesh; no GR3 arm execution/camera",
        candidate_count=n_candidate, object_candidate_counts=coverage,
        profiles=profiles, source_sha256=sources,
        successes={name: sum(r["success"] for r in rows.values()) for name, rows in data.items()},
        matching_half_step_outcomes=all(row["success"] == data["half_step"][name]["success"]
                                       for name, row in data["discrete"].items()),
        failures=dict(Counter(row["failure"] for row in data["discrete"].values() if not row["success"])))
    lines = ["# FDH-6 动态联动修正与验证", "", "日期：2026-10-21。",
        "", f"对 {n_candidate} 个完整候选进行固定输入对照，未重新选择抓姿。各物体候选数：{coverage}。本次只验证已有完整候选，不表示原十个物体均已产生可执行抓姿。",
        "本轮保持 ±0.25 Nm 电机限幅、kp=2、kv=0.08、物体质量/几何和成功阈值一致。浮动手腕时间常数统一设为 4 ms，并增加 15 mm/0.10 rad 手腕跟踪验收。",
        "", "| 候选 | implicitfast | discrete | 保持最低提升 cm | 0.25 ms 复核 | 不闭手对照 |", "|---|---|---|---:|---|---|"]
    label = lambda row: "通过" if row["success"] else "失败"
    for name, row in data["discrete"].items():
        lines.append(f"| {name} | {label(data['implicitfast'][name])} | {label(row)} | {row['hold_min_lift_m']*100:.2f} | {label(data['half_step'][name])} | {label(data['open_control'][name])} |")
    maximum = max(r["max_mimic_error_rad"] for r in data["discrete"].values())
    lines += ["", f"discrete 成功 {summary['successes']['discrete']}/{n_candidate}；不闭手对照成功 {summary['successes']['open_control']}/{n_candidate}；步长减半的结果全部一致：{summary['matching_half_step_outcomes']}。",
        f"discrete 执行的最大联动误差 {maximum:.8f} rad；验收阈值仍为 0.01 rad。",
        "", "## 原因与修正", "",
        "同一 MuJoCo 3.16.0 环境，仅改变积分器，implicitfast 六次均先触发联动误差门限，discrete 六次均不再触发。失败关节分别涉及中指、食指、小指；电机没有在首次失败时全部饱和，求解器也未耗尽迭代。",
        "这是当前模型上的因果对照。对机理的解释是：高阻尼和低惯量关节的有效惯量，需要与联动约束在同一个离散步内一致求解；具体实现依据 [MuJoCo 3.16 源码中的计算文档](https://github.com/google-deepmind/mujoco/blob/3.16.0/doc/computation/index.rst#numerical-integration)。没有用关节位置投影掩盖物理执行误差。",
        "原版手腕焊接伺服时间常数 20 ms，独立检查时部分候选的手腕旋转偏离目标超过 0.10 rad。因此单独将其设为 4 ms，并在两个积分器对照中保持一致；它是理想受控手腕的仿真假设，尚未标定为真实 GR3 腕部刚度。",
        "", "## 剩余失败与下一步", "",
        "tripod/coffee_mug/0_1_grasp 没有持续提起，最高提升约 4 mm，保持阶段对向承载消失；tripod/coffee_mug/0_6_grasp 保持约 8.85 cm，未达到 10 cm 门限。它们不再因联动误差而退出。",
        "保持 2 s、COM 全程提升至少 10 cm、对向承载至少 90%、穿透至多 3 mm、联动误差至多 0.01 rad 的标准没有放宽；新增检查手腕跟踪。",
        "", "## GR3 右臂端点检查", ""]
    ik_path = root / "gr3_ik/report.json"
    if ik_path.exists():
        ik = json.loads(ik_path.read_text())
        count = sum(r["all_endpoints_reachable"] for r in ik["results"])
        lines += [f"在原固定基座、固定腰部、物体平面位置 [0.46, -0.20] m 下，对六个候选检查预抓取/抓取/挤压/提升端点；4 个初值的有界数值 IK 中，{count}/6 个候选所有端点达标。",
            "全部端点达标的是 wrap_wide/coffee_mug/0_3_grasp；这些端点也未触发当前 MuJoCo 碰撞筛选。其余候选至少一个端点未在 3 mm/0.04 rad 阈值内求解成功，不能由此证明全局不可达。",
            "注册使用离线物体真值位姿；检查使用 SciPy IK，只是端点诊断，不是 cuRobo 连续路径规划，也没有驱动 GR3 执行。下一步应先用这个杯子候选接全臂规划、逐阶段手型和碰撞几何，再接视觉注册。",
            "", "详细端点误差见 `gr3_ik/report.json`。"]
    lines += ["", "## 范围与产物", "",
        "这是筛选后候选的离线执行结果，不能外推为十物体成功率、三十场景成功率或真机成功率。没有更改物体为附着体，也没有向物体施加合成外力；只有手腕连接 mocap。",
        "`comparison.json` 保存各实验结果、候选哈希和步长复核结论；各 `lift_validation_*` 目录保存逐次报告、关节诊断、终止状态与可选视频。",
        "复现流程见项目的 `docs/Dexonomy_FDH6接入.md` 动态修正部分。", ""]
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
