#!/usr/bin/env python3
"""Summarize offline synthesis stages separately from physical lift outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from gr3_grasping.dexonomy_backend import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dexonomy_fdh6_v2")
    parser.add_argument("--lift-summary", default="lift_summary_noslip0_dt500us.json")
    args = parser.parse_args()
    root = args.output.resolve()
    summary = json.loads((root / "summary.json").read_text())
    lifts = json.loads((root / args.lift_summary).read_text())
    diagnostics = [r for p in sorted(root.glob("diagnostics_*.json")) for r in json.loads(p.read_text())]
    def phase(record):
        return "pregrasp" if record["path"].endswith(" pregrasp") else "grasp"
    grasp_pass = Counter(Path(r["path"].rsplit(" ", 1)[0]).parent.name
                         for r in diagnostics if r["passed"] and phase(r) == "grasp")
    lifted = Counter(Path(r["source"]).parent.name for r in lifts if r["success"])
    lines = ["# Dexonomy + FDH-6 离线接入实验", "", "日期：2026-10-21。",
        "", "本实验使用完整扫描 mesh 和可控浮动手腕，未使用头部相机或 GR3 机械臂。",
        "每个物体一个初始姿态、三种开发模板，每种保留 16 个初始化；这些模板与该批物体共同用于开发，不是保留测试集。",
        "初始化筛选会补回被拒绝的候选，因此初始化数不表示有效抓姿数。所有生成的完整候选都接受物理验证，允许每物体多个候选；不能与原来的 30 场景单次尝试成功率直接比较。",
        "", "| 物体 | 初始化 | 接触/静力筛选通过 | 含预抓取的完整候选 | 官方动态力闭合通过 | 本地提起通过 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, counts in summary["counts"].items():
        lines.append(f"| {name} | {counts['init_data']} | {grasp_pass[name]} | {counts['grasp_data']} | {counts['succ_grasp']} | {lifted[name]} |")
    lines += ["", "## 最先触发的筛选失败", "", "| 阶段 | 原因 | 数量 |", "|---|---|---:|"]
    failures = Counter((phase(r), r["reason"]) for r in diagnostics if not r["passed"])
    for (stage, reason), count in sorted(failures.items()):
        if stage == "pregrasp" and reason == "hand_scene_collision":
            # Historical diagnostic label: at this stage the threshold is a
            # positive 40 mm clearance, not zero geometric penetration.
            reason = "hand_object_clearance_below_40mm"
        lines.append(f"| {stage} | {reason} | {count} |")
    lines += ["", "## 物理验证", "", f"读取 `{args.lift_summary}`，共 {len(lifts)} 次候选执行，成功 {sum(r['success'] for r in lifts)} 次。",
        "最先触发的终止条件不表示其他条件已通过。", ""]
    for reason, count in Counter(r["failure"] or "passed" for r in lifts).items():
        lines.append(f"- {reason}: {count}")
    lines += ["", "本地验收：物体有重力，手部重力补偿；六个电机力矩上限 ±0.25 Nm，kp=2、kv=0.08；手腕仅由 mocap 焊接控制，物体保持自由体且未绑定到手。",
        "闭手后手腕上移 16 cm，保持 2 s；要求整个保持阶段物体 COM 提升至少 10 cm、拇指及其他手指承载比例至少 90%、不接触桌面，手物/手自身/手桌穿透不超过 3 mm，联动误差不超过 0.01 rad。",
        "实际执行不使用弹簧吸引力、关节投影或物体外力。记录在每个物理步检查，轨迹摘要按 50 Hz 保存。",
        "默认 0.5 ms 步长、noslip_iterations=0；摩擦系数 1.0、扭转摩擦 0.005、condim=4、impratio=10，继承 Dexonomy 的手腕控制与接触求解器。它仍不同于 GR3 全臂执行模型。",
        "官方动态力闭合另用原版默认参数：无重力，依次施加六方向各等于物体重力的外力，每方向最多 2 s，允许位移 5 cm/旋转 15°；它没有执行 10 cm 提起/2 s 保持判定。",
        "", "## 适配与边界", "",
        "原版 Dexonomy 提供模板匹配、MuJoCo 弹簧细化和筛选；本适配在合成细化阶段投影到六主动关节并恢复五个联动角，挤压力矩通过联动雅可比映射到六电机且限幅。此投影仅用于求解，不能作为物理可执行的证明。",
        "导入时显式命名所有物体几何体，保证原版按名称更新接触 margin 生效；视觉网格质量设为零，避免重复计算质量。物体质量来自 benchmark10.yaml，是仿真假设。",
        "`grasp_commands.json` 导出手腕位姿和六个具名电机目标，包含 pregrasp/grasp/squeeze；尚未连接 GR3/cuRobo 执行。",
        "`experiment.json` 固定上游提交、输入哈希、协议与依赖版本；`environment.txt` 保存环境快照。",
        "早期目录 dexonomy_fdh6、dexonomy_fdh6_coupled、dexonomy_fdh6_v1 使用了修复前的适配，不能用于评价性能；v2 下不同积分步长/求解器设置的验证结果分别保存。",
        "", "源码与复现步骤见 `docs/Dexonomy_FDH6接入.md`。", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines))
    print(path)


if __name__ == "__main__":
    main()
