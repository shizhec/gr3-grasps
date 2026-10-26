"""Fixed-protocol multi-object simulation evaluation with complete denominators."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .cli import perceive, write_json  # Configure headless rendering first.
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from .grasping import run_grasp
from .model import ROOT, build_scene
from .perception import segment_tabletop
from .simulation import Simulation

OUTCOME_LABELS = {
    "success": "成功", "no_neural_grasp": "无有效模型候选",
    "planning_failure": "路径不可行", "insufficient_lift_or_drop": "未稳定提起/滑落",
    "execution_collision": "执行碰撞", "tracking_failure": "跟踪失败",
    "perception_failure": "目标分割失败", "timeout": "超时",
    "unstable_contact_or_table_contact": "接触不稳定/未离桌", "pipeline_error": "流程错误",
}

def trial_config(manifest, item, yaw):
    config = yaml.safe_load((ROOT / manifest["base_config"]).read_text())
    config = deepcopy(config)
    config["scene"]["objects"] = [{
        "name": "benchmark_target", "type": "mjcf",
        "mjcf": f"third_party/mujoco_scanned_objects/models/{item['folder']}/model.xml",
        "placement_xy": manifest["placement_xy"], "rotation_rpy_deg": [0, 0, yaw],
        "mass": item["mass_kg"],
    }]
    config["camera"]["seed"] = manifest["seed"]
    config["m2t2"]["seed"] = manifest["seed"]
    config["perception"]["workspace_min"] = manifest["workspace_min"]
    config["perception"]["workspace_max"] = manifest["workspace_max"]
    config["hand"]["evaluation_height_reference"] = manifest.get("evaluation_height_reference", "center_of_mass")
    if "head_scan" in manifest:
        config["head_scan"] = deepcopy(manifest["head_scan"])
    if "fix_object_mask_filter" in manifest:
        config["m2t2"]["fix_object_mask_filter"] = manifest["fix_object_mask_filter"]
    if "planning" in manifest:
        config["planning"].update(deepcopy(manifest["planning"]))
    if "fdh6_contact_offset_m" in manifest:
        config["m2t2"]["fdh6_contact_offset_m"] = list(manifest["fdh6_contact_offset_m"])
    return config


def failure_category(report):
    if report.get("grasp_success"):
        return "success"
    error = report.get("error", "").lower()
    if "timed out" in error:
        return "timeout"
    if "collision during" in error or "penetration:" in error:
        return "execution_collision"
    if "tracking failed" in error:
        return "tracking_failure"
    if "no complete fdh-6 grasp path" in error:
        return "planning_failure"
    if "m2t2 produced no" in error:
        return "no_neural_grasp"
    if "foreground" in error or "no rgb-d" in error:
        return "perception_failure"
    if report.get("grasp_executed") and report.get("hold_min_lift_m") is not None:
        if report.get("hold_min_lift_m", 0) < report.get("success_lift_threshold_m", .1):
            return "insufficient_lift_or_drop"
        return "unstable_contact_or_table_contact"
    return "pipeline_error"


def select_target_index(clouds, placement_xy, radius_m=0.14):
    """Use the fixed task region to reject the robot's own foreground.

    This task prior is shared by all objects and never reads simulator poses.
    """
    eligible = [i for i, cloud in enumerate(clouds)
                if np.linalg.norm(cloud.centroid_world[:2] - np.asarray(placement_xy)) <= radius_m]
    if not eligible:
        raise RuntimeError("No RGB-D foreground object in the fixed target region")
    return max(eligible, key=lambda i: len(clouds[i].points_world))


def run_trial(config, output, arm="right"):
    """Run one physical attempt, saving observations even when planning fails."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", config)
    report = {"stage": "started", "grasp_success": False, "grasp_executed": False,
              "arm": arm, "hand": "fdh6", "fixed_base": True}
    write_json(output / "report.json", report)
    started = time.perf_counter()
    try:
        scene = build_scene(config, output / "model")
        with Simulation(scene, config) as sim:
            target_body = sim.model.body("benchmark_target").id
            try:
                sim.settle()
                frame = sim.capture()
                frame.save(output / "frame.npz")
                Image.fromarray(frame.rgb).save(output / "camera.png")
                Image.fromarray(sim.grasp_view(target_body, "initial")).save(output / "initial_grasp.png")
                clouds = segment_tabletop(frame, config["perception"])
                if not clouds:
                    raise RuntimeError("No RGB-D foreground objects in workspace")
                # One object is present. Select the largest observed foreground
                # component; neither segmentation nor proposals use body IDs.
                index = select_target_index(clouds, config["scene"]["objects"][0]["placement_xy"])
                args = SimpleNamespace(target_text=None, detector_device="cpu", object_index=index,
                                       backend="m2t2", arm=arm, candidate_index=None,
                                       plan_only=False, no_close=False)
                proposals, perception = perceive(frame, config, args, output, sim=sim)
                report.update(perception, stage="perception_complete")
                write_json(output / "report.json", report)
                source = json.loads((output / "m2t2_source_grasps.json").read_text())
                Image.fromarray(sim.grasp_candidates_view([item["T_world_tcp"] for item in source])).save(output / "m2t2_grasps.png")
                report = run_grasp(sim, proposals, report, args, output, write_json)
            finally:
                # Only visualization/evaluation use the known target body.
                Image.fromarray(sim.grasp_view(target_body, report["stage"])).save(output / "final_grasp.png")
                Image.fromarray(sim.capture().rgb).save(output / "final_camera.png")
                np.savez_compressed(output / "final_state.npz", qpos=sim.data.qpos,
                                    qvel=sim.data.qvel, ctrl=sim.data.ctrl, time_s=sim.data.time)
    except Exception as error:
        report.update(failed_stage=report["stage"], stage="failed", error=str(error), grasp_success=False)
    report.update(wall_time_s=time.perf_counter() - started)
    report["failure_category"] = failure_category(report)
    write_json(output / "report.json", report)
    return report


def summarize(rows, expected_trials):
    """Every trial, including perception/planning/timeouts, remains in the rate."""
    successes = sum(bool(row["grasp_success"]) for row in rows)
    objects = []
    for object_id in dict.fromkeys(row["object_id"] for row in rows):
        trials = [row for row in rows if row["object_id"] == object_id]
        objects.append({"object_id": object_id, "label": trials[0]["label"], "trials": len(trials),
                        "successes": sum(bool(row["grasp_success"]) for row in trials),
                        "by_yaw": {str(row["yaw_deg"]): row["failure_category"] for row in trials}})
    failures = {}
    for row in rows:
        failures[row["failure_category"]] = failures.get(row["failure_category"], 0) + 1
    return {"completed_trials": len(rows), "expected_trials": expected_trials,
            "complete": len(rows) == expected_trials, "successes": successes,
            "success_rate": successes / expected_trials if expected_trials else 0,
            "success_rate_denominator": "all configured trials, including planning/perception failures and timeouts",
            "executed_trials": sum(bool(row.get("grasp_executed")) for row in rows),
            "objects": objects, "outcomes": failures, "trials": rows}


def write_results(output, rows, expected_trials):
    output = Path(output)
    summary = summarize(rows, expected_trials)
    write_json(output / "summary.json", summary)
    columns = ["object_id", "label", "yaw_deg", "grasp_success", "grasp_executed", "failure_category",
               "hold_min_lift_m", "final_lift_m", "hold_opposed_contact_fraction",
               "max_target_finger_penetration_m", "selected_candidate_index", "raw_grasps",
               "inference_s", "wall_time_s", "object_height_reference", "error", "directory"]
    columns[columns.index("error"):columns.index("error")] = [
        "head_scan_views", "target_voxel_gain_fraction", "max_camera_baseline_m"]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = ["# 10 物体抓取批量评估", "", "每次桌面放一个物体；0°、45°、90° 朝向各一次。",
              "每次只执行 M2T2 候选中首个通过全程规划的位姿，不进行执行失败后的重试。",
              "升高量按物体质心计算，避免扫描模型原点偏移和旋转带来的误判。",
              "失败、无候选、不可达、碰撞和超时均计入总分母。", "",
              f"完成 {len(rows)}/{expected_trials} 次，成功 {summary['successes']}/{expected_trials}（{summary['success_rate']:.1%}）。", "",
              "分类统计：" + "；".join(f"{OUTCOME_LABELS.get(key, key)} {count} 次" for key, count in summary["outcomes"].items()) + "。", "",
              "| 物体 | 0° | 45° | 90° | 成功次数 |", "|---|---|---|---|---|"]
    for obj in summary["objects"]:
        cells = [OUTCOME_LABELS.get(obj["by_yaw"].get(str(yaw)), "待运行") for yaw in (0, 45, 90)]
        report.append(f"| {obj['label']} | {' | '.join(cells)} | {obj['successes']}/{obj['trials']} |")
    protocol_path = output / "manifest.json"
    if protocol_path.is_file():
        import json
        manifest = json.loads(protocol_path.read_text())["manifest"]
        report.extend(["", "头部多视角扫描：" + ("启用，逐帧外参融合后推理。" if manifest.get("head_scan", {}).get("enabled") else "关闭，单视角输入。"),
                       "M2T2 目标掩码筛选括号修复：" + ("启用。" if manifest.get("fix_object_mask_filter", True) else "关闭，保留上游旧行为作对照。")])
    report.extend(["", "原始结果见 results.csv、summary.json；逐次点云、模型输出、规划诊断和接触轨迹保存在 trials/。",
                   "质量、摩擦、手部偏移和相机内参仍是仿真假设；单场景固定基座结果不代表真机或堆叠场景成功率。", ""])
    (output / "report.md").write_text("\n".join(report))
    return summary


def write_contact_sheet(output, rows):
    """Show all thirty outcomes in fixed yaw columns, including every failure."""
    output = Path(output)
    if not rows:
        return
    objects = list(dict.fromkeys(row["object_id"] for row in rows))
    canvas = Image.new("RGB", (1100, 100 + len(objects) * 256), "#17202b")
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font = ImageFont.truetype(font_path, 17) if Path(font_path).is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    successes = sum(row["grasp_success"] for row in rows)
    draw.text((12, 10), f"GR3 / FDH-6 / M2T2 | {successes}/{len(rows)} 成功 | 单物体 / 质心高度 / 保持 2 秒", fill="white", font=font)
    for col, yaw in enumerate((0, 45, 90)):
        draw.text((180 + 302 * col, 55), f"初始朝向 {yaw}°", fill="white", font=font)
    for index, object_id in enumerate(objects):
        trials = [row for row in rows if row["object_id"] == object_id]
        y = 100 + index * 256
        draw.text((12, y + 20), trials[0]["label"], fill="white", font=font)
        draw.text((12, y + 49), f"{sum(r['grasp_success'] for r in trials)}/{len(trials)} 成功", fill="white", font=font)
        for col, yaw in enumerate((0, 45, 90)):
            row = next((r for r in trials if r["yaw_deg"] == yaw), None)
            if row is None:
                continue
            x = 180 + 302 * col
            color = "#6ee7a5" if row["grasp_success"] else "#ff9b8e"
            path = Path(row["directory"]) / "final_grasp.png"
            if path.is_file():
                with Image.open(path) as panel:
                    closeup = panel.crop((panel.width // 2, 25, panel.width, panel.height))
                    canvas.paste(closeup.resize((294, 209)), (x, y + 27))
            draw.text((x + 3, y), OUTCOME_LABELS.get(row["failure_category"], row["failure_category"]), fill=color, font=font)
            draw.rectangle((x, y + 25, x + 295, y + 237), outline=color, width=2)
    canvas.save(output / "contact_sheet.png")
    passed = [row for row in rows if row["grasp_success"]]
    if passed:
        panel = Image.new("RGB", (640 * len(passed), 525), "#17202b")
        caption = ImageDraw.Draw(panel)
        for index, row in enumerate(passed):
            with Image.open(Path(row["directory"]) / "final_grasp.png") as source:
                panel.paste(source.crop((source.width // 2, 0, source.width, source.height)).resize((640, 480)), (640 * index, 45))
            caption.text((640 * index + 10, 10),
                         f"{row['label']} | {row['yaw_deg']}° | 质心升高 ≥ {row['hold_min_lift_m'] * 100:.2f} cm | 保持 2 s",
                         fill="#6ee7a5", font=font)
        panel.save(output / "successful_cases.png")
