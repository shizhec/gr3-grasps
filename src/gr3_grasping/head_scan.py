"""Acquire observed surfaces by moving the native head camera in simulation.

Fusion consumes RGB-D frames and their measured kinematic poses. It never
reads object body poses, simulator segmentation, meshes, or hidden surfaces.
The scene must remain static while scanning. Head rotation gives a limited
camera translation baseline; this is not complete object reconstruction.
"""
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import transform_points
from .perception import segment_tabletop

HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")
DEFAULTS = {"enabled": False, "yaw_offsets_rad": [-0.30, 0.0, 0.30],
            "model_context_margin_m": 0.12,
            "joint_margin_rad": 0.001, "max_joint_speed_rad_s": 0.35,
            "settle_s": 0.35, "tracking_tolerance_rad": 0.015,
            "voxel_size_m": 0.002, "association_radius_m": 0.006,
            "min_overlap_fraction": 0.08}


def voxel_fuse(points, rgb, voxel_size):
    """Average observed samples per occupied world-frame voxel."""
    points, rgb = np.asarray(points), np.asarray(rgb)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size_m must be positive")
    if points.ndim != 2 or points.shape[1:] != (3,) or rgb.shape != points.shape:
        raise ValueError("Expected matching Nx3 point and color arrays")
    valid = np.isfinite(points).all(1)
    points, rgb = points[valid], rgb[valid]
    if not len(points):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    _, inverse, count = np.unique(np.floor(points / voxel_size).astype(np.int64),
                                  axis=0, return_inverse=True, return_counts=True)
    fused = np.column_stack([np.bincount(inverse, weights=points[:, i]) / count for i in range(3)])
    color = np.column_stack([np.bincount(inverse, weights=rgb[:, i]) / count for i in range(3)])
    return fused.astype(np.float32), np.rint(color).clip(0, 255).astype(np.uint8)


def observed_scene(frame, scene, perception):
    points = frame.points_world()
    lower, upper = np.array(perception["workspace_min"]), np.array(perception["workspace_max"])
    lower[2] = scene["table_top_m"] - 0.01
    valid = np.isfinite(points).all(-1) & (frame.depth_m > 0)
    valid &= ((points >= lower) & (points <= upper)).all(-1)
    return points[valid], frame.rgb[valid]


def associate_target(clouds, reference_points, config):
    """Match the already observed target using overlap in world coordinates."""
    tree = cKDTree(reference_points)
    best, best_fraction = None, 0.0
    for cloud in clouds:
        distances = tree.query(cloud.points_world)[0]
        fraction = float(np.mean(distances <= config["association_radius_m"]))
        if fraction > best_fraction:
            best, best_fraction = cloud, fraction
    if best_fraction < config["min_overlap_fraction"]:
        return None, best_fraction
    return best, best_fraction


def head_coordinates(sim):
    joints = [sim.model.joint(name) for name in HEAD_JOINTS]
    qids = np.array([j.qposadr[0] for j in joints])
    aids = np.array([sim.model.actuator(f"servo_{name}").id for name in HEAD_JOINTS])
    limits = sim.model.jnt_range[[j.id for j in joints]].copy()
    return qids, aids, limits


def select_scan_poses(sim, target_world, config):
    """Aim at an observed target, respecting official joint limits."""
    qids, _, limits = head_coordinates(sim)
    limits[:, 0] += config["joint_margin_rad"]
    limits[:, 1] -= config["joint_margin_rad"]
    scratch = mujoco.MjData(sim.model)
    scratch.qpos[:] = sim.data.qpos
    cid = sim.model.camera("native_camera").id

    def error(q):
        scratch.qpos[qids] = q
        mujoco.mj_forward(sim.model, scratch)
        rotation = scratch.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        point = (target_world - scratch.cam_xpos[cid]) @ rotation
        if point[2] <= 0:
            return np.array([10.0, 10.0])
        # Keep the target below the image center to retain tabletop context.
        return point[:2] / point[2] - [0, 0.15]

    start = np.clip(sim.data.qpos[qids], limits[:, 0], limits[:, 1])
    result = least_squares(error, start, bounds=(limits[:, 0], limits[:, 1]), max_nfev=80)
    poses = []
    for offset in config["yaw_offsets_rad"]:
        q = result.x.copy()
        q[0] = np.clip(q[0] + offset, *limits[0])
        if not any(np.linalg.norm(q - previous) < 0.01 for previous in [start, *poses]):
            poses.append(q)
    return poses, {"center_command_rad": result.x.tolist(),
                   "target_centering_residual": error(result.x).tolist(),
                   "joint_limits_rad": (limits + np.array([-1, 1]) * config["joint_margin_rad"]).tolist()}


def move_head(sim, goal, config):
    """Execute bounded smooth position commands through the physics servos."""
    qids, aids, limits = head_coordinates(sim)
    goal = np.asarray(goal, dtype=float)
    speed = config["max_joint_speed_rad_s"]
    if speed <= 0 or not np.isfinite(goal).all() or np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
        raise ValueError("Head command exceeds official limits or speed is invalid")
    start = sim.data.qpos[qids].copy()
    # Quintic smoothstep has maximum derivative 1.875.
    duration = max(0.25, 1.875 * float(np.max(np.abs(goal - start))) / speed)
    steps = max(1, int(np.ceil(duration / sim.model.opt.timestep)))
    max_speed = 0.0
    previous = start.copy()
    for index in range(1, steps + 1):
        alpha = index / steps
        smooth = 10 * alpha ** 3 - 15 * alpha ** 4 + 6 * alpha ** 5
        sim.data.ctrl[aids] = start + smooth * (goal - start)
        sim.step()
        actual = sim.data.qpos[qids]
        max_speed = max(max_speed, float(np.max(np.abs(actual - previous))) / sim.model.opt.timestep)
        previous = actual.copy()
        if np.any(actual < limits[:, 0] - 0.002) or np.any(actual > limits[:, 1] + 0.002):
            raise RuntimeError("Head servo exceeded joint limits")
    sim.data.ctrl[aids] = goal
    sim.settle(config["settle_s"])
    error = float(np.max(np.abs(sim.data.qpos[qids] - goal)))
    if error > config["tracking_tolerance_rad"]:
        raise RuntimeError(f"Head tracking error {error:.4f} rad exceeds tolerance")
    return {"command_rad": goal.tolist(), "measured_rad": sim.data.qpos[qids].tolist(),
            "max_measured_speed_rad_s": max_speed, "tracking_error_rad": error}


def write_scan_views(directory, frames, targets, records, summary):
    width, height, title = 480, 360, 45
    sheet = Image.new("RGB", (width * 2, (height + title) * int(np.ceil(len(frames) / 2)) + 45), "#151a20")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, cloud, record) in enumerate(zip(frames, targets, records)):
        x, y = (index % 2) * width, (index // 2) * (height + title)
        panel = Image.fromarray(frame.rgb)
        if cloud is not None:
            overlay = np.asarray(panel).copy()
            overlay[cloud.mask] = (0.6 * overlay[cloud.mask] + [0, 100, 40]).clip(0, 255)
            panel = Image.fromarray(overlay)
        sheet.paste(panel.resize((width, height)), (x, y + title))
        angles = np.rad2deg(record["measured_rad"])
        count = 0 if cloud is None else len(cloud.points_world)
        draw.text((x + 8, y + 7), f"view {index}: yaw {angles[0]:.1f} deg / pitch {angles[1]:.1f} deg", fill="white")
        draw.text((x + 8, y + 24), f"observed target samples: {count}", fill="white")
    draw.text((10, sheet.height - 32),
              f"{summary['settings']['voxel_size_m'] * 1000:g} mm occupied target voxels: {summary['initial_target_voxels']} -> {summary['fused_target_voxels']} | "
              f"camera baseline: {summary['max_camera_baseline_m'] * 100:.1f} cm", fill="white")
    sheet.save(directory / "views.png")


def capture_head_scan(sim, frame, target, config, output):
    """Return a fused scene and target in the original camera reference frame."""
    from .cli import write_json

    settings = {**DEFAULTS, **config.get("head_scan", {})}
    directory = Path(output) / "head_scan"
    directory.mkdir(parents=True, exist_ok=True)
    qids, _, _ = head_coordinates(sim)
    initial_q = sim.data.qpos[qids].copy()
    poses, aiming = select_scan_poses(sim, target.centroid_world, settings)
    frames, targets = [frame], [target]
    records = [{"measured_rad": initial_q.tolist(), "accepted": True, "initial": True}]
    scene_points, scene_rgb = observed_scene(frame, config["scene"], config["perception"])
    scenes, colors = [scene_points], [scene_rgb]
    target_points, target_colors = [target.points_world], [target.rgb]
    try:
        for pose in poses:
            record = move_head(sim, pose, settings)
            current = sim.capture()  # Pose and timestamp sampled after the servo settles.
            clouds = segment_tabletop(current, config["perception"])
            matched, overlap = associate_target(clouds, target.points_world, settings)
            record.update(accepted=matched is not None, overlap_fraction=overlap)
            frames.append(current)
            targets.append(matched)
            records.append(record)
            if matched is not None:
                points, rgb = observed_scene(current, config["scene"], config["perception"])
                scenes.append(points)
                colors.append(rgb)
                target_points.append(matched.points_world)
                target_colors.append(matched.rgb)
    finally:
        restoration = move_head(sim, initial_q, settings)
    voxel = settings["voxel_size_m"]
    initial_points, _ = voxel_fuse(target.points_world, target.rgb, voxel)
    fused_points, fused_rgb = voxel_fuse(np.concatenate(target_points), np.concatenate(target_colors), voxel)
    fused_scene, fused_scene_rgb = voxel_fuse(np.concatenate(scenes), np.concatenate(colors), voxel)
    positions = np.array([f.T_world_camera[:3, 3] for f in frames])
    baseline = float(np.linalg.norm(positions[:, None] - positions[None], axis=-1).max())
    initial_keys = set(map(tuple, np.floor(initial_points / voxel).astype(np.int64)))
    fused_keys = set(map(tuple, np.floor(fused_points / voxel).astype(np.int64)))
    summary = {"settings": settings, "aiming": aiming, "view_count": len(frames),
               "accepted_views": sum(r["accepted"] for r in records),
               "initial_target_samples": len(target.points_world), "initial_target_voxels": len(initial_points),
               "fused_target_voxels": len(fused_points), "new_target_voxels": len(fused_keys - initial_keys),
               "target_voxel_gain_fraction": len(fused_points) / len(initial_points) - 1,
               "max_camera_baseline_m": baseline, "scene_voxels": len(fused_scene),
               "head_restoration": restoration, "observed_surfaces_only": True,
               "coverage_metric": "occupied observed voxels, not complete surface reconstruction",
               "static_scene_required": True, "object_pose_or_mesh_used": False, "views": records}
    for index, (observed, cloud, record) in enumerate(zip(frames, targets, records)):
        observed.save(directory / f"frame_{index:02d}.npz")
        Image.fromarray(observed.rgb).save(directory / f"camera_{index:02d}.png")
        if cloud is not None:
            Image.fromarray((cloud.mask * 255).astype(np.uint8)).save(directory / f"mask_{index:02d}.png")
        record.update(timestamp_s=observed.timestamp_s, T_world_camera=observed.T_world_camera.tolist(),
                      target_samples=0 if cloud is None else len(cloud.points_world))
    np.savez_compressed(directory / "fused_cloud.npz", scene_points_world=fused_scene, scene_rgb=fused_scene_rgb,
                        target_points_world=fused_points, target_rgb=fused_rgb, voxel_size_m=voxel)
    write_scan_views(directory, frames, targets, records, summary)
    write_json(directory / "scan.json", summary)
    fused_target = replace(target, points_world=fused_points, rgb=fused_rgb,
                           points_camera=transform_points(np.linalg.inv(frame.T_world_camera), fused_points),
                           centroid_world=np.median(fused_points, axis=0))
    return fused_scene, fused_target, summary
