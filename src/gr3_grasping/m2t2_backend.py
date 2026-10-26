"""Run the official trained M2T2 network on camera-derived scene points."""
import hashlib
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from scipy.spatial import cKDTree

from .geometry import transform, validate_transform
from .grasps import Grasp
from .model import ROOT

WEIGHT_REPO = "wentao-yuan/m2t2"
WEIGHT_REVISION = "a20325dcd19cd1b838274974d0ebe35bfe383796"
WEIGHT_SHA256 = "e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb"


def crop_target_context(points, target_points, margin):
    """Keep observed XY target bounds plus support/obstacle context in meters."""
    if margin is None:
        return points
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Target context margin must be positive or null")
    target_xy = np.asarray(target_points)[:, :2]
    lower, upper = target_xy.min(0) - margin, target_xy.max(0) + margin
    return points[((points[:, :2] >= lower) & (points[:, :2] <= upper)).all(1)]


def base_to_tcp(base_pose):
    """M2T2 action_decoder.py places the contact center 0.1034 m along +Z."""
    return validate_transform(np.asarray(base_pose, dtype=float) @ transform(translation=[0, 0, 0.1034]))


def retarget_fdh6(proposals, side, contact_offset=(-0.012, 0, -0.045)):
    """Transfer learned approach/closing axes to the FDH-6 hand frame.

    This is a rigid-frame candidate transfer, not a trained dexterous grasp
    policy. Collision checking and actual contact/lift evaluation remain required.
    """
    result = []
    if side not in ("right", "left"):
        raise ValueError("FDH-6 side must be right or left")
    # Panda: Z approaches, X closes. The existing FDH-6 wrist site has X
    # through the palm opening and Z along the thumb/finger opposition.
    # Its contact region is offset from that site; this provisional simulation
    # estimate must be measured again on the physical hand.
    mapping = (np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) if side == "right" else
               np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    for grasp in proposals:
        for flip in (False, True):
            pose = grasp.T_world_tcp.copy()
            pose[:3, :3] = pose[:3, :3] @ mapping
            if flip:
                pose[:3, :3] = pose[:3, :3] @ np.diag([1, -1, -1])
            pose[:3, 3] -= pose[:3, :3] @ np.asarray(contact_offset)
            direction = pose[:3, 0] * (1 if side == "right" else -1)
            result.append(Grasp(pose, grasp.score,
                                grasp.source + ":fdh6_frame_transfer" + (":closing_symmetry" if flip else ""), direction))
    return result


class M2T2Backend:
    def __init__(self, config):
        import torch
        from omegaconf import OmegaConf
        from .pointnet_cpu import install

        self.config = config
        source = ROOT / "third_party/M2T2"
        if not (source / "config.yaml").is_file():
            raise FileNotFoundError("Run scripts/fetch_assets.py to download pinned M2T2 source")
        # Explicit CPU implementation; does not attempt to allocate CUDA memory.
        install()
        sys.path.insert(0, str(source))
        from m2t2.m2t2 import M2T2

        torch.set_num_threads(config.get("cpu_threads", 4))
        torch.manual_seed(config.get("seed", 7))
        self.torch = torch
        self.cfg = OmegaConf.load(source / "config.yaml")
        self.model = M2T2.from_config(self.cfg.m2t2).eval()
        fixes = []
        if config.get("fix_object_mask_filter", True):
            from .m2t2_compat import correct_object_mask_filter
            fixes.append(correct_object_mask_filter(self.model))
        checkpoint = (ROOT / config.get("checkpoint", "third_party/M2T2Models/m2t2.pth")).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"M2T2 checkpoint missing: {checkpoint}; run scripts/fetch_m2t2_weights.py")
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if "checkpoint" not in config and checksum != WEIGHT_SHA256:
            raise ValueError("Default M2T2 checkpoint checksum differs from the pinned author release")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"], strict=True)
        self.metadata = {"network": "official M2T2", "device": "cpu",
                         "pointnet_operators": "gr3 inference-only CPU reference; FPS ties may differ from CUDA",
                         "checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
                         "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                         "weights_repo": WEIGHT_REPO, "weights_revision": WEIGHT_REVISION,
                         "source_gripper": "franka_panda", "base_to_tcp_m": 0.1034,
                         "compatibility_fixes": fixes,
                         "trained_scene_point_count": 16384, "fdh6_quality_validated": False,
                         "score_semantics": "M2T2 contact confidence, not calibrated FDH-6 grasp success probability"}

    def infer(self, frame, target, scene, perception, output, scene_points_world=None,
              target_context_margin_m=None):
        torch = self.torch
        started = time.perf_counter()
        low, high = np.array(perception["workspace_min"]), np.array(perception["workspace_max"])
        low[2] = scene["table_top_m"] - 0.01  # Include the observed support surface.
        points = frame.points_world() if scene_points_world is None else np.asarray(scene_points_world)
        valid = np.isfinite(points).all(-1)
        if scene_points_world is None:
            valid &= frame.depth_m > 0
        valid &= ((points >= low) & (points <= high)).all(-1)
        points = points[valid].astype(np.float32)
        points = crop_target_context(points, target.points_world, target_context_margin_m)
        context_count = len(points)
        if len(points) < 128:
            raise RuntimeError("M2T2 needs an observed scene with table/object context")
        rng = np.random.default_rng(self.config.get("seed", 7))
        indices = rng.choice(len(points), self.cfg.data.num_points, replace=len(points) < self.cfg.data.num_points)
        points = points[indices]
        origin = np.array([*scene["table_center_xy"], scene["table_top_m"]], dtype=np.float32)
        local = torch.from_numpy(points - origin)
        data = {"points": local[None], "inputs": (local - local.mean(0))[None],
                "object_inputs": torch.rand(1, 1024, 6), "task_is_place": torch.zeros(1),
                "ee_pose": torch.eye(4)[None], "bottom_center": torch.zeros(1, 3)}
        with torch.inference_mode():
            prediction = self.model.infer(data, self.cfg.eval)
        poses, scores, contacts, instance_ids = [], [], [], []
        for index, (pose, confidence, contact) in enumerate(zip(
                prediction["grasps"][0], prediction["grasp_confidence"][0], prediction["grasp_contacts"][0])):
            pose = pose.numpy().astype(float)
            pose[:, :3, 3] += origin
            poses.extend(pose)
            scores.extend(confidence.numpy())
            contacts.extend(contact.numpy() + origin)
            instance_ids.extend([index] * len(pose))
        poses = np.asarray(poses).reshape(-1, 4, 4)
        scores = np.asarray(scores)
        contacts = np.asarray(contacts).reshape(-1, 3)
        target_distances = cKDTree(target.points_world).query(contacts)[0]
        np.savez_compressed(Path(output) / "m2t2_raw.npz", T_world_gripper_base=poses,
                            confidence=scores, contact_world_m=contacts, instance_id=instance_ids,
                            distance_to_selected_cloud_m=target_distances,
                            input_points_world_m=points, model_frame_origin_world_m=origin)
        # Keep the network confidence order; use the observed target mask to
        # select its contacts, then suppress nearly duplicate predictions.
        result = []
        candidates = np.flatnonzero(target_distances <= self.config.get("target_contact_radius_m", 0.005))
        for index in candidates[np.argsort(-scores[candidates])]:
            pose = base_to_tcp(poses[index])
            if any(np.linalg.norm(pose[:3, 3] - g.T_world_tcp[:3, 3]) < 0.012 and
                   np.trace(pose[:3, :3].T @ g.T_world_tcp[:3, :3]) > 2.8 for g in result):
                continue
            result.append(Grasp(pose, float(scores[index]), "m2t2:franka_panda"))
            if len(result) >= self.config.get("topk", 24):
                break
        self.metadata.update(inference_s=time.perf_counter() - started,
                             scene_input="single_rgbd_frame" if scene_points_world is None else "fused_head_rgbd_frames",
                             target_context_margin_m=target_context_margin_m, context_scene_points=context_count,
                             input_target_neighborhood_fraction=float(np.mean(cKDTree(target.points_world).query(points)[0] <= .003)),
                             observed_scene_points=int(valid.sum()), input_scene_points=len(points),
                             raw_grasps=len(poses), target_contact_grasps=len(candidates), selected_grasps=len(result),
                             objectness=prediction["objectness"][0].numpy().tolist())
        if not result:
            raise RuntimeError(f"M2T2 produced no contact candidates on the selected object ({len(poses)} scene candidates)")
        return result
