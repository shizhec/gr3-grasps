"""Known-mesh object pose from observed RGB-D surfaces.

Assumes a rigid object resting upright on a calibrated horizontal table, so the
unknowns are table-plane position and yaw. Geometry (observed points, free
space seen by the cameras) narrows the yaw; rendering the known textured model
at the remaining hypotheses and comparing colors resolves shape symmetries.
Simulator object poses are never read here.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import transform, transform_points

DEFAULTS = {"model_voxel_m": 0.003, "observed_voxel_m": 0.003, "yaw_grid_deg": 10,
            "iterations": 40, "trim_start_m": 0.02, "trim_final_m": 0.004,
            "residual_cap_m": 0.01, "free_space_margin_m": 0.01, "free_space_weight_m": 0.05,
            "table_clearance_m": 0.0, "refine_top": 8,
            "sweep_step_deg": 5, "sweep_polish_m": 0.003, "geometric_tolerance_m": 0.001}


def voxel_downsample(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def model_points(model, body, voxel):
    """Body-frame vertices of the object's visual meshes (the known model, not its pose)."""
    result = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body or model.geom_contype[gid] or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        adr, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        R = Rotation.from_quat(model.geom_quat[gid][[1, 2, 3, 0]]).as_matrix()
        result.append(model.mesh_vert[adr:adr + count] @ R.T + model.geom_pos[gid])
    if not result:
        raise ValueError("Object has no visual mesh")
    return voxel_downsample(np.concatenate(result), voxel)


def planar_pose(yaw, xy, z):
    return transform(Rotation.from_euler("z", yaw).as_matrix(), [*xy, z])


def free_space_violation(T, points, frames, margin):
    """Fraction of projected model points lying in space a camera saw through."""
    world = transform_points(T, points)
    violations, total = 0, 0
    for frame in frames:
        local = transform_points(np.linalg.inv(frame.T_world_camera), world)
        z = local[:, 2]
        u = np.rint(frame.K[0, 0] * local[:, 0] / z + frame.K[0, 2]).astype(int)
        v = np.rint(frame.K[1, 1] * local[:, 1] / z + frame.K[1, 2]).astype(int)
        h, w = frame.depth_m.shape
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        observed = frame.depth_m[v[inside], u[inside]]
        valid = observed > 0
        violations += int(np.count_nonzero(z[inside][valid] < observed[valid] - margin))
        total += int(np.count_nonzero(valid))
    return violations / max(total, 1)


class GeometricScore:
    """Capped observed-to-model distance plus the free-space penalty, for planar poses."""

    def __init__(self, observed_world, body_points, table_top, frames, settings):
        self.s = settings
        self.observed = voxel_downsample(np.asarray(observed_world, dtype=float), settings["observed_voxel_m"])
        if len(self.observed) < 30:
            raise RuntimeError(f"Too few observed target points for pose estimation: {len(self.observed)}")
        self.body, self.frames = body_points, frames
        self.tree = cKDTree(body_points)
        self.z = table_top + settings["table_clearance_m"] - body_points[:, 2].min()
        self.center = body_points[:, :2].mean(0)

    def __call__(self, yaw, xy):
        T = planar_pose(yaw, xy, self.z)
        d = self.tree.query(transform_points(np.linalg.inv(T), self.observed))[0]
        residual = float(np.mean(np.minimum(d, self.s["residual_cap_m"])))
        violation = free_space_violation(T, self.body, self.frames, self.s["free_space_margin_m"])
        return dict(yaw_deg=float(np.rad2deg(yaw) % 360), xy=np.asarray(xy, dtype=float).tolist(),
                    residual_m=residual, free_space_violation=violation,
                    inlier_fraction_3mm=float(np.mean(d < .003)),
                    score_m=residual + self.s["free_space_weight_m"] * violation)

    def icp(self, yaw, xy, fixed_yaw=False):
        """Trimmed point-to-point ICP in the table plane; height selects matching surfaces."""
        s = self.s
        for k in range(s["iterations"]):
            c, si = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -si], [si, c]])
            local = np.c_[(self.observed[:, :2] - xy) @ R, self.observed[:, 2] - self.z]
            d, idx = self.tree.query(local)
            keep = d < max(s["trim_start_m"] * 0.9 ** k, s["trim_final_m"])
            if np.count_nonzero(keep) < 10:
                break
            a, b = self.body[idx[keep], :2], self.observed[keep, :2]
            if fixed_yaw:
                xy = xy + np.mean(b - (a @ R.T + xy), axis=0)
                continue
            ma, mb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
            Rn = (U @ Vt).T
            if np.linalg.det(Rn) < 0:
                Vt[-1] *= -1
                Rn = (U @ Vt).T
            yaw, xy = np.arctan2(Rn[1, 0], Rn[0, 0]), mb - Rn @ ma
        return yaw, xy

    def refine(self, yaw, xy, step_m=.005, step_deg=8):
        # ICP slides on round bodies and snaps to inner walls; refine on the score itself.
        x0 = np.r_[xy, yaw]
        simplex = x0 + np.array([[0, 0, 0], [step_m, 0, 0], [0, step_m, 0], [0, 0, np.deg2rad(step_deg)]])
        result = minimize(lambda x: self(x[2], x[:2])["score_m"], x0, method="Nelder-Mead",
                          options=dict(initial_simplex=simplex, xatol=2e-4, fatol=1e-6, maxfev=300))
        return result.x[2], result.x[:2]


def estimate_upright_pose(observed_world, body_points, table_top, frames, settings=None):
    """Geometric stage: multi-start planar ICP, then score refinement of the best starts."""
    s = {**DEFAULTS, **(settings or {})}
    score = GeometricScore(observed_world, body_points, table_top, frames, s)
    hypotheses = []
    for yaw0 in np.deg2rad(np.arange(0, 360, s["yaw_grid_deg"])):
        c, si = np.cos(yaw0), np.sin(yaw0)
        xy0 = score.observed[:, :2].mean(0) - np.array([[c, -si], [si, c]]) @ score.center
        hypotheses.append(dict(yaw0_deg=float(np.rad2deg(yaw0)), **score(*score.icp(yaw0, xy0))))
    for h in sorted(hypotheses, key=lambda h: h["score_m"])[:s["refine_top"]]:
        hypotheses.append(dict(yaw0_deg=h["yaw0_deg"], refined=True,
                               **score(*score.refine(np.deg2rad(h["yaw_deg"]), np.array(h["xy"])))))
    best = min(hypotheses, key=lambda h: h["score_m"])
    T = planar_pose(np.deg2rad(best["yaw_deg"]), best["xy"], score.z)
    return T, dict(settings=s, observed_points=len(score.observed), model_points=len(body_points),
                   best=best, hypotheses=hypotheses, assumption="upright on horizontal table; planar x/y/yaw")


def yaw_sweep(score, best):
    """Best position for every yaw, starting with the model centroid where the best fit put it."""
    c, si = np.cos(np.deg2rad(best["yaw_deg"])), np.sin(np.deg2rad(best["yaw_deg"]))
    anchor = np.array([[c, -si], [si, c]]) @ score.center + best["xy"]
    sweep = []
    for yaw in np.deg2rad(np.arange(0, 360, score.s["sweep_step_deg"])):
        c, si = np.cos(yaw), np.sin(yaw)
        _, xy = score.icp(yaw, anchor - np.array([[c, -si], [si, c]]) @ score.center, fixed_yaw=True)
        sweep.append(score(yaw, xy))
    return sweep


class ColorCheck:
    """Render the known textured model at a hypothesis from each recorded head pose."""

    def __init__(self, sim, frames, masks, head_qpos):
        self.sim, self.frames, self.masks, self.head_qpos = sim, frames, masks, head_qpos
        self.model = sim.model
        self.qid = sim.model.joint("target_free").qposadr[0]
        self.head = [sim.model.joint(n).qposadr[0] for n in ("head_yaw_joint", "head_pitch_joint")]
        self.data = mujoco.MjData(sim.model)
        # Robot proprioception only; the object slot is overwritten by every hypothesis.
        self.data.qpos[:] = sim.data.qpos
        self.object_geoms = np.flatnonzero(sim.model.geom_bodyid == sim.model.body("target").id)

    def render(self, T, view):
        self.data.qpos[self.qid:self.qid + 7] = np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]]
        self.data.qpos[self.head] = self.head_qpos[view]
        mujoco.mj_forward(self.model, self.data)
        cid = self.model.camera("native_camera").id
        if np.linalg.norm(self.data.cam_xpos[cid] - self.frames[view].T_world_camera[:3, 3]) > 1e-3:
            raise RuntimeError("Re-rendered head camera does not match the recorded frame pose")
        renderer = self.sim.renderer
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera="native_camera", scene_option=self.sim.render_options)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()[..., 0].copy()
        renderer.disable_segmentation_rendering()
        return rgb, np.isin(segmentation, self.object_geoms)

    def __call__(self, T):
        """Mean absolute color difference (0..1) over observed-or-rendered object pixels."""
        errors, counts = 0.0, 0
        for view, (frame, mask) in enumerate(zip(self.frames, self.masks)):
            if mask is None:
                continue
            rgb, rendered = self.render(T, view)
            union = mask | rendered
            diff = np.abs(rgb[union].astype(float) - frame.rgb[union].astype(float)) / 255
            errors += diff.sum()
            counts += diff.size
        return errors / max(counts, 1)


def observe_object_pose(sim, cfg, output, settings=None):
    """Head-scan the single tabletop object and fit its known mesh; returns T_world_body."""
    from pathlib import Path
    from PIL import Image
    from .cli import write_json
    from .geometry import RGBDFrame
    from .head_scan import capture_head_scan
    from .perception import segment_tabletop

    output = Path(output)
    frame = sim.capture()
    clouds = segment_tabletop(frame, cfg["perception"])
    if not clouds:
        raise RuntimeError("No RGB-D foreground object in workspace")
    target = max(clouds, key=lambda c: len(c.points_world))
    _, fused, scan = capture_head_scan(sim, frame, target, {**cfg, "head_scan": {**cfg.get("head_scan", {}), "enabled": True}}, output)
    directory = output / "head_scan"
    frames = [RGBDFrame.load(directory / f"frame_{i:02d}.npz") for i in range(scan["view_count"])]
    masks = [np.asarray(Image.open(directory / f"mask_{i:02d}.png")) > 0 if (directory / f"mask_{i:02d}.png").is_file()
             else None for i in range(scan["view_count"])]
    s = {**DEFAULTS, **(settings or {})}
    points = model_points(sim.model, sim.model.body("target").id, s["model_voxel_m"])
    _, diagnostics = estimate_upright_pose(fused.points_world, points, cfg["scene"]["table_top_m"], frames, s)
    score = GeometricScore(fused.points_world, points, cfg["scene"]["table_top_m"], frames, s)
    geometric = diagnostics["best"]
    sweep = yaw_sweep(score, geometric)
    # Fixed-yaw ICP leaves position slightly off; polish plausible yaws before filtering.
    polished = [score(*score.refine(np.deg2rad(h["yaw_deg"]), np.array(h["xy"]), step_m=.002, step_deg=2))
                for h in sweep if h["score_m"] <= geometric["score_m"] + s["sweep_polish_m"]]
    feasible = [dict(geometric)] + [h for h in polished if h["score_m"] <= geometric["score_m"] + s["geometric_tolerance_m"]]
    color = ColorCheck(sim, frames, masks, [v["measured_rad"] for v in scan["views"]])
    for h in feasible:
        h["color_error"] = color(planar_pose(np.deg2rad(h["yaw_deg"]), h["xy"], score.z))
    chosen = min(feasible, key=lambda h: h["color_error"])
    final = score(*score.refine(np.deg2rad(chosen["yaw_deg"]), np.array(chosen["xy"]), step_m=.002, step_deg=2))
    final["color_error"] = color(planar_pose(np.deg2rad(final["yaw_deg"]), final["xy"], score.z))
    if final["color_error"] > chosen["color_error"]:
        final = chosen  # keep the color-resolved pose if the geometric polish drifted
    T = planar_pose(np.deg2rad(final["yaw_deg"]), final["xy"], score.z)
    diagnostics.update(geometric_best=geometric, best=final, sweep=sweep, feasible=feasible,
                       segmented_clouds=len(clouds), initial_target_pixels=int(target.mask.sum()),
                       fused_target_points=len(fused.points_world), head_views=scan["view_count"],
                       accepted_views=scan["accepted_views"], camera=cfg["camera"],
                       assumption="upright on horizontal table; planar x/y/yaw; color by rendering the known textured model")
    write_json(output / "pose_estimate.json", dict(T_world_body=T, **diagnostics))
    return T, diagnostics


def pose_errors(T_est, T_true, body_points):
    """Translation/yaw errors and ADD / ADD-S (symmetry-tolerant) in metres."""
    est, true = transform_points(T_est, body_points), transform_points(T_true, body_points)
    yaw = np.rad2deg(np.arctan2(*(T_est[:3, :3].T @ T_true[:3, :3])[[1, 0], 0]))
    return dict(translation_xy_m=float(np.linalg.norm(T_est[:2, 3] - T_true[:2, 3])),
                translation_z_m=float(T_est[2, 3] - T_true[2, 3]), yaw_error_deg=float(yaw),
                add_m=float(np.mean(np.linalg.norm(est - true, axis=1))),
                add_s_m=float(np.mean(cKDTree(est).query(true)[0])))
