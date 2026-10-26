"""Offline Dexonomy adapter for the official coupled FDH-6 right hand.

Uses complete object meshes and a floating wrist. This is a grasp synthesis
experiment, not the GR3 camera/arm execution benchmark.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from .model import ROOT, resolve_meshes, values

DEXONOMY_COMMIT = "d561ff9df8545deef7a3f9e7cfb4045ef66f6913"
HAND_SOURCE = ROOT / "third_party/Wiki-GRx-Models/Dexterous_hand/fourier_hand_6dof/urdf/fourier_right_hand_6dof.urdf"
UPSTREAM = ROOT / "third_party/Dexonomy"


# Protocol v2 budget. Larger budgets must be requested explicitly.
DEFAULT_BUDGET = dict(n_init_point=256, n_init_inplane=8, n_final_per_template=16, epoch=1)


def load_arm_prior(path):
    prior = yaml.safe_load(Path(path).read_text())
    if prior.get("side") != "right":
        raise ValueError("The Dexonomy FDH-6 adapter exports only the right hand")
    for key in ("shoulder_world_m", "object_xy_world_m", "table_top_world_m", "object_yaw_deg", "min_thumb_up"):
        if key not in prior:
            raise ValueError(f"Arm prior misses {key}")
    if "reachability_map" in prior:
        path = ROOT / prior["reachability_map"]
        if not path.is_file():
            raise FileNotFoundError(f"{path}; run scripts/build_reachability_map.py")
        # Protocol signatures must change when the sampled map changes.
        prior["reachability_map_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return prior


_REACH_TREES = {}


def _reach_tree(prior):
    """KD-tree over (position / tol, quaternion / chord tol), both quaternion signs."""
    from scipy.spatial import cKDTree

    path = ROOT / prior["reachability_map"]
    stat = path.stat()  # A regenerated map must not reuse a stale tree.
    key = (str(path), stat.st_mtime_ns, stat.st_size, prior["reach_position_tol_m"], prior["reach_rotation_tol_rad"])
    if key not in _REACH_TREES:
        with np.load(path) as data:
            p, q = data["position_world_m"], data["quat_world_wxyz"]
        chord = 2 * np.sin(prior["reach_rotation_tol_rad"] / 4)
        points = np.r_[np.c_[p / prior["reach_position_tol_m"], q / chord],
                       np.c_[p / prior["reach_position_tol_m"], -q / chord]]
        _REACH_TREES[key] = (cKDTree(points), chord)
    return _REACH_TREES[key]


def reachability_distance(position_world, rotation_world, prior):
    """Scaled distance to the nearest sampled GR3 hand-base pose; <= 1 is reachable."""
    from scipy.spatial.transform import Rotation

    tree, chord = _reach_tree(prior)
    q = Rotation.from_matrix(rotation_world).as_quat()[:, [3, 0, 1, 2]]
    return tree.query(np.c_[position_world / prior["reach_position_tol_m"], q / chord])[0]


def arm_prior_metrics(grasp_poses, prior):
    """Score scene-frame wrist poses (x, y, z, qw, qx, qy, qz) for the GR3 right arm.

    FDH-6 right hand base axes, checked from fingertip positions: +X points to
    the thumb side, +Y out of the palm, -Z along the fingers. The scene table
    plane is z=0 and the object sits at the scene XY origin; the GR3 world
    places it at object_xy_world_m with object_yaw_deg about the table normal.
    """
    from scipy.spatial.transform import Rotation

    poses = np.atleast_2d(np.asarray(grasp_poses, dtype=float))
    R_world_scene = Rotation.from_euler("z", prior["object_yaw_deg"], degrees=True).as_matrix()
    R = R_world_scene @ Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    hand_world = poses[:, :3] @ R_world_scene.T + [*prior["object_xy_world_m"], prior["table_top_world_m"]]
    # Reach direction from the shoulder to the wrist: palm should face along it.
    reach = hand_world - np.asarray(prior["shoulder_world_m"])
    reach /= np.linalg.norm(reach, axis=1, keepdims=True)
    horizontal = reach * [1, 1, 0]
    horizontal /= np.maximum(np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-9)
    thumb, palm, fingers = R[:, :, 0], R[:, :, 1], -R[:, :, 2]
    metrics = dict(thumb_up=thumb[:, 2], approach_alignment=np.einsum("ij,ij->i", palm, reach),
                   finger_forward=np.einsum("ij,ij->i", fingers, horizontal))
    if "reachability_map" in prior:
        metrics["reach_distance"] = reachability_distance(hand_world, R, prior)
        # The wrist must also reach the same orientation after the lift.
        metrics["lift_reach_distance"] = reachability_distance(
            hand_world + [0, 0, prior["lift_check_m"]], R, prior)
    return metrics


def arm_prior_mask(grasp_poses, prior):
    """Hard right-arm prior: no thumb-down wraps, and GR3-reachable grasp/lift poses.

    The cone thresholds are optional; the sampled reachability map replaces them.
    """
    m = arm_prior_metrics(grasp_poses, prior)
    keep = m["thumb_up"] >= prior["min_thumb_up"]
    if "min_approach_alignment" in prior:
        keep &= m["approach_alignment"] >= prior["min_approach_alignment"]
    if "min_finger_forward" in prior:
        keep &= m["finger_forward"] >= prior["min_finger_forward"]
    if "reachability_map" in prior:
        keep &= (m["reach_distance"] <= 1) & (m["lift_reach_distance"] <= 1)
    return keep


def install_arm_prior(prior, records):
    """Apply the prior before upstream InitFilter, whose own filters are soft.

    Upstream pads rejected samples back up to a minimum count, and its pose
    filter is disabled for objects at the scene origin. Removing rows here is
    a hard constraint; upstream filters then only pad from accepted samples.
    Returns a callable that restores the original filter.
    """
    import dexonomy.op.gen_init as upstream

    original = upstream.InitFilter.forward

    def forward(self, data):
        keep = arm_prior_mask(data["grasp_pose"].detach().cpu().numpy(), prior)
        records.append(dict(sampled=int(len(keep)), accepted=int(keep.sum())))
        if not keep.any():
            return {k: v[:0] for k, v in data.items()}
        index = upstream.torch.as_tensor(np.flatnonzero(keep), device=data["grasp_pose"].device)
        return original(self, {k: v[index] for k, v in data.items()})

    upstream.InitFilter.forward = forward
    return lambda: setattr(upstream.InitFilter, "forward", original)


# Replaces upstream's 40 mm whole-hand pregrasp clearance. Execution allows
# whole-hand target contact (<= 3 mm) only during approach/close, never on the
# global path to the first pregrasp pose, and hand/table contact never.
PALM_RETREAT = dict(retreat_m=[.08, .06], lift_bias=[0., .5], open_ratios=[1., .6, .3], samples=9,
                    clearance_m=.02, approach_penetration_m=.001, table_clearance_m=.005)


def _env_geoms(model):
    hand, target, table = [], [], []
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = model.body(model.geom_bodyid[g]).name
        if body.startswith("hand-"):
            hand.append(g)
        elif body.startswith("obj-"):
            target.append(g)
        elif model.geom_bodyid[g] == 0 and model.geom(g).name.startswith("plane-"):
            table.append(g)
    if not (hand and target and table):
        raise ValueError("Expected hand, object and table collision geometry")
    return hand, target, table


def hand_clearances(env, hand_qpos, distmax=.1):
    """Minimum hand/object and hand/table signed distances (m) at a hand pose."""
    model, data = env._model, env._data
    env.reset_qpos(np.asarray(hand_qpos), set_ctrl=False)
    hand, target, table = _env_geoms(model)
    distance = lambda others: min(mujoco.mj_geomDistance(model, data, h, o, distmax, None)
                                  for h in hand for o in others)
    return distance(target), distance(table)


def palm_retreat_pregrasp(env, grasp_qpos, meta, settings=PALM_RETREAT, reachable=None):
    """Back the wrist off along -palm normal with opened fingers; far -> near path.

    FDH-6 +Y is the palm normal. The thumb yaw keeps its grasp value; thumb
    pitch and finger flexion open toward 0 (the URDF open limit). Returns
    (path, info) or (None, info) with the closest failing check.
    """
    from scipy.spatial.transform import Rotation

    grasp_qpos = np.asarray(grasp_qpos, dtype=float)
    palm = Rotation.from_quat(grasp_qpos[[4, 5, 6, 3]]).as_matrix()[:, 1]
    active_index = [meta["joint_order"].index(n) for n in meta["active_joints"]]
    closed = grasp_qpos[7:][active_index]
    opened = closed.copy()
    opened[1:] = 0
    failures = []
    for bias in settings["lift_bias"]:
        direction = -palm + [0, 0, bias]
        direction /= np.linalg.norm(direction)
        for retreat in settings["retreat_m"]:
            for ratio in settings["open_ratios"]:
                hand = expand_hand_q(closed + ratio * (opened - closed), meta)
                path = np.array([np.r_[grasp_qpos[:3] + s * retreat * direction, grasp_qpos[3:7], hand]
                                 for s in np.linspace(1, 0, settings["samples"])])
                info = dict(direction=direction.tolist(), retreat_m=retreat, open_ratio=ratio, lift_bias=bias)
                if reachable is not None and not reachable(path[0][:7]):
                    failures.append(dict(info, reason="pregrasp_wrist_unreachable"))
                    continue
                clear = [hand_clearances(env, q) for q in path]
                target, table = np.array(clear).T
                info.update(far_target_clearance_m=float(target[0]),
                            min_approach_target_m=float(target.min()), min_table_m=float(table.min()))
                if table.min() < settings["table_clearance_m"]:
                    failures.append(dict(info, reason="table_clearance"))
                elif target[0] < settings["clearance_m"]:
                    failures.append(dict(info, reason="far_pose_target_clearance"))
                elif target.min() < -settings["approach_penetration_m"]:
                    failures.append(dict(info, reason="approach_target_penetration"))
                else:
                    return path, info
    return None, dict(reason="no_retreat_passed", attempts=failures)


def apply_palm_retreat(output, object_ids, prior=None, settings=PALM_RETREAT):
    """Rewrite pregrasp_qpos of synthesized grasps; move failures out of grasp_data.

    The upstream path is kept as upstream_pregrasp_qpos. Rejected candidates go
    to run/pregrasp_rejected with the same relative path, so downstream scripts
    that glob run/grasp_data only see executable candidates.
    """
    from dexonomy.sim import HandCfg, MuJoCo_OptEnv, MuJoCo_OptCfg
    from dexonomy.util.file_util import load_scene_cfg

    output = Path(output)
    meta = json.loads((output / "assets/hand/fdh6/hand_meta.json").read_text())
    hand_xml = str(output / "assets/hand/fdh6/hand.xml")
    reachable = None
    if prior is not None and "reachability_map" in prior:
        def reachable(wrist):
            m = arm_prior_metrics(wrist, prior)
            return bool(m["reach_distance"][0] <= 1)
    envs, report = {}, {}
    for path in sorted((output / "run/grasp_data").glob("*/*/*.npy")):
        if path.parent.name not in object_ids:
            continue
        grasp = np.load(path, allow_pickle=True).item()
        relative = str(path.relative_to(output / "run/grasp_data"))
        if grasp.get("pregrasp_source") == "palm_retreat":
            continue  # Resumed run: keep the recorded decision.
        if grasp["scene_path"] not in envs:
            envs[grasp["scene_path"]] = MuJoCo_OptEnv(HandCfg(hand_xml, freejoint=True),
                load_scene_cfg(grasp["scene_path"]), MuJoCo_OptCfg(obj_margin=0))
        retreat, info = palm_retreat_pregrasp(envs[grasp["scene_path"]], grasp["grasp_qpos"][-1],
                                              meta, settings, reachable)
        report[relative] = dict(accepted=retreat is not None, **info)
        if retreat is None:
            rejected = output / "run/pregrasp_rejected" / relative
            rejected.parent.mkdir(parents=True, exist_ok=True)
            path.rename(rejected)
            continue
        grasp.update(upstream_pregrasp_qpos=grasp["pregrasp_qpos"], pregrasp_source="palm_retreat",
                     pregrasp_qpos=retreat.astype(np.float32), pregrasp_retreat=info)
        np.save(path, grasp)
    previous = output / "pregrasp_retreat.json"
    if previous.exists():
        report = {**json.loads(previous.read_text()), **report}
    previous.write_text(json.dumps(report, indent=2) + "\n")
    return report


def record_experiment(output, refinement, budget=None, prior=None, pregrasp=None):
    """Prevent mixing different model/template/protocol revisions on resume."""
    output = Path(output)
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if actual != DEXONOMY_COMMIT:
        raise RuntimeError(f"Dexonomy revision mismatch: {actual}; run setup_dexonomy.sh")
    budget = budget or DEFAULT_BUDGET
    signature = dict(protocol_version=2, dexonomy_commit=actual, refinement=refinement,
        hand_urdf_sha256=hashlib.sha256(HAND_SOURCE.read_bytes()).hexdigest(),
        template_seeds_sha256=hashlib.sha256((ROOT / "configs/dexonomy_fdh6.yaml").read_bytes()).hexdigest(),
        object_manifest_sha256=hashlib.sha256((ROOT / "configs/benchmark10.yaml").read_bytes()).hexdigest(),
        seed=12, **budget,
        annotation_rule="distal pad surface normals v1", obj_visual_mass=0,
        named_object_geoms=True, template_evolution=False)
    if prior is not None:
        # Keep v2 signatures byte-identical when no prior is used.
        signature.update(protocol_version=3, arm_prior=prior)
    if pregrasp is not None:
        signature.update(protocol_version=3, pregrasp=pregrasp)
    path = output / "experiment.json"
    if path.exists() and json.loads(path.read_text())["signature"] != signature:
        raise RuntimeError("Experiment protocol differs; select a new --output directory")
    if not path.exists():
        versions = {pkg: importlib.metadata.version(pkg) for pkg in
                    ("torch", "mujoco", "numpy", "warp-lang", "trimesh", "qpsolvers")}
        path.write_text(json.dumps(dict(signature=signature, versions=versions,
            scope="offline complete mesh, floating wrist, no camera or GR3 arm"), indent=2) + "\n")


def export_grasp_commands(output):
    """Serialize named six-motor commands for a future GR3 planner adapter."""
    from scipy.spatial.transform import Rotation

    output = Path(output)
    meta = json.loads((output / "assets/hand/fdh6/hand_meta.json").read_text())
    ids = [7 + meta["joint_order"].index(name) for name in meta["active_joints"]]
    records = []
    for path in sorted((output / "run/grasp_data").rglob("*.npy")):
        grasp = np.load(path, allow_pickle=True).item()
        phases = {}
        for name in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            phases[name.removesuffix("_qpos")] = []
            for q in grasp[name]:
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
                transform[:3, 3] = q[:3]
                phases[name.removesuffix("_qpos")].append(dict(
                    T_scene_hand_base=transform.tolist(), active_joint_rad=q[ids].tolist()))
        relative = path.relative_to(output / "run/grasp_data")
        records.append(dict(id=str(relative.with_suffix("")), template=grasp["tmpl_name"],
            object_id=path.parent.name, phases=phases,
            official_force_closure_pass=(output / "run/succ_grasp" / relative).is_file()))
    payload = dict(schema_version=1, active_joint_names=meta["active_joints"],
        hand_frame="official R_hand_base_link; identical to wrist attachment frame",
        world_frame="offline scene frame; registration to GR3 world still required",
        scope="synthesized candidates; see lift_summary_*.json for physical outcomes",
        grasps=records)
    (output / "grasp_commands.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def export_hand(directory):
    """Preserve 6 independent actuators and 5 URDF mimic constraints."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    urdf = ET.parse(HAND_SOURCE).getroot()
    resolve_meshes(urdf, HAND_SOURCE)
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", meshdir="", strippath="false",
                  fusestatic="false", discardvisual="false", balanceinertia="true")
    for i, geom in enumerate(urdf.iter("collision")):
        geom.set("name", f"hand_collision_{i}")
    work = directory / "hand.urdf"
    ET.ElementTree(urdf).write(work)
    model = mujoco.MjModel.from_xml_path(str(work))
    xml_path = directory / "hand.xml"
    mujoco.mj_saveLastXML(str(xml_path), model)
    xml = ET.parse(xml_path).getroot()
    for old in xml.findall("equality"):
        xml.remove(old)
    equality = ET.SubElement(xml, "equality")
    actuator = ET.SubElement(xml, "actuator")
    active, mimics = [], {}
    for joint in urdf.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        mimic = joint.find("mimic")
        if mimic is not None:
            parent, ratio = mimic.get("joint"), float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            mimics[name] = dict(parent=parent, ratio=ratio, offset=offset)
            ET.SubElement(equality, "joint", joint1=name, joint2=parent,
                          polycoef=values([offset, ratio, 0, 0, 0]),
                          solref="0.002 1", solimp="0.9999 0.99999 0.0001")
        else:
            active.append(name)
            limit = joint.find("limit")
            ET.SubElement(actuator, "position", name=f"servo_{name}", joint=name,
                          kp="2", kv="0.08", forcerange="-0.25 0.25",
                          ctrlrange=f"{limit.get('lower')} {limit.get('upper')}")
    ET.indent(xml)
    ET.ElementTree(xml).write(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = [model.joint(i).name for i in range(model.njnt)]
    meta = dict(source=str(HAND_SOURCE), active_joints=active, joint_order=names,
                mimic=mimics, kp=2, kv=.08, torque_limit_nm=.25,
                dexonomy_commit=DEXONOMY_COMMIT, nq=model.nq, nu=model.nu, neq=model.neq)
    (directory / "hand_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if (model.nq, model.nu, model.neq) != (11, 6, 5):
        raise RuntimeError(f"Unexpected coupled hand dimensions: {meta}")
    return xml_path, meta


def expand_hand_q(active_q, meta):
    """Map six independent angles to the exact eleven-coordinate URDF pose."""
    if len(active_q) != 6:
        raise ValueError("FDH-6 requires six active angles")
    q = dict(zip(meta["active_joints"], active_q, strict=True))
    for name, spec in meta["mimic"].items():
        q[name] = q[spec["parent"]] * spec["ratio"] + spec["offset"]
    return np.array([q[name] for name in meta["joint_order"]])


def annotate_templates(hand_path, meta, seed_path=ROOT / "configs/dexonomy_fdh6.yaml"):
    """Create surface-exact contact seeds on the CAD collision hulls.

    Find surface points on the distal pads (thumb +Y, other fingers -Y in
    official link coordinates). These geometric seeds need validation; they do not
    encode demonstrated or human-approved grasp quality.
    """
    from dexonomy.sim import HandCfg, MuJoCo_VisEnv

    directory = Path(hand_path).parent
    env = MuJoCo_VisEnv(HandCfg(str(hand_path)), vis_mode="collision")
    groups, skeleton = [], {}
    for name in env.get_all_body_names():
        name = name.removeprefix("hand-")
        mesh = env.get_body_mesh(name)
        lo, hi = mesh.bounds
        axis = int(np.argmax(hi - lo))
        a, b = mesh.bounds.mean(0), mesh.bounds.mean(0)
        a[axis], b[axis] = lo[axis], hi[axis]
        skeleton[name] = [[float(x) for x in np.r_[a, b]]]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        groups.append([name for name in skeleton if f"R_{finger}_" in name])
    (directory / "body_group.yaml").write_text(yaml.safe_dump(dict(body_group=groups)))
    (directory / "skeleton.yaml").write_text(yaml.safe_dump(skeleton))
    (directory / "keypoint.yaml").write_text("{}\n")
    raw = directory / "raw_anno"
    raw.mkdir(exist_ok=True)
    audit = {}
    for template, seed in yaml.safe_load(Path(seed_path).read_text())["templates"].items():
        qpos = expand_hand_q(seed["active_q"], meta)
        rots, trans = env.forward_kinematics(qpos)
        bodies = [f"R_{f}_{'distal' if f == 'thumb' else 'intermediate'}_link" for f in seed["fingers"]]
        centers = []
        for name in bodies:
            bid, mesh = env.get_body_id(name), env.get_body_mesh(name)
            # Distal half of each segment approximates its finger pad region.
            local = mesh.bounds.mean(0)
            local[0] = .75 * mesh.bounds[1, 0]
            centers.append(local @ rots[bid].T + trans[bid])
        center = np.mean(centers, axis=0)
        contact, distances = {}, {}
        for name in bodies:
            bid, mesh = env.get_body_id(name), env.get_body_mesh(name)
            # Do not use the common center to select the surface: that can
            # annotate finger sidewalls instead of the opposing finger pads.
            target = mesh.bounds.mean(0)
            target[0] = .65 * mesh.bounds[1, 0]
            target[1] = mesh.bounds[1, 1] + .02 if "thumb" in name else mesh.bounds[0, 1] - .02
            point, _, face = mesh.nearest.on_surface([target])
            normal = mesh.face_normals[face[0]]
            world_point = point[0] @ rots[bid].T + trans[bid]
            world_normal = normal @ rots[bid].T
            contact[name] = [float(x) for x in np.r_[world_point, world_normal]]
            distances[name] = float(mesh.nearest.on_surface([point[0]])[1][0])
        required = [[bodies[0]], bodies[1:]]
        (raw / f"{template}.yaml").write_text(yaml.safe_dump(dict(qpos=qpos.tolist(),
            contact=contact, required_cbody=required), sort_keys=False))
        audit[template] = dict(active_q=seed["active_q"], contact=contact,
            max_surface_distance_m=max(distances.values()), center_m=center.tolist())
    (directory / "annotation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return list(audit)


def coupled_refinement(output, operation):
    """Project the *synthesis optimizer* to six DOF; never modify evaluation.

    Upstream optimizes with artificial spring forces and soft constraints. FDH-6
    mimic constraints can stretch under those forces. Projecting each optimizer
    step enforces URDF bounds/coupling exactly. This is an algorithm adaptation,
    not a simulation controller or a physical success test.
    """
    import dexonomy.op.gen_grasp as upstream

    meta = json.loads((Path(output) / "assets/hand/fdh6/hand_meta.json").read_text())
    original = upstream.MuJoCo_OptEnv

    class CoupledOptEnv(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._active_ids = [self._model.joint("hand-" + n).id for n in meta["active_joints"]]
            self._qadr = [self._model.jnt_qposadr[i] for i in self._active_ids]
            self._vadr = [self._model.jnt_dofadr[self._model.joint("hand-" + n).id] for n in meta["joint_order"]]
            self._limits = self._model.jnt_range[self._active_ids].copy()

        def step_sim(self, substep):
            for _ in range(substep):
                mujoco.mj_step(self._model, self._data)
                q = np.clip(self._data.qpos[self._qadr], self._limits[:, 0], self._limits[:, 1])
                self._data.qpos[7:18] = expand_hand_q(q, meta)
                # Artificial spring optimization is already quasi-static;
                # remove finger velocity at each projected optimizer step.
                self._data.qvel[self._vadr] = 0
                mujoco.mj_forward(self._model, self._data)

        def get_squeeze_qpos(self, grasp_qpos, hand_cbody, hand_cp_w, cw):
            # Virtual work: map contact torque through the mimic Jacobian
            # before creating six actuator targets. Passive joints receive no
            # independent command. Bound motor torque to our explicit .25 Nm.
            self._data.qfrc_applied[:] = 0
            for body, point, wrench in zip(hand_cbody, hand_cp_w, cw):
                mujoco.mj_applyFT(self._model, self._data, wrench[:3], np.zeros(3),
                    point[:3], self._model.body("hand-" + body).id,
                    self._data.qfrc_applied)
            jac = np.column_stack([expand_hand_q(row, meta) for row in np.eye(6)])
            torque = jac.T @ self._data.qfrc_applied[6:17]
            self._data.qfrc_applied[:] = 0
            active = grasp_qpos[self._qadr]
            target = np.clip(active + np.clip(torque, -.25, .25) / 2,
                             self._limits[:, 0], self._limits[:, 1])
            return np.r_[grasp_qpos[:7], expand_hand_q(target, meta)]

    upstream.MuJoCo_OptEnv = CoupledOptEnv
    try:
        operation()
    finally:
        upstream.MuJoCo_OptEnv = original


def export_object(item, directory):
    """Retain original GSO visual/convex geometry; write its surface in XML frame."""
    import trimesh

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "third_party/mujoco_scanned_objects/models" / item["folder"] / "model.xml"
    xml = ET.parse(source).getroot()
    for asset in xml.findall("asset/*"):
        if asset.get("file"):
            asset.set("file", str((source.parent / asset.get("file")).resolve()))
    # Infer masses from the collision geometry at known density, so evaluation
    # can set the manifest mass without changing the mesh scale.
    for inertial in xml.findall(".//inertial"):
        raise ValueError(f"Explicit inertia requires an adapter: {source}")
    for gid, geom in enumerate(xml.iter("geom")):
        # Dexonomy.set_obj_margin selects geoms by their attached name prefix.
        # Unnamed GSO geoms otherwise keep the initial 45 mm pregrasp margin!
        geom.set("name", f"object_geom_{gid}")
        if geom.get("contype") == "0":
            # Visual meshes also contribute inferred inertia unless explicitly
            # zero-mass. Upstream changes density only on collision geometry.
            # Counting visual mass here would silently corrupt target mass.
            geom.set("mass", "0")
        else:
            geom.attrib.pop("mass", None)
            geom.set("density", "1000")
    xml_path = directory / "object.xml"
    ET.ElementTree(xml).write(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surface = []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] != 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        surface.append(trimesh.Trimesh(verts, model.mesh_face[fa:fa + fn], process=False))
    if not surface:
        raise ValueError(f"No visual surface in {source}")
    mesh = trimesh.util.concatenate(surface)
    mesh_path = directory / "surface.obj"
    mesh.export(mesh_path)
    info_path = directory / "info.json"
    info_path.write_text(json.dumps(dict(mass=float(model.body_mass.sum()), density=1000, scale=1)))
    # Pose at table height zero, using visual bounds as in the original benchmark.
    pose = [0, 0, float(-mesh.bounds[0, 2] + .001), 1, 0, 0, 0]
    scene = dict(scene={item["id"]: dict(type="rigid_object", xml_path=str(xml_path),
                    file_path=str(mesh_path), info_path=str(info_path),
                    scale=[1, 1, 1], pose=pose)},
                 task=dict(type="force_closure", obj_name=item["id"]), scene_id=item["id"])
    scene["scene"]["table"] = dict(type="plane", pose=[0, 0, 0, 1, 0, 0, 0])
    np.save(directory / "scene.npy", scene)
    (directory / "provenance.json").write_text(json.dumps(dict(source=str(source),
        object_id=item["id"], mass_kg=item["mass_kg"], bounds_m=mesh.bounds.tolist(),
        pose=pose, convex_collision_geoms=int(np.count_nonzero(model.geom_contype)),
        surface_frame="original MJCF world frame; no geometry rescale"), indent=2))
    return directory / "scene.npy"


def configuration(output, op_name, scene_pattern=None, mass=.1, budget=None):
    """Load official algorithm defaults, overriding only paths and batch sizes."""
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    cfg = OmegaConf.load(UPSTREAM / "dexonomy/config/base.yaml")
    del cfg["defaults"]
    del cfg["hydra"]
    cfg.hand_name, cfg.op_name = "fdh6", op_name
    cfg.asset_root, cfg.save_root = str(output / "assets"), str(output)
    cfg.save_dir = str(output / "run")
    cfg.n_worker, cfg.init_gpu, cfg.traj_gpu = 1, [0], [0]
    cfg.skip_done, cfg.tmpl_upd_mode, cfg.obj_mass = True, "disabled", mass
    cfg.hand = dict(xml_path=str(output / "assets/hand/fdh6/hand.xml"), arm_hand=None)
    cfg.op = OmegaConf.load(UPSTREAM / f"dexonomy/config/op/{op_name}.yaml")
    if op_name == "init":
        budget = budget or DEFAULT_BUDGET
        cfg.op.epoch = budget["epoch"]
        cfg.op.object.batch_size = 1
        cfg.op.object.n_init_point = budget["n_init_point"]
        cfg.op.object.n_init_inplane = budget["n_init_inplane"]
        cfg.op.object.n_cfg = None
        cfg.op.object.cfg_path = scene_pattern
        cfg.op.filter.general.n_final = budget["n_final_per_template"]
    return cfg


def audited_grasp(cfg, refinement="coupled"):
    """Record final filter decisions without relaxing upstream thresholds."""
    import dexonomy.op.gen_grasp as upstream

    records = []
    original = upstream.GraspFilter

    class AuditedFilter(original):
        def forward(self, info):
            ok, ho = super().forward(info)
            env, model, data = self.sim_env, self.sim_env._model, self.sim_env._data
            hh = env.get_contacts(.002)[1]
            limits = {}
            for j in range(model.njnt):
                if model.jnt_limited[j]:
                    q = data.qpos[model.jnt_qposadr[j]]
                    violation = max(model.jnt_range[j, 0] - q, q - model.jnt_range[j, 1], 0)
                    if violation:
                        limits[model.joint(j).name] = float(violation)
            reason = "passed"
            if not ok:
                if limits:
                    reason = "joint_limit"
                elif len(hh["dist"]) and min(hh["dist"]) < self.cfg.collision.hh_thre:
                    reason = "hand_self_collision"
                elif len(ho["dist"]) and min(ho["dist"]) < self.cfg.collision.ho_thre:
                    reason = ("hand_object_clearance_below_40mm" if self.cfg.collision.ho_thre > 0
                              else "hand_scene_collision")
                elif not len(ho["dist"]):
                    reason = "no_contacts"
                elif info and any(not set(group).intersection(ho["bn1"]) for group in info["required_cbody"]):
                    reason = "missing_required_finger_contacts"
                else:
                    reason = "force_closure_qp"
            errors = []
            for eq in range(model.neq):
                if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                    j1, j2 = model.eq_obj1id[eq], model.eq_obj2id[eq]
                    errors.append(abs(data.qpos[model.jnt_qposadr[j1]] - model.eq_data[eq, 1] * data.qpos[model.jnt_qposadr[j2]]))
            records.append(dict(path=self.log_name, passed=bool(ok), reason=reason,
                limit_violation_rad=limits, max_mimic_error_rad=float(max(errors, default=0)),
                contact_bodies=ho["bn1"], hand_qpos=env.get_hand_qpos().tolist()))
            return ok, ho

    upstream.GraspFilter = AuditedFilter
    try:
        operation = lambda: upstream.operate_grasp(cfg)
        if refinement == "coupled":
            coupled_refinement(cfg.save_root, operation)
        else:
            operation()
    finally:
        upstream.GraspFilter = original
    path = Path(cfg.save_root) / f"diagnostics_{cfg.debug_name}.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    return records
