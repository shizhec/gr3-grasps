"""Full GR3 arm + six-motor FDH-6 execution of an offline Dexonomy grasp.

Object registration uses simulator ground truth for this controlled experiment.
The arm and hand move exclusively through motor targets in physical execution.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from .geometry import transform, rotation_error
from .model import ROOT, build_scene, values
from .dexonomy_backend import expand_hand_q
from .planning import Planner


def pose(q):
    q = np.asarray(q)
    return transform(Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix(), q[:3])


def interpolate_hand_pose(a, b, alpha):
    result = (1 - alpha) * a + alpha * b
    rot = Slerp([0, 1], Rotation.from_quat(np.stack([a[3:7], b[3:7]])[:, [1, 2, 3, 0]]))
    result[3:7] = rot(alpha).as_quat()[[3, 0, 1, 2]]
    return result


def compiled_model_sha256(model):
    """Fingerprint physics and mesh data, independent of XML MIME inference/cache."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer).hexdigest()


def build_experiment(settings, directory):
    """Use the same hand contact model/object inertia as standalone validation."""
    if settings.get("contact_profile") != "dexonomy_eval":
        raise ValueError("Only the explicitly selected dexonomy_eval contact profile is supported")
    cfg = yaml.safe_load((ROOT / settings["base_config"]).read_text())
    cfg["scene"]["objects"] = [dict(name="target", type="mjcf", mass=settings["mass_kg"],
        placement_xy=settings["placement_xy"], rotation_rpy_deg=[0, 0, settings.get("yaw_deg", 0)],
        mjcf=f"third_party/mujoco_scanned_objects/models/{settings['object_folder']}/model.xml")]
    cfg["scene"]["timestep_s"] = settings["timestep_s"]
    cfg["planning"]["pregrasp_distance_m"] = 0.  # Exact Dexonomy first waypoint.
    path = build_scene(cfg, directory)
    xml = ET.parse(path).getroot()
    xml.find("option").attrib.update(integrator="discrete", impratio="10", noslip_iterations="0")
    grasp = np.load(ROOT / settings["candidate"], allow_pickle=True).item()
    source_scene = np.load(grasp["scene_path"], allow_pickle=True).item()
    source = source_scene["scene"][settings["object_id"]]
    info = json.loads(Path(source["info_path"]).read_text())
    spec = mujoco.MjSpec.from_file(source["xml_path"])
    density = settings["mass_kg"] / (info["mass"] / info["density"])
    for geom in spec.geoms:
        if geom.contype:
            geom.density = density
    obj = spec.compile()
    if obj.nbody != 2 or not np.isclose(obj.body_mass.sum(), settings["mass_kg"]):
        raise ValueError("Expected one rigid object with manifest mass")
    body = xml.find("worldbody/body[@name='target']")
    inertial = body.find("inertial")
    inertial.attrib.update(mass=str(settings["mass_kg"]), pos=values(obj.body_ipos[1]),
        quat=values(obj.body_iquat[1]), diaginertia=values(obj.body_inertia[1]))
    for b in xml.iter("body"):
        if b.get("name", "").startswith(("R_", "L_")):
            for geom in b.findall("geom"):
                if geom.get("contype", "1") != "0":
                    geom.attrib.update(solref="0.005 1", solimp="0.5 0.99 0.0001", condim="4",
                                       friction="1 0.005 0.0001")
    for geom in body.findall("geom"):
        if geom.get("contype", "1") != "0":
            geom.attrib.update(solref="0.02 1", solimp="0.9 0.95 0.001", condim="4",
                               friction="1 0.005 0.0001")
    xml.find("worldbody/geom[@name='table']").attrib.update(condim="4", friction="1 0.005 0.0001")
    ET.indent(xml)
    ET.ElementTree(xml).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nmocap or np.any(model.eq_type != mujoco.mjtEq.mjEQ_JOINT):
        raise ValueError("Full GR3 execution must have no mocap, weld or object attachment")
    provenance = dict(source_sha256=hashlib.sha256((ROOT / settings["candidate"]).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), settings=settings,
        compiled_model_sha256=compiled_model_sha256(model),
        registration="offline simulator object-pose oracle", full_gr3=True,
        fixed_base=True, ideal_robot_gravity_compensation=True, object_attachment=False,
        hand_mocap=False, runtime_joint_projection=False, contact_profile="dexonomy_eval",
        object_mass_kg=float(model.body_mass[model.body('target').id]), mujoco_version=mujoco.__version__)
    return path, cfg, grasp, source, provenance


def body_descendants(model, root):
    result = set()
    for bid in range(model.nbody):
        current = bid
        while current:
            if current == root:
                result.add(bid)
                break
            current = int(model.body_parentid[current])
    return result


def forbidden_contacts(model, data, moving, hand_bodies, target, allow_contact):
    """Use the actual hand shape, permitting bounded whole-hand target contact."""
    result = []
    for contact in data.contact:
        b1, b2 = (int(model.geom_bodyid[g]) for g in contact.geom)
        target_pair = target in (b1, b2)
        if b1 not in moving and b2 not in moving and not target_pair:
            continue
        distance = float(contact.dist)
        if b1 in hand_bodies and b2 in hand_bodies:
            threshold = .003
        elif target in (b1, b2) and (b2 if b1 == target else b1) in hand_bodies and allow_contact:
            threshold = .003
        elif target_pair and 0 in (b1, b2):
            threshold = .003  # Shallow support contact; also check carried object/table geometry.
        else:
            threshold = .001
        if distance < -threshold:
            result.append(dict(geom1=model.geom(int(contact.geom[0])).name,
                geom2=model.geom(int(contact.geom[1])).name, penetration_m=-distance,
                body1=model.body(b1).name, body2=model.body(b2).name))
    return result


@dataclass
class DexterousPlan:
    arm: np.ndarray
    hand: np.ndarray
    times: np.ndarray
    target_tcp: np.ndarray
    source: str


class GraspPathPlanner:
    def __init__(self, sim, cfg, settings, meta, directory, target_points, object_pose=None):
        """object_pose: estimated T_world_body used for every planning copy (None: simulator state)."""
        from .curobo_backend import CuroboPlanner

        self.sim, self.meta, self.settings = sim, meta, settings
        self.hand_ids = np.array([sim.model.joint(n).qposadr[0] for n in meta["joint_order"]])
        self.active_local = [meta["joint_order"].index(n) for n in meta["active_joints"]]
        self.target = sim.model.body("target").id
        hand_root = sim.model.body("R_hand_base_link").id
        self.hand_bodies = body_descendants(sim.model, hand_root)
        self.object_qid = sim.model.joint("target_free").qposadr[0]
        self.carried_relative = None
        self.contact_shape = None
        owner = self

        class WholeHandPlanner(CuroboPlanner):
            def pose(self, q):
                result = super().pose(q)
                if owner.carried_relative is not None:
                    T_hand = transform(self.data.xmat[hand_root].reshape(3, 3), self.data.xpos[hand_root])
                    T_obj = T_hand @ owner.carried_relative
                    self.data.qpos[owner.object_qid:owner.object_qid + 7] = np.r_[
                        T_obj[:3, 3], Rotation.from_matrix(T_obj[:3, :3]).as_quat()[[3, 0, 1, 2]]]
                    mujoco.mj_forward(self.model, self.data)
                return result

            def collisions(self, q, allowed_object=None):
                self.pose(q)
                return forbidden_contacts(self.model, self.data, self.moving_bodies,
                    owner.hand_bodies, owner.target, allowed_object is not None)

        self.arm = WholeHandPlanner(sim.model, sim.data, cfg, "right", directory, target_points)
        self.object_qpos = None
        if object_pose is not None:
            self.object_qpos = np.r_[object_pose[:3, 3], Rotation.from_matrix(object_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]]
            self.arm.data.qpos[self.object_qid:self.object_qid + 7] = self.object_qpos
            mujoco.mj_forward(self.arm.model, self.arm.data)
        self.arm.metadata.update(hand_target_contacts="whole right hand, penetration <=3mm",
            finger_path_recheck="interpolated coupled hand coordinates in MuJoCo",
            gpu_hand_geometry="frozen at preform; dynamic hand geometry rechecked in MuJoCo",
            ik_polish="bounded CPU least-squares only after valid cuRobo IK",
            carried_geometry="kinematic object in planning copy only; free in execution")
        self.diagnostics = []

    def set_hand(self, q):
        self.arm.data.qpos[self.hand_ids] = (expand_hand_q(q, self.meta)
            if self.contact_shape is None else self.contact_shape)

    def make_plan(self, arms, hands, target, source, minimum_duration):
        arms, hands = np.asarray(arms), np.asarray(hands)
        dt = np.maximum(np.max(abs(np.diff(arms, axis=0)), axis=1) / .35,
                        np.max(abs(np.diff(hands, axis=0)), axis=1) / self.settings["max_hand_joint_speed_rad_s"])
        times = np.r_[0., np.cumsum(np.maximum(dt, .01))]
        times *= max(1., minimum_duration / times[-1])
        return DexterousPlan(arms, hands, times, target, source)

    def append_checked(self, arms, hands, goal_arm, goal_hand, allow):
        n = max(1, int(np.ceil(max(np.max(abs(goal_arm-arms[-1])), np.max(abs(goal_hand-hands[-1]))) / .02)))
        start_arm, start_hand = arms[-1].copy(), hands[-1].copy()
        for alpha in np.linspace(0, 1, n+1)[1:]:
            q, hand = (1-alpha)*start_arm+alpha*goal_arm, (1-alpha)*start_hand+alpha*goal_hand
            self.set_hand(hand)
            collision = self.arm.collisions(q, self.target if allow else None)
            if collision:
                raise RuntimeError(f"Path collision: {collision[:2]}")
            arms.append(q.copy()); hands.append(hand.copy())

    def local(self, name, previous, a, targets, world_scene, hand_tcp, minimum_duration):
        arms, hands = [previous.arm[-1].copy()], [previous.hand[-1].copy()]
        begin = a.copy()
        for target in targets:
            angle = np.linalg.norm(rotation_error(pose(target)[:3, :3], pose(begin)[:3, :3]))
            count = max(1, int(np.ceil(max(np.linalg.norm(target[:3]-begin[:3])/.004,
                angle/.04, np.max(abs(target[7:]-begin[7:]))/.02))))
            for alpha in np.linspace(0, 1, count+1)[1:]:
                q = interpolate_hand_pose(begin, target, alpha)
                hand = q[7:][self.active_local]
                T = world_scene @ pose(q) @ hand_tcp
                self.set_hand(hand)
                current = self.arm.pose(arms[-1])
                if np.linalg.norm(current-T) < 1e-7:
                    goal = arms[-1].copy()
                else:
                    goal, pe, re, valid = self.arm.ik(T, arms[-1])
                    if not valid:
                        raise RuntimeError(f"{name}: cuRobo IK failed at alpha={alpha:.3f}, errors={pe:.5f}m/{re:.5f}rad")
                    goal, pe, re, valid = Planner.ik(self.arm, T, goal)
                    if not valid:
                        raise RuntimeError(f"{name}: IK polish failed")
                self.append_checked(arms, hands, goal, hand, True)
            begin = target.copy()
        endpoint = self.arm.pose(arms[-1])
        self.diagnostics.append(dict(stage=name, samples=len(arms), target_tcp=T,
            ik_position_error_m=float(np.linalg.norm(endpoint[:3, 3]-T[:3, 3])),
            ik_rotation_error_rad=float(np.linalg.norm(rotation_error(T[:3, :3], endpoint[:3, :3])))))
        return self.make_plan(arms, hands, T, "curobo_cartesian_ik_with_polish", minimum_duration)

    def plan(self, grasp, world_scene, hand_tcp):
        from .grasps import Grasp
        first = grasp["pregrasp_qpos"][0]
        qhand = first[7:][self.active_local]
        self.set_hand(qhand)
        target = world_scene @ pose(first) @ hand_tcp
        start = self.sim.data.qpos[self.arm.qids].copy()
        pre = self.arm.plan([Grasp(target, 1., "dexonomy_exact_pregrasp")], start)
        arms, hands = list(pre.waypoints.copy()), [qhand.copy() for _ in pre.waypoints]
        goal, pe, re, valid = Planner.ik(self.arm, target, arms[-1])
        if not valid:
            raise RuntimeError("Pregrasp IK polish failed")
        self.append_checked(arms, hands, goal, qhand, False)
        plans = {"pregrasp": self.make_plan(arms, hands, target, "curobo_global", 1.)}
        plans["approach"] = self.local("approach", plans["pregrasp"], first,
            grasp["pregrasp_qpos"][1:], world_scene, hand_tcp, self.settings["approach_duration_s"])
        plans["grasp"] = self.local("grasp", plans["approach"], grasp["pregrasp_qpos"][-1],
            grasp["grasp_qpos"], world_scene, hand_tcp, self.settings["grasp_duration_s"])
        squeeze = grasp["squeeze_qpos"][-1]
        if not np.allclose(pose(squeeze), pose(grasp["grasp_qpos"][-1]), atol=1e-8):
            raise ValueError("This executor expects squeeze at a stationary wrist")
        # A force-producing position target is NOT the reachable hand shape.
        # Check this contact phase with motors in a separate physical rollout.
        plans["squeeze"] = self.make_plan([plans["grasp"].arm[-1]]*2,
            [plans["grasp"].hand[-1], squeeze[7:][self.active_local]],
            plans["grasp"].target_tcp, "motor_contact_rollout", self.settings["squeeze_duration_s"])
        import copy
        preview_sim = copy.copy(self.sim)
        preview_sim.data = mujoco.MjData(self.sim.model)
        mujoco.mj_copyData(preview_sim.data, self.sim.model, self.sim.data)
        if self.object_qpos is not None:
            # Rehearse against the perceived object, never the simulator's true pose.
            preview_sim.data.qpos[self.object_qid:self.object_qid + 7] = self.object_qpos
            preview_sim.data.qvel[sim_dof(self.sim.model, "target_free")] = 0
            mujoco.mj_forward(self.sim.model, preview_sim.data)
        preview_sim.renderer = None
        preview_dir = self.arm.directory.parent / "contact_preview"
        preview_dir.mkdir()
        preview = MotorExecution(preview_sim, self.meta, preview_dir, self.settings)
        try:
            preview.execute(plans, hold=False)
        except Exception as error:
            preview.finish(f"{type(error).__name__}: {error}")
            raise
        preview.finish("planning_preview_without_lift")
        self.contact_shape = preview_sim.data.qpos[self.hand_ids].copy()
        self.arm.metadata.update(contact_geometry="motor rollout actual squeeze shape, frozen for lift planning; actual geometry checked every execution step",
            squeeze_target_interpretation="motor setpoint, never kinematically imposed on physical fingers")
        last = grasp["squeeze_qpos"][-1].copy()
        root_id = self.sim.model.body("R_hand_base_link").id
        T_hand = transform(preview_sim.data.xmat[root_id].reshape(3, 3), preview_sim.data.xpos[root_id])
        T_object = transform(preview_sim.data.xmat[self.target].reshape(3, 3), preview_sim.data.xpos[self.target])
        self.carried_relative = np.linalg.inv(T_hand) @ T_object
        lift = last.copy()
        # Local scene Z is the object's registered upright axis in this experiment.
        lift[2] += self.settings["lift_distance_m"]
        try:
            plans["lift"] = self.local("lift", plans["squeeze"], last, [lift], world_scene,
                                       hand_tcp, self.settings["lift_duration_s"])
        finally:
            self.carried_relative = None
            self.contact_shape = None
        return plans


def sim_dof(model, joint):
    adr = model.joint(joint).dofadr[0]
    return slice(adr, adr + 6)


def visible_object_vertices(model, data, target):
    result = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != target or model.geom_contype[gid] or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        adr, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        result.append(model.mesh_vert[adr:adr+count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    return np.concatenate(result)


class MotorExecution:
    """Observe every physics step; commands never overwrite simulated state."""
    def __init__(self, sim, meta, directory, settings, video=False, no_close=False):
        self.sim, self.meta, self.directory, self.settings = sim, meta, directory, settings
        self.model, self.data = sim.model, sim.data
        from .model import arm_names
        self.arm_ids = np.array([self.model.joint(n).qposadr[0] for n in arm_names("right")])
        self.hand_ids = np.array([self.model.joint(n).qposadr[0] for n in meta["active_joints"]])
        self.arm_motors = [int(np.flatnonzero(sim.motor_qpos == q)[0]) for q in self.arm_ids]
        self.hand_motors = [int(np.flatnonzero(sim.motor_qpos == q)[0]) for q in self.hand_ids]
        self.target = self.model.body("target").id
        self.hand_bodies = body_descendants(self.model, self.model.body("R_hand_base_link").id)
        self.moving = body_descendants(self.model, self.model.body("right_upper_arm_pitch_link").id)
        self.tcp = self.model.site("right_tcp").id
        self.reference = mujoco.MjData(self.model)
        self.baseline = float(self.data.subtree_com[self.target, 2])
        self.samples, self.states, self.stages = [], [], {}
        self.counter, self.hold_steps, self.opposed_steps = 0, 0, 0
        self.hold_min, self.peak_lift = None, 0.
        self.hold_table = False
        self.max_mimic, self.max_penetration = 0., 0.
        self.max_tracking = dict(position_m=0., rotation_rad=0.)
        self.phase, self.failure = "initial", None
        self.no_close, self.open_hand = no_close, None
        self.writer = None
        if video:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(directory / "execution.mp4", fps=15, codec="libx264")

    def command(self, arm, hand):
        self.data.ctrl[self.arm_motors] = arm
        self.data.ctrl[self.hand_motors] = self.open_hand if self.no_close and self.open_hand is not None else hand

    def tick(self, phase):
        self.phase = phase
        self.sim.step()
        self.counter += 1
        if any(w.number for w in self.data.warning):
            raise RuntimeError("MuJoCo numerical warning")
        violations = forbidden_contacts(self.model, self.data, self.moving, self.hand_bodies,
                                        self.target, phase not in ("preform", "pregrasp"))
        if violations:
            raise RuntimeError(f"Physical collision: {violations[:2]}")
        mimic = 0.
        for eq in range(self.model.neq):
            j1, j2 = self.model.eq_obj1id[eq], self.model.eq_obj2id[eq]
            q2 = self.data.qpos[self.model.jnt_qposadr[j2]]
            expected = np.polynomial.polynomial.polyval(q2, self.model.eq_data[eq, :5])
            mimic = max(mimic, abs(self.data.qpos[self.model.jnt_qposadr[j1]] - expected))
        self.max_mimic = max(self.max_mimic, float(mimic))
        if mimic > .01:
            raise RuntimeError(f"Hand mimic error {mimic:.6f} rad >0.01")
        fingers, contacts, table, penetration = set(), [], False, 0.
        for ci, contact in enumerate(self.data.contact):
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in contact.geom)
            if self.target not in (b1, b2):
                continue
            other = b2 if b1 == self.target else b1
            penetration = max(penetration, max(0., -float(contact.dist)))
            if other in self.hand_bodies:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, ci, force)
                name = self.model.body(other).name
                contacts.append(dict(body=name, normal_force_n=float(force[0])))
                if force[0] > .01:
                    fingers.update(f for f in ("thumb", "index", "middle", "ring", "pinky") if f"R_{f}_" in name)
            else:
                table = True  # Any non-hand support prevents hold success.
        self.max_penetration = max(self.max_penetration, penetration)
        if penetration > .003:
            raise RuntimeError(f"Target penetration {penetration:.6f} m >0.003")
        self.reference.qpos[:] = self.data.qpos
        self.reference.qpos[self.arm_ids] = self.data.ctrl[self.arm_motors]
        mujoco.mj_kinematics(self.model, self.reference)
        pe = float(np.linalg.norm(self.reference.site_xpos[self.tcp] - self.data.site_xpos[self.tcp]))
        re = float(np.linalg.norm(rotation_error(self.reference.site_xmat[self.tcp].reshape(3, 3),
                                                self.data.site_xmat[self.tcp].reshape(3, 3))))
        self.max_tracking["position_m"] = max(self.max_tracking["position_m"], pe)
        self.max_tracking["rotation_rad"] = max(self.max_tracking["rotation_rad"], re)
        stat = self.stages.setdefault(phase, dict(max_position_error_m=0., max_rotation_error_rad=0.))
        stat["max_position_error_m"] = max(stat["max_position_error_m"], pe)
        stat["max_rotation_error_rad"] = max(stat["max_rotation_error_rad"], re)
        # During the free-space reach, assess tracking once the arm settles.
        if phase not in ("preform", "pregrasp") and (pe > .015 or re > .10):
            raise RuntimeError(f"TCP tracking error {pe:.6f} m / {re:.6f} rad")
        lift = float(self.data.subtree_com[self.target, 2] - self.baseline)
        self.peak_lift = max(self.peak_lift, lift)
        opposed = "thumb" in fingers and len(fingers) > 1
        if phase == "hold":
            self.hold_steps += 1
            self.opposed_steps += int(opposed)
            self.hold_table |= table
            self.hold_min = lift if self.hold_min is None else min(self.hold_min, lift)
        if self.counter % max(1, round(.02 / self.model.opt.timestep)) == 0:
            self.samples.append(dict(time_s=float(self.data.time), phase=phase, lift_m=lift,
                contacts=contacts, opposed=opposed, non_hand_support=table, penetration_m=penetration,
                mimic_error_rad=float(mimic), tcp_position_error_m=pe, tcp_rotation_error_rad=re,
                arm_target_rad=self.data.ctrl[self.arm_motors].copy(),
                arm_actual_rad=self.data.qpos[self.arm_ids].copy(),
                hand_target_rad=self.data.ctrl[self.hand_motors].copy(),
                hand_actual_rad=self.data.qpos[self.hand_ids].copy(),
                hand_motor_torque_nm=self.data.actuator_force[self.hand_motors].copy()))
            self.states.append(dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                                    ctrl=self.data.ctrl.copy(), time_s=float(self.data.time), phase=phase))
        if self.writer and self.counter % max(1, round(1 / (15 * self.model.opt.timestep))) == 0:
            from PIL import Image, ImageDraw
            panel = Image.fromarray(self.sim.grasp_view(self.target, phase))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 25, panel.width, 48), fill="black")
            pose = "head-camera pose estimate" if self.settings.get("registration") == "camera" else "known object pose"
            draw.text((10, 29), f"Dexonomy + cuRobo | {pose} | fixed base + gravity compensation | COM lift {lift*100:.1f}cm", fill="white")
            self.writer.append_data(np.asarray(panel))
        return pe, re

    def preform(self, target):
        self.baseline = float(self.data.subtree_com[self.target, 2])
        arm = self.data.qpos[self.arm_ids].copy()
        begin = self.data.qpos[self.hand_ids].copy()
        steps = round(self.settings["preform_duration_s"] / self.model.opt.timestep)
        for alpha in np.linspace(0, 1, steps+1)[1:]:
            self.command(arm, begin*(1-alpha)+target*alpha)
            self.tick("preform")
        for _ in range(round(.5 / self.model.opt.timestep)):
            self.tick("preform")
        self.open_hand = target.copy()
        self.baseline = float(self.data.subtree_com[self.target, 2])

    def execute(self, plans, hold=True):
        for phase, plan in plans.items():
            print(f"Executing {phase}: {len(plan.arm)} samples, {plan.times[-1]:.2f}s", flush=True)
            steps = int(np.ceil(plan.times[-1] / self.model.opt.timestep))
            for step in range(1, steps+1):
                t = min(step*self.model.opt.timestep, plan.times[-1])
                arm = np.array([np.interp(t, plan.times, plan.arm[:, i]) for i in range(7)])
                hand = np.array([np.interp(t, plan.times, plan.hand[:, i]) for i in range(6)])
                self.command(arm, hand)
                self.tick(phase)
            for _ in range(round(.5 / self.model.opt.timestep)):
                pe, re = self.tick(phase)
            if pe > .015 or re > .10:
                raise RuntimeError(f"{phase} endpoint tracking {pe:.6f} m / {re:.6f} rad")
            self.stages[phase].update(endpoint_position_error_m=pe, endpoint_rotation_error_rad=re)
        if hold:
            for _ in range(round(self.settings["hold_duration_s"] / self.model.opt.timestep)):
                self.tick("hold")

    def finish(self, failure=None):
        from .cli import write_json
        if self.writer:
            self.writer.close()
        duration = self.hold_steps * self.model.opt.timestep
        opposed = self.opposed_steps / max(1, self.hold_steps)
        success = (failure is None and duration >= self.settings["hold_duration_s"]-.001 and
            self.hold_min is not None and self.hold_min >= self.settings["success_lift_m"] and
            opposed >= .9 and not self.hold_table)
        result = dict(success=bool(success), failure=failure if failure else None if success else "insufficient_lift_or_opposed_hold",
            termination_phase=self.phase, termination_time_s=float(self.data.time), no_close=self.no_close,
            hold_duration_s=duration, hold_min_lift_m=self.hold_min, peak_lift_m=self.peak_lift,
            hold_opposed_fraction=opposed, hold_non_hand_support=self.hold_table,
            max_target_penetration_m=self.max_penetration, max_mimic_error_rad=self.max_mimic,
            max_tracking=self.max_tracking, stages=self.stages,
            thresholds=dict(lift_m=self.settings["success_lift_m"], hold_s=self.settings["hold_duration_s"],
                opposed_fraction=.9, target_penetration_m=.003, mimic_error_rad=.01,
                tcp_position_m=.015, tcp_rotation_rad=.10, forbidden_penetration_m=.001),
            scope="Full GR3 right-arm motors; fixed base and ideal gravity compensation; known mesh and oracle object pose; no camera input")
        result["tracking_policy"] = "Free-space pregrasp assessed after 0.5s settling; approach/grasp/squeeze/lift/hold assessed every physical step"
        result["physics"] = dict(integrator=mujoco.mjtIntegrator(int(self.model.opt.integrator)).name,
            timestep_s=float(self.model.opt.timestep), gravity=self.model.opt.gravity.copy(),
            hand_motor_kp=self.model.actuator_gainprm[self.hand_motors, 0].copy(),
            hand_motor_kv=-self.model.actuator_biasprm[self.hand_motors, 2].copy(),
            hand_motor_torque_range_nm=self.model.actuator_forcerange[self.hand_motors].copy(),
            robot_gravity_compensation={self.model.body(i).name: float(self.model.body_gravcomp[i])
                for i in range(1, self.model.nbody) if i != self.target},
            object_mass_kg=float(self.model.body_mass[self.target]),
            object_attachment=False, runtime_joint_projection=False, mocap=False)
        write_json(self.directory / "execution_report.json", result)
        write_json(self.directory / "contact_trace.json", self.samples)
        if self.states:
            np.savez_compressed(self.directory / "execution_states.npz",
                **{k: np.stack([s[k] for s in self.states]) for k in self.states[0]})
        np.savez_compressed(self.directory / "terminal_state.npz", qpos=self.data.qpos,
                            qvel=self.data.qvel, ctrl=self.data.ctrl, time_s=self.data.time)
        return result
