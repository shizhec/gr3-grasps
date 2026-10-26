"""Bounded 7-DOF pose IK and discretely checked joint-space approach paths."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .geometry import rotation_error, transform
from .model import arm_names


@dataclass
class MotionPlan:
    joint_names: list
    waypoints: np.ndarray
    timestamps_s: np.ndarray
    target_tcp: np.ndarray
    ik_position_error_m: float
    ik_rotation_error_rad: float
    source: str
    checked_samples: int


def make_planner(model, data, config, side="right", directory=None, target_points=None):
    backend = config["planning"].get("backend", "scipy")
    if backend == "scipy":
        planner = Planner(model, data, config["planning"], side)
        planner.metadata = {"backend": "scipy_least_squares", "pregrasp_path": "joint_interpolation"}
        return planner
    if backend == "curobo":
        from .curobo_backend import CuroboPlanner
        return CuroboPlanner(model, data, config, side, directory, target_points)
    raise ValueError(f"Unknown planning backend: {backend}")


class Planner:
    def __init__(self, model, data, config, side="right"):
        self.model, self.config = model, config
        self.side = side
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        self.names = arm_names(side)
        self.qids = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.jids = np.array([model.joint(n).id for n in self.names])
        self.limits = model.jnt_range[self.jids]
        self.site_id = model.site(f"{side}_tcp").id
        self.moving_bodies = set()
        root = model.body(f"{side}_upper_arm_pitch_link").id
        for bid in range(root, model.nbody):
            ancestor = bid
            while ancestor > 0:
                if ancestor == root:
                    self.moving_bodies.add(bid)
                    break
                ancestor = model.body_parentid[ancestor]
        self.rng = np.random.default_rng(17)

    def pose(self, q):
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)
        return transform(self.data.site_xmat[self.site_id].reshape(3, 3), self.data.site_xpos[self.site_id])

    def collisions(self, q, allowed_object=None):
        self.pose(q)
        result = []
        for contact in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 not in self.moving_bodies and b2 not in self.moving_bodies:
                continue
            if allowed_object is not None and allowed_object in (b1, b2):
                other = b2 if b1 == allowed_object else b1
                name = self.model.body(other).name
                prefix = "R_" if self.side == "right" else "L_"
                if name.startswith(prefix) and "hand_base" not in name:
                    continue  # only intended finger contacts, never palm/table
            if allowed_object is not None:
                prefix = "R_" if self.side == "right" else "L_"
                names = [self.model.body(b).name for b in (b1, b2)]
                if all(n.startswith(prefix) and "hand_base" not in n for n in names):
                    # Fingertips may touch during closure. Keep physical self
                    # collision enabled, but use the hand contact tolerance.
                    if contact.dist >= -self.config.get("finger_self_penetration_m", 0.003):
                        continue
            if contact.dist < -self.config["collision_penetration_m"]:
                result.append({"geom1": self.model.geom(contact.geom1).name,
                               "geom2": self.model.geom(contact.geom2).name,
                               "penetration_m": float(-contact.dist)})
        return result

    def ik(self, target, seed):
        def residual(q):
            pose = self.pose(q)
            return np.r_[pose[:3, 3] - target[:3, 3],
                         0.20 * rotation_error(target[:3, :3], pose[:3, :3])]
        result = least_squares(residual, np.clip(seed, self.limits[:, 0] + 1e-6, self.limits[:, 1] - 1e-6),
                               bounds=(self.limits[:, 0], self.limits[:, 1]), max_nfev=200,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        actual = self.pose(result.x)
        pos_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rot_error = np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3]))
        valid = (pos_error < self.config["position_tolerance_m"] and
                 rot_error < self.config["rotation_tolerance_rad"])
        return result.x, float(pos_error), float(rot_error), valid

    def interpolate(self, start, goal):
        count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / self.config["max_joint_step_rad"])) + 1)
        return np.linspace(start, goal, count)

    def plan(self, grasps, start):
        initial_collision = self.collisions(start)
        if initial_collision:
            raise RuntimeError(f"Initial {self.side} arm is already in collision: {initial_collision}")
        checked, failures = 0, []
        for grasp in grasps:
            target = grasp.pregrasp(self.config["pregrasp_distance_m"])
            for attempt in range(4):
                seed = start if attempt == 0 else start + self.rng.normal(0, 0.7, len(start))
                goal, pos_error, rot_error, valid = self.ik(target, seed)
                if not valid:
                    failures.append("IK unreachable")
                    continue
                path = self.interpolate(start, goal)
                collision = []
                for q in path:
                    checked += 1
                    collision = self.collisions(q)
                    if collision:
                        break
                if collision:
                    failures.append(str(collision))
                    continue
                # Linear segments obey the configured bound on each joint's speed.
                duration = max(1.0, float(np.max(np.abs(goal - start))) / self.config["max_joint_speed_rad_s"])
                return MotionPlan(self.names, path, np.linspace(0, duration, len(path)), target,
                                  pos_error, rot_error, grasp.source, checked)
        raise RuntimeError(f"No collision-free pregrasp path among {len(grasps)} proposals. Last failures: {failures[-3:]}")

    def pose_path(self, targets, start, source, allowed_object=None):
        """Follow Cartesian samples with bounded IK and check interpolated joints."""
        path = [start.copy()]
        checked = 0
        for target in targets:
            goal, pe, re, valid = self.ik(target, path[-1])
            if not valid:
                raise RuntimeError("Cartesian path has an unreachable pose")
            segment = self.interpolate(path[-1], goal)[1:]
            for q in segment:
                checked += 1
                collisions = self.collisions(q, allowed_object)
                if collisions:
                    raise RuntimeError(f"Cartesian path has a forbidden collision: {collisions[:2]}")
                path.append(q)
        path = np.asarray(path)
        distances = np.max(np.abs(np.diff(path, axis=0)), axis=1)
        times = np.r_[0, np.cumsum(np.maximum(distances / self.config["max_joint_speed_rad_s"], 0.02))]
        return MotionPlan(self.names, path, times, targets[-1], pe, re, source, checked)

    def plan_grasp(self, grasps, start, target_body, lift_distance):
        failures = []
        self.grasp_diagnostics = []
        for index, grasp in enumerate(grasps):
            try:
                pre = self.plan([grasp], start)
                final = grasp.T_world_tcp
                approach_targets = []
                for alpha in np.linspace(0, 1, 13)[1:]:
                    T = final.copy()
                    T[:3, 3] = (1 - alpha) * pre.target_tcp[:3, 3] + alpha * final[:3, 3]
                    approach_targets.append(T)
                try:
                    approach = self.pose_path(approach_targets, pre.waypoints[-1], grasp.source, target_body)
                except RuntimeError as error:
                    raise RuntimeError(f"Approach: {error}") from error
                lift_targets = []
                for height in np.linspace(0, lift_distance, 13)[1:]:
                    T = final.copy()
                    T[2, 3] += height
                    lift_targets.append(T)
                try:
                    lift = self.pose_path(lift_targets, approach.waypoints[-1], grasp.source, target_body)
                except RuntimeError as error:
                    raise RuntimeError(f"Lift: {error}") from error
                self.grasp_diagnostics.append({"candidate_index": index, "source": grasp.source,
                                               "score": grasp.score, "planned": True})
                return {"pregrasp": pre, "approach": approach, "lift": lift}
            except RuntimeError as error:
                failures.append(str(error))
                self.grasp_diagnostics.append({"candidate_index": index, "source": grasp.source,
                                               "score": grasp.score, "planned": False, "reason": str(error)})
        raise RuntimeError(f"No complete FDH-6 grasp path. Last failures: {failures[-4:]}")
