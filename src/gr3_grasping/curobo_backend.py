"""cuRobo V2 planning, with every returned path checked again in MuJoCo."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .geometry import rotation_error, transform
from .model import ROOT, attach_fdh6, resolve_meshes, values
from .planning import MotionPlan, Planner


def export_robot(model, data, config, side, directory):
    """Freeze other joints at measured angles; retain the official arm chain."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference = Planner(model, data, config["planning"], side)
    source = (ROOT / config["robot"]["urdf"]).resolve()
    root = ET.parse(source).getroot()
    resolve_meshes(root, source)
    attach_fdh6(root)
    for extension in root.findall("mujoco"):
        root.remove(extension)
    for joint in root.findall("joint"):
        name = joint.get("name")
        if joint.get("type") == "revolute" and name not in reference.names:
            angle = float(data.qpos[model.joint(name).qposadr[0]])
            origin = joint.find("origin")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            rot = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(axis * angle)
            origin.set("rpy", values(rot.as_euler("xyz")))
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "mimic", "dynamics"):
                for element in joint.findall(tag):
                    joint.remove(element)
        elif name in reference.names:
            # Simulation uses this deliberately slow bound for either backend.
            joint.find("limit").set("velocity", str(config["planning"]["max_joint_speed_rad_s"]))
    tcp = f"{side}_tcp"
    ET.SubElement(root, "link", name=tcp)
    joint = ET.SubElement(root, "joint", name=f"{tcp}_fixed", type="fixed")
    body = model.site_bodyid[reference.site_id]
    ET.SubElement(joint, "parent", link=model.body(body).name)
    ET.SubElement(joint, "child", link=tcp)
    rpy = Rotation.from_quat(model.site_quat[reference.site_id][[1, 2, 3, 0]]).as_euler("xyz")
    ET.SubElement(joint, "origin", xyz=values(model.site_pos[reference.site_id]), rpy=values(rpy))
    urdf = directory / "gr3_frozen.urdf"
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)

    # Conservative sphere covers of the same collision geometry used by MuJoCo.
    # They are a search approximation; acceptance always uses MuJoCo geometry.
    spheres, body_ids = {}, {}
    robot_links = {link.get("name") for link in root.findall("link")}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name
        if not model.geom_contype[gid] or name not in robot_links:
            continue
        center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
        pitch = .018 if name.startswith(("R_", "L_")) else (.025 if name == "torso_link" or "upper_arm_roll" in name else .05)
        counts = np.maximum(1, np.ceil(2 * half / pitch).astype(int))
        cell = 2 * half / counts
        grid = np.stack(np.meshgrid(*[center[i] - half[i] + cell[i] * (.5 + np.arange(counts[i]))
                                     for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        radius = float(np.linalg.norm(cell) / 2 + .0005)
        R_body = data.xmat[bid].reshape(3, 3)
        world = grid @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        local = (world - data.xpos[bid]) @ R_body
        spheres.setdefault(name, []).extend({"center": point.tolist(), "radius": radius} for point in local)
        body_ids[name] = bid
    ignore = {}
    names = list(spheres)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = body_ids[first], body_ids[second]
            both_fixed = a not in reference.moving_bodies and b not in reference.moving_bodies
            same_hand = first[:2] == second[:2] and first.startswith(("R_", "L_"))
            # Adjacent and second-neighbour collision proxies can overlap.
            nearby = bool({a, int(model.body_parentid[a])} & {b, int(model.body_parentid[b])})
            if both_fixed or same_hand or nearby:
                ignore.setdefault(first, []).append(second)
    robot = {"robot_cfg": {"kinematics": {
        "urdf_path": str(urdf), "asset_root_path": str(directory), "base_link": "base_link",
        "tool_frames": [tcp], "collision_link_names": names, "collision_spheres": spheres,
        "collision_sphere_buffer": 0.0, "self_collision_ignore": ignore, "self_collision_buffer": {},
        "cspace": {"joint_names": reference.names, "default_joint_position": data.qpos[reference.qids].tolist(),
                   "cspace_distance_weight": [1.] * 7, "null_space_weight": [1.] * 7,
                   "max_acceleration": 2.0, "max_jerk": 20.0},
    }}}
    (directory / "robot.yml").write_text(yaml.safe_dump(robot, sort_keys=False))
    return robot


class CuroboPlanner(Planner):
    def __init__(self, model, data, config, side="right", directory=None, target_points=None):
        super().__init__(model, data, config["planning"], side)
        try:
            import torch
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import GoalToolPose, JointState, Pose
        except ImportError as error:
            raise RuntimeError("cuRobo requested: run scripts/setup_curobo.sh and use .venv-curobo/bin/python") from error
        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo requested but CUDA PyTorch/GPU is unavailable; no silent CPU fallback")
        self.torch, self.GoalToolPose, self.JointState, self.Pose = torch, GoalToolPose, JointState, Pose
        self.tool = f"{side}_tcp"
        self.directory = Path(directory or ROOT / "outputs/curobo")
        robot = export_robot(model, data, config, side, self.directory)
        base = model.body("base_link").id
        self.T_base_world = np.linalg.inv(transform(data.xmat[base].reshape(3, 3), data.xpos[base]))
        scene = config["scene"]
        table = [*scene["table_center_xy"], scene["table_top_m"] - .025]
        table_local = self.T_base_world[:3, :3] @ table + self.T_base_world[:3, 3]
        self.world = {"cuboid": {"table": {"dims": [*(2 * np.asarray(scene["table_half_size_xy"])), .05],
                                          "pose": [*table_local.tolist(), 1, 0, 0, 0]}}}
        options = config["planning"].get("curobo", {})
        cfg = MotionPlannerCfg.create(robot=robot, scene_model=self.world, use_cuda_graph=True,
                                      position_tolerance=self.config["position_tolerance_m"],
                                      orientation_tolerance=self.config["rotation_tolerance_rad"],
                                      num_ik_seeds=options.get("ik_seeds", 32), num_trajopt_seeds=4,
                                      collision_cache={"cuboid": 8},
                                      interpolation_dt=.025, interpolation_buffer_size=2000,
                                      optimizer_collision_activation_distance=.002)
        self.gpu = MotionPlanner(cfg)
        self.gpu.warmup(enable_graph=True, num_warmup_iterations=1)
        self.order = [self.gpu.joint_names.index(n) for n in self.names]
        self.target_points = None if target_points is None else np.asarray(target_points)
        self.metadata = {"backend": "curobo_v2", "cuda_device": torch.cuda.get_device_name(),
                         "ik_seeds": options.get("ik_seeds", 32), "trajopt_seeds": 4,
                         "cuda_graph": True, "mujoCo_path_recheck": True,
                         "robot_config": str(self.directory / "robot.yml"),
                         "approach_and_lift": "Cartesian samples with cuRobo IK and MuJoCo checks",
                         "collision_sphere_count": sum(map(len, robot["robot_cfg"]["kinematics"]["collision_spheres"].values()))}
        self.verify_fk(data.qpos[self.qids])

    def goal(self, target):
        local = self.T_base_world @ target
        position = self.torch.tensor(local[None, :3, 3], device="cuda", dtype=self.torch.float32)
        quaternion = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
        pose = self.Pose(position=position, quaternion=self.torch.tensor(quaternion[None], device="cuda", dtype=self.torch.float32))
        return self.GoalToolPose.from_poses({self.tool: pose}, num_goalset=1)

    def joint_state(self, q):
        ordered = np.array([q[self.names.index(n)] for n in self.gpu.joint_names])
        return self.JointState.from_position(self.torch.tensor(ordered[None], device="cuda", dtype=self.torch.float32),
                                              joint_names=self.gpu.joint_names)

    def verify_fk(self, q):
        state = self.gpu.kinematics.compute_kinematics(self.joint_state(q))
        pose = state.tool_poses.get_link_pose(self.tool)
        p = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        gpu_pose = transform(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(), p)
        expected = self.T_base_world @ self.pose(q)
        pe = float(np.linalg.norm(expected[:3, 3] - gpu_pose[:3, 3]))
        re = float(np.linalg.norm(rotation_error(expected[:3, :3], gpu_pose[:3, :3])))
        if pe > 1e-5 or re > 1e-4:
            raise RuntimeError(f"cuRobo/MuJoCo FK mismatch: {pe} m, {re} rad")
        self.metadata.update(fk_position_difference_m=pe, fk_rotation_difference_rad=re)

    def ik(self, target, seed):
        result = self.gpu.ik_solver.solve_pose(self.goal(target), current_state=self.joint_state(seed), return_seeds=8)
        solutions = result.js_solution.position.detach().cpu().numpy().reshape(-1, len(self.names))[:, self.order]
        feasible = result.success.detach().cpu().numpy().reshape(-1)
        distance = np.linalg.norm(solutions - seed, axis=1)
        distance[~feasible] = np.inf
        q = solutions[np.argmin(distance)]
        actual = self.pose(q)
        pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        re = float(np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])))
        valid = bool(feasible.any()) and pe < self.config["position_tolerance_m"] and re < self.config["rotation_tolerance_rad"]
        return q, pe, re, valid

    def plan(self, grasps, start):
        from copy import deepcopy
        from curobo.scene import Scene

        if self.collisions(start):
            raise RuntimeError("Initial arm state collides in MuJoCo")
        failures = []
        for grasp in grasps:
            target = grasp.pregrasp(self.config["pregrasp_distance_m"])
            world = deepcopy(self.world)
            if self.target_points is not None:
                points = self.target_points @ self.T_base_world[:3, :3].T + self.T_base_world[:3, 3]
                low, high = points.min(0), points.max(0)
                world["cuboid"]["observed_target_bounds"] = {
                    "dims": np.maximum(high - low, .002).tolist(),
                    "pose": [*((high + low) / 2).tolist(), 1, 0, 0, 0]}
            try:
                self.gpu.update_world(Scene.create(world))
                result = self.gpu.plan_pose(self.goal(target), self.joint_state(start), max_attempts=3)
            finally:
                # Finger/target contact is allowed in approach and lift only;
                # MuJoCo retains table, palm, object, and self checks throughout.
                self.gpu.update_world(Scene.create(self.world))
            if result is None or not bool(result.success.any()):
                failures.append("cuRobo pregrasp IK/trajectory/graph search failed")
                continue
            trajectory = result.get_interpolated_plan()
            coarse = trajectory.position.detach().cpu().numpy().reshape(-1, len(self.names))[:, self.order]
            path = [start.copy()]
            for q in coarse:
                path.extend(self.interpolate(path[-1], q)[1:])
            path = np.asarray(path)
            collision = next((c for q in path if (c := self.collisions(q))), None)
            if collision:
                failures.append(f"MuJoCo rejected cuRobo path: {collision[:2]}")
                continue
            actual = self.pose(path[-1])
            pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            re = float(np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])))
            if pe >= self.config["position_tolerance_m"] or re >= self.config["rotation_tolerance_rad"]:
                failures.append(f"MuJoCo endpoint error {pe} m, {re} rad")
                continue
            # Preserve dense cuRobo geometry and slow it to the common bound.
            increments = np.max(np.abs(np.diff(path, axis=0)), axis=1)
            times = np.r_[0, np.cumsum(np.maximum(increments / self.config["max_joint_speed_rad_s"], .01))]
            return MotionPlan(self.names, path, times, target, pe, re, grasp.source, len(path))
        raise RuntimeError(f"No collision-free cuRobo pregrasp path: {failures[-3:]}")
