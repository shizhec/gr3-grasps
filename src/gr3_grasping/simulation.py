import mujoco
import numpy as np

from .geometry import RGBDFrame, transform


class Simulation:
    def __init__(self, scene_path, config, render=True):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.motor_qpos = np.array([self.model.jnt_qposadr[j] for j in self.model.actuator_trnid[:, 0]])
        for name, value in config["robot"]["initial_joints"].items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.data.ctrl[:] = self.data.qpos[self.motor_qpos]
        mujoco.mj_forward(self.model, self.data)
        self.renderer = None
        self.rng = np.random.default_rng(config["camera"]["seed"])
        self.render_options = mujoco.MjvOption()
        self.render_options.geomgroup[3] = 0  # hide collision proxies
        self.render_options.sitegroup[:] = 0
        if render:
            cam = config["camera"]
            self.renderer = mujoco.Renderer(self.model, height=cam["height"], width=cam["width"])

    def settle(self, seconds=0.5):
        for _ in range(round(seconds / self.model.opt.timestep)):
            self.step()

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite state in physics simulation")

    def capture(self):
        if self.renderer is None:
            raise RuntimeError("Rendering was disabled")
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="native_camera", scene_option=self.render_options)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        cam = self.config["camera"]
        if cam["depth_noise_std_m"] > 0:
            depth += self.rng.normal(0, cam["depth_noise_std_m"], depth.shape)
        if cam["depth_dropout"] > 0:
            depth[self.rng.random(depth.shape) < cam["depth_dropout"]] = 0
        f = cam["height"] / (2 * np.tan(np.deg2rad(cam["fovy_deg"]) / 2))
        K = np.array([[f, 0, (cam["width"] - 1) / 2], [0, f, (cam["height"] - 1) / 2], [0, 0, 1]])
        cid = self.model.camera("native_camera").id
        # MuJoCo camera: x right, y up, -z forward -> OpenCV optical convention.
        T = transform(self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]), self.data.cam_xpos[cid])
        return RGBDFrame(rgb, depth.astype(np.float32), K, T, float(self.data.time), "mujoco_native_mount")

    def overview(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.30, 0, 0.95]
        camera.distance, camera.azimuth, camera.elevation = 2.35, 135, -18
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
        return self.renderer.render().copy()

    def grasp_view(self, target_body, stage):
        from PIL import Image, ImageDraw
        overview = self.overview()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[target_body]
        camera.distance, camera.azimuth, camera.elevation = 0.65, 145, -20
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
        closeup = self.renderer.render().copy()
        panel = Image.fromarray(np.concatenate([overview, closeup], axis=1))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 25), fill="black")
        draw.text((10, 6), f"GR3 / FDH-6 | {stage} | simulation t={self.data.time:.2f}s", fill="white")
        return np.asarray(panel)

    def grasp_candidates_view(self, poses):
        """Close-up of source-model contact frames, before hand transfer."""
        from PIL import Image, ImageDraw
        poses = np.asarray(poses[:3])
        camera = mujoco.MjvCamera()
        camera.lookat[:] = np.median(poses[:, :3, 3], axis=0)
        camera.distance, camera.azimuth, camera.elevation = 0.42, 130, -25
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_options)
        scene = self.renderer.scene
        for pose in poses:
            for axis, color in enumerate(([1, .15, .15, 1], [.15, 1, .15, 1], [.2, .5, 1, 1])):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=np.float32))
                # Draw in free space behind the contact center; otherwise
                # axes inside the opaque scanned object are invisible.
                start = pose[:3, 3] - .08 * pose[:3, 2]
                length = .08 if axis == 2 else .025
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .001, start, start + length * pose[:3, axis])
                scene.ngeom += 1
        panel = Image.fromarray(self.renderer.render().copy())
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 25), fill="black")
        draw.text((8, 7), "M2T2 | blue -> contact center | red: closing axis (8 cm back)", fill="white")
        return np.asarray(panel)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
