"""Head RGB-D capture over ROS 2: realsense2_camera topics plus TF from robot_state_publisher.

Subscriptions only (rosout and parameter services disabled; rclpy's /parameter_events publisher
stays idle). The camera extrinsic is the live TF chain base_frame <- camera_frame (URDF mount and
measured head/waist joints) times a measured correction; it is not a calibration until verified.
"""
from collections import deque
import threading
import time

import numpy as np

COLOR_ENCODINGS = {"rgb8": False, "bgr8": True}


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec*1e-9


def decode_color(message):
    if message.encoding not in COLOR_ENCODINGS:
        raise ValueError(f"Unsupported color encoding {message.encoding}")
    rows = np.frombuffer(bytes(message.data), np.uint8).reshape(message.height, message.step)
    rgb = rows[:, :message.width*3].reshape(message.height, message.width, 3)
    return np.ascontiguousarray(rgb[..., ::-1] if COLOR_ENCODINGS[message.encoding] else rgb)


def decode_depth(message, depth_scale_m):
    """Metric depth in metres; zero marks missing depth."""
    if message.encoding in ("16UC1", "mono16"):
        dtype, scale = np.dtype(np.uint16), depth_scale_m
    elif message.encoding == "32FC1":
        dtype, scale = np.dtype(np.float32), 1.
    else:
        raise ValueError(f"Unsupported depth encoding {message.encoding}")
    dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    rows = np.frombuffer(bytes(message.data), np.uint8).reshape(message.height, message.step)
    depth = rows[:, :message.width*dtype.itemsize].copy().view(dtype).reshape(message.height, message.width)
    depth = depth.astype(np.float32)*np.float32(scale)
    depth[~np.isfinite(depth)] = 0
    return depth


def transform_matrix(transform):
    """geometry_msgs/Transform to a 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    q, t = transform.rotation, transform.translation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def transform_change(A, B):
    """(translation m, rotation rad) between two poses."""
    from scipy.spatial.transform import Rotation
    delta = np.linalg.inv(A) @ B
    return float(np.linalg.norm(delta[:3, 3])), float(Rotation.from_matrix(delta[:3, :3]).magnitude())


def matched_pair(colors, depths, tolerance_s):
    """Newest (color, depth) messages whose stamps agree within tolerance."""
    for color in reversed(colors):
        for depth in reversed(depths):
            if abs(stamp_s(color.header)-stamp_s(depth.header)) <= tolerance_s:
                return color, depth
    return None


class RosCamera:
    def __init__(self, config):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        self.config, self.lock, self.info = config, threading.Lock(), None
        self.colors, self.depths = deque(maxlen=10), deque(maxlen=10)
        self.correction = np.asarray(config["extrinsic_correction"], float)
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = rclpy.create_node("gr3_grasping_camera", enable_rosout=False, start_parameter_services=False)
        self.node.create_subscription(Image, config["color_topic"], lambda m: self._keep(self.colors, m), qos_profile_sensor_data)
        self.node.create_subscription(Image, config["depth_topic"], lambda m: self._keep(self.depths, m), qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, config["info_topic"], self._camera_info, qos_profile_sensor_data)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.node, spin_thread=False)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _keep(self, queue, message):
        with self.lock:
            queue.append(message)

    def _camera_info(self, message):
        self.info = message

    def base_camera(self, stamp=None):
        """T_base_camera from TF at stamp (seconds; None for latest) times the measured correction."""
        from rclpy.duration import Duration
        from rclpy.time import Time
        when = Time() if stamp is None else Time(seconds=stamp)
        transform = self.tf.lookup_transform(self.config["base_frame"], self.config["camera_frame"], when,
                                             timeout=Duration(seconds=self.config["timeout_s"]))
        return transform_matrix(transform.transform) @ self.correction

    def capture(self, T_world_base):
        """One synchronized aligned RGB-D frame as an RGBDFrame record plus capture metadata."""
        from gr3_grasping.geometry import RGBDFrame
        deadline = time.monotonic() + self.config["timeout_s"]
        while True:
            with self.lock:
                pair = matched_pair(list(self.colors), list(self.depths), self.config["sync_tolerance_s"])
            if pair is not None and self.info is not None:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("No synchronized color/aligned-depth pair with camera_info")
            time.sleep(.01)
        color, depth = pair
        stamp = stamp_s(color.header)
        if abs(time.time()-stamp) > self.config["max_image_age_s"]:
            raise RuntimeError("Head camera image is stale or the camera clock is not synchronized")
        if color.header.frame_id != self.config["camera_frame"] or depth.header.frame_id != self.config["camera_frame"]:
            raise RuntimeError("Depth is not aligned to the configured color optical frame")
        if (self.info.width, self.info.height) != (color.width, color.height) or self.info.distortion_model not in ("plumb_bob", "rational_polynomial") \
                or np.any(abs(np.asarray(self.info.d)) > 1e-9):
            raise RuntimeError("camera_info does not describe an undistorted image of this size")
        T_base_camera = self.base_camera(stamp)
        frame = RGBDFrame(rgb=decode_color(color), depth_m=decode_depth(depth, self.config["depth_scale_m"]),
            K=np.asarray(self.info.k, float).reshape(3, 3), T_world_camera=np.asarray(T_world_base) @ T_base_camera,
            timestamp_s=stamp, source="ros_head_camera")
        return frame, dict(stamp_s=stamp, T_base_camera=T_base_camera, camera_frame=self.config["camera_frame"],
                           base_frame=self.config["base_frame"], depth_valid_ratio=float((frame.depth_m > 0).mean()))

    def close(self):
        import rclpy
        self.executor.shutdown()
        self.node.destroy_node()
        if self.owns_context and rclpy.ok():
            rclpy.shutdown()
        self.thread.join(2)
