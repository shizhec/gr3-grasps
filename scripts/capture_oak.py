#!/usr/bin/env python3
"""Optional direct OAK capture, only for an accessible compatible stereo OAK.

This uses DepthAI 2.x, not an invented GR-Sensor camera API. Do not run alongside
the vendor service owning the same USB device. Extrinsics must correspond to
the head/base pose at capture time. This program never opens a motor interface.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gr3_grasping.geometry import RGBDFrame, validate_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extrinsics", type=Path, required=True,
                        help="JSON with T_world_camera, calibrated=true; optical frame, meters")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=20)
    args = parser.parse_args()
    import depthai as dai

    calibration = json.loads(args.extrinsics.read_text())
    if calibration.get("calibrated") is not True:
        raise ValueError("A measured camera-to-world calibration is required")
    T = validate_transform(calibration["T_world_camera"])
    pipeline = dai.Pipeline()
    color = pipeline.create(dai.node.Camera)
    color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    color.setSize(640, 360)
    color.setFps(15)
    # As in Luxonis' v2.30 rgb_depth_aligned example: rectify the RGB image
    # using the device calibration so that it shares the aligned depth geometry.
    color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
    left, right = pipeline.create(dai.node.MonoCamera), pipeline.create(dai.node.MonoCamera)
    for camera, socket in [(left, dai.CameraBoardSocket.CAM_B), (right, dai.CameraBoardSocket.CAM_C)]:
        camera.setBoardSocket(socket)
        camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        camera.setFps(15)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 360)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    for name, stream in [("rgb", color.video), ("depth", stereo.depth)]:
        out = pipeline.create(dai.node.XLinkOut)
        out.setStreamName(name)
        stream.link(out.input)
    with dai.Device() as device:
        device_calibration = device.readCalibration2()
        focus = device_calibration.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if focus:
            color.initialControl.setManualFocus(focus)
        device.startPipeline(pipeline)
        rgb_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)
        K = np.array(device_calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 360))
        deadline = time.monotonic() + args.timeout_s
        rgb_msg = depth_msg = None
        while time.monotonic() < deadline:
            rgb_msg = rgb_q.tryGet() or rgb_msg
            depth_msg = depth_q.tryGet() or depth_msg
            if rgb_msg is not None and depth_msg is not None:
                delta = abs((rgb_msg.getTimestamp() - depth_msg.getTimestamp()).total_seconds())
                if delta <= 0.025:
                    rgb = rgb_msg.getCvFrame()[:, :, ::-1].copy()
                    depth = depth_msg.getFrame().astype(np.float32) * 0.001
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    RGBDFrame(rgb, depth, K, T, rgb_msg.getTimestamp().total_seconds(),
                              "oak_depthai2_device_monotonic").save(args.output)
                    print(args.output)
                    return
            time.sleep(0.005)
        raise TimeoutError("No synchronized RGB/depth pair before deadline")


if __name__ == "__main__":
    main()
