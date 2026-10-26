#!/usr/bin/env python3
"""Two-process TCP smoke test. Robot is always mock, even with real cuRobo inference."""
import argparse
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

from gr3_grasping.deployment.protocol import COMMAND_NAMES, VERSION, load, save
from gr3_grasping.deployment.transport import Client


def synthetic_observation():
    """Small, explicitly artificial input for transport checks on a fresh clone."""
    positions = dict(zip(COMMAND_NAMES, [-.1,0,0,-1.7,0,0,0,-1.4,.1,-.1,-.1,-.1,-.1]))
    positions.update(head_yaw_joint=0., head_pitch_joint=.14)
    return dict(version=VERSION, observation_id=uuid.uuid4().hex, object_id="coffee_mug",
        source="synthetic_transport_demo", joint_positions_rad=positions,
        T_world_base=np.eye(4), table_top_m=1., table_center_xy=[.6,0],
        table_half_size_xy=[.3,.4], calibration_verified=False, state_mapping_verified=False,
        frames=[dict(rgb=np.zeros((48,64,3),np.uint8), depth_m=np.ones((48,64),np.float32),
            K=np.array([[100.,0,32],[0,100.,24],[0,0,1]]), T_world_camera=np.eye(4),
            timestamp_s=0., source="synthetic_transport_demo")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation",type=Path,
        help="Saved RGB-D/state; optional only for the synthetic transport demo")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--planner",choices=["transport-demo","dexonomy"],default="transport-demo")
    args = parser.parse_args()
    if args.planner == "dexonomy" and args.observation is None:
        parser.error("--planner dexonomy requires --observation with saved RGB-D/state")
    args.output.mkdir(parents=True,exist_ok=False)
    obs = load(args.observation) if args.observation else synthetic_observation()
    obs["observation_id"] = uuid.uuid4().hex
    if args.observation:
        obs["source"] = "smoke_replay"
    obs["calibration_verified"] = False
    save(args.output/"observation.msgpack",obs)
    token = secrets.token_urlsafe(32)
    environment = {**os.environ,"GR3_DEPLOY_TOKEN":token}
    with socket.socket() as port:
        port.bind(("127.0.0.1",0))
        endpoint=f"tcp://127.0.0.1:{port.getsockname()[1]}"
    command = [sys.executable,"-m","gr3_grasping.deployment"]
    with (args.output/"server.log").open("w") as log:
        server = subprocess.Popen(command+["server","--bind",endpoint,"--planner",args.planner,
            "--output",str(args.output/"server")],stdout=log,stderr=subprocess.STDOUT,env=environment)
        try:
            deadline = time.monotonic()+30
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server exited; inspect server.log")
                try:
                    Client(endpoint,token,.2).call("health")
                    break
                except TimeoutError:
                    if time.monotonic()>deadline:
                        raise
                    time.sleep(.1)
            result = subprocess.run(command+["client","--endpoint",endpoint,"--observation",str(args.output/"observation.msgpack"),
                "--output",str(args.output/"client"),"--robot","mock","--execute","--time-scale","0.05"],
                env=environment,timeout=360)
            return result.returncode
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
