#!/usr/bin/env python3
"""Render four views of a saved GR3 Dexonomy hold state for visual review."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="run_dexonomy_gr3.py output directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        model = mujoco.MjModel.from_xml_path(str(run / "model/scene.xml"))
        data = mujoco.MjData(model)
        data.qpos[:] = np.load(run / "terminal_state.npz")["qpos"]
        mujoco.mj_forward(model, data)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        tiles = []
        with mujoco.Renderer(model, 360, 480) as renderer:
            for azimuth in (90, 135, 180, 270):  # azimuth 0 sits inside the torso
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.xipos[model.body("target").id]
                camera.distance, camera.azimuth, camera.elevation = .55, azimuth, -20
                renderer.update_scene(data, camera=camera, scene_option=options)
                tile = Image.fromarray(renderer.render())
                ImageDraw.Draw(tile).text((8, 8), f"{run.name} | azimuth {azimuth}", fill="yellow")
                tiles.append(np.asarray(tile))
        rows.append(np.concatenate(tiles, axis=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.output)


if __name__ == "__main__":
    main()
