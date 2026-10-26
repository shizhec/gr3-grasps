"""Import the official URDF unchanged in source, then add a tabletop MJCF scene."""
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .objects import import_rigid_object

ROOT = Path(__file__).resolve().parents[2]
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
                "wrist_yaw", "wrist_pitch", "wrist_roll")


def arm_names(side="right"):
    return [f"{side}_{suffix}_joint" for suffix in ARM_SUFFIXES]


def values(array):
    return " ".join(f"{v:.10g}" for v in array)


def resolve_meshes(root, source):
    for mesh in root.iter("mesh"):
        resolved = (source.parent / mesh.get("filename")).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mesh.set("filename", str(resolved))


def attach_fdh6(root):
    """Use the identity wrist attachment from upstream add_end_effecor.py."""
    for side, prefix in (("right", "R"), ("left", "L")):
        source = ROOT / f"third_party/Wiki-GRx-Models/Dexterous_hand/fourier_hand_6dof/urdf/fourier_{side}_hand_6dof.urdf"
        hand = ET.parse(source).getroot()
        resolve_meshes(hand, source)
        for element in hand.findall("link") + hand.findall("joint"):
            root.append(element)
        joint = ET.SubElement(root, "joint", name=f"{prefix}_hand_attachment", type="fixed")
        ET.SubElement(joint, "parent", link=f"{side}_end_effector_link")
        ET.SubElement(joint, "child", link=f"{prefix}_hand_base_link")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")


def build_scene(config, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    urdf = (ROOT / config["robot"]["urdf"]).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"Official URDF not found: {urdf}. Run scripts/fetch_assets.py")
    root = ET.parse(urdf).getroot()
    resolve_meshes(root, urdf)
    if config["robot"]["hand"] == "fdh6":
        attach_fdh6(root)
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        extension = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(meshdir="", strippath="false", fusestatic="false", discardvisual="false", balanceinertia="true")
    # The upstream URDF repeats foot collision names. Rename only the working copy.
    for i, collision in enumerate(root.iter("collision")):
        collision.set("name", f"urdf_collision_{i}")
    imported_urdf = directory / "gr3_import.urdf"
    ET.ElementTree(root).write(imported_urdf, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(imported_urdf))
    converted = directory / "gr3_import.xml"
    mujoco.mj_saveLastXML(str(converted), model)
    mjcf = ET.parse(converted).getroot()
    ET.SubElement(mjcf, "option", timestep=str(config["scene"].get("timestep_s", 0.002)), integrator="implicitfast", gravity="0 0 -9.81", cone="elliptic", iterations="100")
    visual = ET.SubElement(mjcf, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "map", znear="0.01", zfar="10")
    world = mjcf.find("worldbody")
    base = world.find("body[@name='base_link']")
    base.set("pos", f"0 0 {config['robot']['base_height_m']}")
    for body in base.iter("body"):
        body.set("gravcomp", "1")
    for i, geom in enumerate(base.iter("geom")):
        geom.set("name", f"robot_geom_{i}")
        # MuJoCo URDF importer keeps separate visual and collision geometries.
        if geom.get("contype") != "0":
            geom.attrib.update(contype="1", conaffinity="3", group="3")
            if config["robot"]["hand"] == "fdh6":
                geom.attrib.update(solref="0.004 1", solimp="0.95 0.99 0.001",
                                   friction=f"{config['hand']['friction']} 0.005 0.0001")
    # The camera uses the official camera_link, including the URDF's 15-degree tilt.
    camera = base.find(f".//body[@name='{config['camera']['body']}']")
    if camera is None:
        raise ValueError("URDF does not preserve the configured camera link")
    ET.SubElement(camera, "camera", name="native_camera", pos="0 0 0",
                  xyaxes="0 -1 0 0 0 1", fovy=str(config["camera"]["fovy_deg"]))
    for side in ("right", "left"):
        ee = base.find(f".//body[@name='{side}_end_effector_link']")
        rot = Rotation.from_euler("xyz", config["robot"]["tcp_rotation_rpy"]).as_quat()
        offset = np.array(config["robot"]["tcp_offset_m"])
        if side == "left" and config["robot"]["hand"] == "fdh6":
            offset[1] *= -1
        ET.SubElement(ee, "site", name=f"{side}_tcp", pos=values(offset),
                      quat=values(rot[[3, 0, 1, 2]]), size="0.008", rgba="0 1 0 1")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="2 1 2", dir="-1 0 -1", diffuse="0.5 0.5 0.5")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", rgba="0.20 0.24 0.27 1",
                  contype="2", conaffinity="3")
    scene = config["scene"]
    xy, half = scene["table_center_xy"], scene["table_half_size_xy"]
    ET.SubElement(world, "geom", name="table", type="box", pos=values([*xy, scene["table_top_m"] - 0.025]),
                  size=values([*half, 0.025]), rgba="0.56 0.42 0.29 1", contype="2", conaffinity="3")
    object_assets = []
    for obj in scene["objects"]:
        if obj["type"] == "mjcf":
            object_assets.append(import_rigid_object(mjcf, obj, ROOT, directory, scene["table_top_m"]))
            continue
        body = ET.SubElement(world, "body", name=obj["name"], pos=values(obj["pos"]))
        ET.SubElement(body, "freejoint", name=f"{obj['name']}_free")
        ET.SubElement(body, "geom", name=f"{obj['name']}_geom", type=obj["type"], size=values(obj["size"]),
                      rgba=values(obj["rgba"]), mass=str(obj["mass"]), friction="1 0.005 0.0001",
                      contype="2", conaffinity="3", solref="0.004 1", solimp="0.95 0.99 0.001")
    actuator = ET.SubElement(mjcf, "actuator")
    # New MuJoCo versions import URDF mimic constraints; replace once so this
    # also works with versions that did not import them.
    for old in mjcf.findall("equality"):
        mjcf.remove(old)
    equality = ET.SubElement(mjcf, "equality")
    joints = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    for joint in joints:
        name = joint.get("name")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        if mimic is not None:
            ET.SubElement(equality, "joint", joint1=name, joint2=mimic.get("joint"),
                          polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0",
                          solref="0.002 1", solimp="0.9999 0.99999 0.0001")
            continue
        kp = 350 if "shoulder" in name or "elbow" in name else 120
        force = float(limit.get("effort"))
        if name.startswith(("R_", "L_")):
            kp = config["hand"]["kp"]
            force = config["hand"]["force_limit_nm"]
        kv = config["hand"]["kv"] if name.startswith(("R_", "L_")) else 2 * np.sqrt(kp)
        ET.SubElement(actuator, "position", name=f"servo_{name}", joint=name, kp=str(kp), kv=str(kv),
                      ctrlrange=f"{limit.get('lower')} {limit.get('upper')}", forcerange=f"{-force} {force}")
    scene_path = directory / "scene.xml"
    ET.indent(mjcf)
    ET.ElementTree(mjcf).write(scene_path, encoding="utf-8", xml_declaration=True)
    provenance = {"source_urdf": str(urdf), "fixed_base": True, "gravity_compensated": True,
                  "hand": config["robot"]["hand"], "camera_intrinsics": "assumed_fovy_requires_calibration",
                  "tcp": "provisional_requires_hand_calibration", "mujoco_version": mujoco.__version__,
                  "scanned_objects": object_assets}
    if config["robot"]["hand"] == "fdh6":
        provenance.update(hand_attachment="official identity transform", hand_active_dof_per_side=6,
                          hand_mimic_joints_per_side=5, hand_dynamics="simulation gains/torque limits; upstream effort=0 is unspecified",
                          object_attachment=False, hand_parameters=config["hand"],
                          right_tcp_offset_m=config["robot"]["tcp_offset_m"],
                          left_tcp_offset_m=(np.array(config["robot"]["tcp_offset_m"]) * [1, -1, 1]).tolist())
    repo = ROOT / "third_party/Wiki-GRx-Models"
    provenance["models_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return scene_path
