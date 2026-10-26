"""Compose rigid scanned-object MJCFs without discarding textures or convex parts."""
from copy import deepcopy
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _values(x):
    return " ".join(f"{v:.10g}" for v in x)


def import_rigid_object(scene, spec, root, directory, table_top):
    """Import one rigid, jointless body; reject articulated scenes explicitly.

    Meshes retain their source metric scale. Mass is a declared simulation
    assumption; the source's inferred inertia is scaled to that mass.
    """
    source = (Path(root) / spec["mjcf"]).resolve()
    original = ET.parse(source).getroot()
    compiler = original.find("compiler")
    dirs = {} if compiler is None else compiler.attrib
    model = mujoco.MjModel.from_xml_path(str(source))
    if model.nbody != 2 or model.njnt or model.ngeom == 0:
        raise ValueError(f"Expected one jointless rigid scanned object: {source}")
    canonical = Path(directory) / f"{spec['name']}_source.xml"
    mujoco.mj_saveLastXML(str(canonical), model)
    obj = ET.parse(canonical).getroot()
    if obj.find("default") is not None:
        raise ValueError("Scanned-object defaults must be expanded before composition")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Bounds use actual mesh vertices, including MuJoCo's mesh preprocessing
    # transforms; mesh origins are not necessarily object centers.
    vertices = []
    for gid in range(model.ngeom):
        mid = model.geom_dataid[gid]
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Scanned-object importer currently requires mesh geoms")
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        world = model.mesh_vert[start:start + count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        vertices.append((world - data.xpos[1]) @ data.xmat[1].reshape(3, 3))
    vertices = np.concatenate(vertices)
    rotation = Rotation.from_euler("xyz", spec.get("rotation_rpy_deg", [0, 0, 0]), degrees=True)
    rotated = vertices @ rotation.as_matrix().T
    low, high = rotated.min(0), rotated.max(0)
    xy = np.asarray(spec["placement_xy"], dtype=float) - (low[:2] + high[:2]) / 2
    pos = [*xy, table_top + spec.get("clearance_m", 0.001) - low[2]]
    prefix = spec["name"] + "__"
    assets = obj.find("asset")
    names = {asset.get("name"): prefix + asset.get("name") for asset in assets}
    target_assets = scene.find("asset")
    if target_assets is None:
        target_assets = ET.SubElement(scene, "asset")
    for asset in assets:
        asset = deepcopy(asset)
        for element in asset.iter():
            for attr in ("name", "mesh", "material", "texture"):
                if element.get(attr) in names:
                    element.set(attr, names[element.get(attr)])
            if element.get("file"):
                asset_dir = dirs.get("texturedir" if element.tag == "texture" else "meshdir", "")
                path = (source.parent / asset_dir / element.get("file")).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                element.set("file", str(path))
        target_assets.append(asset)
    body = deepcopy(obj.find("worldbody/body"))
    if any(child.tag not in ("geom", "inertial") for child in body):
        raise ValueError("Unsupported contents in scanned object's rigid body")
    body.attrib.clear()
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    body.attrib.update(name=spec["name"], pos=_values(pos), quat=_values(quat))
    for old in body.findall("inertial"):
        body.remove(old)
    mass = float(spec["mass"])
    if not np.isfinite(mass) or mass <= 0 or model.body_mass[1] <= 0:
        raise ValueError("Object requires a positive finite mass in kg")
    ET.SubElement(body, "inertial", pos=_values(model.body_ipos[1]), quat=_values(model.body_iquat[1]),
                  mass=str(mass), diaginertia=_values(model.body_inertia[1] * mass / model.body_mass[1]))
    ET.SubElement(body, "freejoint", name=spec["name"] + "_free")
    collision_count = 0
    for index, geom in enumerate(body.findall("geom")):
        geom.set("name", prefix + f"geom_{index}")
        for attr in ("mesh", "material"):
            if geom.get(attr) in names:
                geom.set(attr, names[geom.get(attr)])
        if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
            collision_count += 1
            geom.attrib.update(contype="2", conaffinity="3", group="3", solref="0.004 1",
                               solimp="0.95 0.99 0.001", friction=f"{spec.get('friction', 1.0)} 0.005 0.0001")
    scene.find("worldbody").append(body)
    return {"name": spec["name"], "source_mjcf": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dimensions_m": (high - low).tolist(), "mass_kg": mass,
            "mass_source": "simulation assumption, not measured", "scale": 1.0,
            "convex_collision_parts": collision_count, "free_body": True}
