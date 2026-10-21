"""Build the robot description cap-x needs for a YAM-Ultra with i2rt's stock gripper.

    ./scripts/capx_python.sh -m capx_sparklab.robot.build

Writes next to this module: ``yam.urdf`` (sparklab_yam's model rooted at ``world``,
with cap-x's tool frames), ``assets/``, the cuRobo robot config ``curobo_yam.yml``
(fitted collision spheres and self-collision matrix) and the GraspGenX gripper
``graspgenx/sparklab_yam/`` (``config.json``, ``coll_mesh.obj``).
Rerun after changing the gripper constants in ``capx_sparklab.robot``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

from . import FINGER_OPEN, GRASPGENX_GRIPPER, GRASPGENX_ROOT, HERE, TOOL0_DEPTH, TOOL_FRAME, URDF

SOURCE = HERE.parents[2] / "sparklab_yam" / "sparklab_yam" / "model"
# capx_python.sh exports the cap-x checkout it runs.
STOCK_GRIPPER = (Path(os.environ["CAPX"]) / "capx/third_party/GraspGenX/ext/gripper_descriptions"
                 / "gripper_descriptions/assets/x_grippers/yam_gripper/config.json")
FINGERS = ("tip_left", "tip_right")
REACHED_WITHOUT_CONTACT = {
    # Joints 2 and 3 on their 0 stops, links millimetres apart.
    "parked": np.zeros(6),
    # Fingers straight down over the table, 0.3 m up: the grasp pose cap-x uses most.
    "fingers down": np.array([-0.183, 1.824, 1.736, -1.482, 0.004, -0.183]),
}
FINGER_JOINTS = ("dof_joint7", "dof_joint8")
GRIPPER_LINKS = ["gripper", *FINGERS]


def link_poses(q: np.ndarray) -> dict[str, np.ndarray]:
    """World pose of every URDF link at arm joints q, fingers open as cuRobo locks them."""
    from scipy.spatial.transform import Rotation

    angles = {f"dof_joint{i + 1}": float(v) for i, v in enumerate(q)}
    joints = ET.parse(URDF).getroot().findall("joint")
    poses = {"world": np.eye(4)}
    while len(poses) <= len(joints):
        for joint in joints:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in poses or child in poses:
                continue
            origin = joint.find("origin")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
            T[:3, 3] = [float(v) for v in origin.get("xyz").split()]
            if joint.get("type") == "revolute":
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                spin = np.eye(4)
                spin[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angles[joint.get("name")]).as_matrix()
                T = T @ spin
            elif joint.get("type") == "prismatic":
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                T[:3, 3] += T[:3, :3] @ (axis * FINGER_OPEN)
            poses[child] = poses[parent] @ T
    return poses


def overlapping_pairs(kinematics: dict, q: np.ndarray) -> set[tuple[str, str]]:
    """Link pairs, not already ignored, whose collision spheres overlap at q."""
    poses = link_poses(q)
    spheres = [
        (link, poses[link][:3, :3] @ np.asarray(s["center"]) + poses[link][:3, 3], s["radius"])
        for link, entries in kinematics["collision_spheres"].items()
        for s in entries
    ]
    ignored = {tuple(sorted((a, b))) for a, bs in kinematics["self_collision_ignore"].items() for b in bs}
    pairs = set()
    for i, (a, ca, ra) in enumerate(spheres):
        for b, cb, rb in spheres[i + 1:]:
            pair = tuple(sorted((a, b)))
            if a != b and pair not in ignored and np.linalg.norm(ca - cb) < ra + rb:
                pairs.add(pair)
    return pairs


def write_urdf() -> None:
    root = ET.parse(SOURCE / "yam_ultra.urdf").getroot()

    def fixed(name: str, parent: str, child: str, xyz: str, rpy: str) -> None:
        joint = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)

    for name in ("world", TOOL_FRAME, "tool0"):
        ET.SubElement(root, "link", name=name)
    # Placed per arm at run time; see capx_sparklab.robot.curobo_config.
    fixed("world_joint", "world", "base", "0 0 0", "0 0 0")
    # GraspGenX's gripper frame: at the gripper origin, z out along the fingers.
    fixed(f"{TOOL_FRAME}_joint", "gripper", TOOL_FRAME, "0 0 0", f"{np.pi} 0 0")
    fixed("tool0_joint", TOOL_FRAME, "tool0", f"0 0 {TOOL0_DEPTH}", "0 0 0")
    ET.indent(root)
    URDF.write_text(ET.tostring(root, encoding="unicode") + "\n")


def write_assets() -> None:
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    for stl in sorted((SOURCE / "assets").glob("*.stl")):
        link = assets / stl.name
        link.unlink(missing_ok=True)
        link.symlink_to(Path("../../../../sparklab_yam/sparklab_yam/model/assets") / stl.name)


def write_curobo_config() -> Path:
    import torch
    from curobo._src.robot.builder.builder_robot import RobotBuilder

    # Sphere fitting and the collision-matrix sampling are both random.
    torch.manual_seed(0)
    builder = RobotBuilder(str(URDF), str(HERE), tool_frames=[TOOL_FRAME])
    # The base mesh ends at its mounting face, z = 0; unclipped spheres hang 3 cm into the table.
    builder.fit_collision_spheres(sphere_density=1.0, clip_links={"base": ("z", 0.0)})
    builder.compute_collision_matrix()
    out = HERE / "curobo_yam.yml"
    builder.save(builder.build(), str(out))

    config = yaml.safe_load(out.read_text())
    kinematics = config["kinematics"]
    # Absolute at run time (capx_sparklab.robot.curobo_config); relative here so the file travels.
    kinematics["urdf_path"] = URDF.name
    kinematics["asset_root_path"] = "."
    cspace = kinematics["cspace"]
    cspace["default_joint_position"] = [0.0, 1.2, 1.2, -0.6, 0.0, 0.0, 0.0, 0.0]
    cspace["max_acceleration"] = [5.0] * len(cspace["joint_names"])
    # Fingers held open: the widest the hand gets, so plans stay valid either way.
    kinematics["lock_joints"] = {j: FINGER_OPEN for j in FINGER_JOINTS}
    kinematics["grasp_contact_link_names"] = [*GRIPPER_LINKS, "attached_object"]
    kinematics["extra_links"] = {
        "attached_object": {
            "parent_link_name": TOOL_FRAME, "link_name": "attached_object",
            "joint_name": "attach_joint", "joint_type": "FIXED",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
        }
    }
    # cap-x's attach_box fits exactly this many spheres to a held object.
    kinematics["extra_collision_spheres"] = {"attached_object": 4}
    # The saved file aliases this list with mesh_link_names; attached_object has no mesh.
    kinematics["collision_link_names"] = [*kinematics["collision_link_names"], "attached_object"]
    kinematics["mesh_link_names"] = list(kinematics["mesh_link_names"])
    ignore = kinematics.setdefault("self_collision_ignore", {})
    for link in GRIPPER_LINKS:
        ignore.setdefault(link, []).append("attached_object")
    # The real arm reaches these without touching itself, so sphere overlaps there are
    # fitting artefacts; the fit is random, and which pairs overlap changes per build.
    for name, q in REACHED_WITHOUT_CONTACT.items():
        for a, b in sorted(overlapping_pairs(kinematics, q)):
            ignore.setdefault(a, []).append(b)
            print(f"spheres overlap {name}, not self-checked: {a} - {b}")
    kinematics.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def write_graspgenx_gripper() -> None:
    out = GRASPGENX_ROOT / GRASPGENX_GRIPPER / "config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(STOCK_GRIPPER.read_text())
    write_graspgenx_collision_mesh(out.parent / "coll_mesh.obj")


def write_graspgenx_collision_mesh(out: Path) -> None:
    """The open hand as convex hulls, one per piece, in GraspGenX's frame: z the approach, x closing."""
    from scipy.spatial.transform import Rotation

    from capx.integrations.vision.graspgenx import TO_TOOL_FRAME

    poses = link_poses(np.zeros(6))
    frame = np.eye(4)
    frame[:3, :3] = TO_TOOL_FRAME
    frame = frame @ np.linalg.inv(poses[TOOL_FRAME])
    parts = []
    for link in ET.parse(URDF).getroot().findall("link"):
        if link.get("name") not in GRIPPER_LINKS:
            continue
        origin = link.find("visual/origin")
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        mesh = trimesh.load(HERE / link.find("visual/geometry/mesh").get("filename"), force="mesh")
        # One hull over a whole multi-piece mesh could bridge into the grasp.
        for piece in mesh.split(only_watertight=False):
            hull = piece.convex_hull
            hull.apply_transform(frame @ poses[link.get("name")] @ T)
            parts.append(hull)
    trimesh.util.concatenate(parts).export(out)


def main() -> None:
    write_assets()
    write_urdf()
    print("wrote", URDF)
    write_graspgenx_gripper()
    print("wrote the GraspGenX gripper", GRASPGENX_GRIPPER)
    print("fitting collision spheres (a minute or two on the GPU)...")
    print("wrote", write_curobo_config())


if __name__ == "__main__":
    main()
