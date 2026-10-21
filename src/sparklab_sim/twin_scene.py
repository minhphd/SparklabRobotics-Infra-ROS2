"""The digital twin's stage: the calibrated rig, its three cameras, and props on the table.

The world frame is left_base, the frame every ROS consumer of the rig works in.
Call build() after SimulationApp exists; Isaac's modules only import then.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import calibration, looks, objects, paths
from .objects import Prop

# Within the 0.35 m an arm grasps at from its base (DESIGN.md#cap-x-on-the-rig).
DEFAULT_PROPS = [
    Prop("marker", "marker", (0.32, 0.10), yaw_deg=30),
    Prop("soda_can", "can_of_soda", (0.32, -0.06)),
    Prop("bowl", "bowl", (0.30, -0.27)),
    Prop("apple", "apple", (0.32, -0.48)),
    Prop("mug", "mug", (0.33, -0.63), yaw_deg=-90),
]

# ---- the arms ----------------------------------------------------------------
ARM_JOINTS = [f"dof_joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["dof_joint7", "dof_joint8"]
# The URDF's finger travel: 0 is closed (the tips meet), this is open (9.4 cm apart).
FINGER_OPEN = -0.04695
# PD gains per DOF, N m/rad (arm) and N/m (fingers). The arm's are the real motors', from
# i2rt's yam_ultra.yml.
STIFFNESS = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 2000.0, 2000.0])
DAMPING = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 100.0, 100.0])
# Drive limits, N m and N: the DM4340s' and DM4310s' peak torques. The fingers' is above
# anything their stiffness asks across the stroke, so a held object stays centred.
MAX_EFFORT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0, 100.0, 100.0])
# The fingers' friction: an estimate, not measured on the real ones.
FINGER_FRICTION = 1.5


@dataclass
class Twin:
    """Handles to what build() made. arms and cameras need world.reset() before use."""

    arms: dict          # arm -> SingleArticulation
    cameras: dict       # camera id -> isaacsim Camera
    specs: dict         # camera id -> calibration.CameraSpec
    props: dict         # prop name -> SingleRigidPrim, its body


def build(world, props: list[Prop] = DEFAULT_PROPS) -> Twin:
    """Populate world's stage with the rig. Returns handles to the arms and cameras."""
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

    stage = world.stage
    looks.room(stage)
    arms = {}
    for arm in calibration.ARMS:
        path = _add_arm(stage, arm)
        arms[arm] = SingleArticulation(path, name=f"{arm}_arm")
    specs = calibration.cameras()
    cameras = {cam_id: _add_camera(stage, spec) for cam_id, spec in specs.items()}
    bodies = {}
    for prop in props:
        path = objects.add(stage, prop, calibration.TABLE_TOP_Z)
        bodies[prop.name] = SingleRigidPrim(f"{path}/base_link", name=prop.name)
    return Twin(arms, cameras, specs, bodies)


def _add_arm(stage, arm: str) -> str:
    """Reference the rig's arm at its calibrated base. Returns its prim path."""
    from pxr import PhysxSchema

    path = f"/World/{arm}"
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(paths.require(paths.TWIN_USD)))
    objects.set_pose(prim, calibration.arm_base(arm))
    # The real drivers compensate gravity, so the twin's links feel none.
    for link in prim.GetChildren():
        if link.HasAPI("PhysicsRigidBodyAPI"):
            PhysxSchema.PhysxRigidBodyAPI.Apply(link).CreateDisableGravityAttr(True)
    for finger in ("tip_left", "tip_right"):
        _bind_grip(stage, f"{path}/{finger}")
    looks.paint_arm(stage, path)
    # Force drives, so the gains are the motors'; the importer's acceleration drives scale with mass.
    for joint in ARM_JOINTS + FINGER_JOINTS:
        kind = "linear" if joint in FINGER_JOINTS else "angular"
        stage.GetPrimAtPath(f"{path}/joints/{joint}").GetAttribute(f"drive:{kind}:physics:type").Set("force")
    for finger in FINGER_JOINTS:
        joint = stage.GetPrimAtPath(f"{path}/joints/{finger}")
        # Start open, as the real grippers rest.
        joint.GetAttribute("state:linear:physics:position").Set(FINGER_OPEN)
        joint.GetAttribute("drive:linear:physics:targetPosition").Set(FINGER_OPEN)
    return path


def _bind_grip(stage, link_path: str) -> None:
    from pxr import UsdPhysics, UsdShade

    material_path = "/World/Looks/FingerPad"
    if not stage.GetPrimAtPath(material_path):
        material = UsdShade.Material.Define(stage, material_path)
        pad = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        pad.CreateStaticFrictionAttr(FINGER_FRICTION)
        pad.CreateDynamicFrictionAttr(FINGER_FRICTION)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{link_path}/collisions"))
    binding.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def _add_camera(stage, spec: calibration.CameraSpec):
    """A camera at its calibrated pose with the real lens: top fixed, wrists riding the gripper."""
    from isaacsim.sensors.camera import Camera

    if spec.parent == "left_base":
        camera = Camera(f"/World/{spec.id}", resolution=(spec.width, spec.height))
        pose = spec.pose
        camera.set_world_pose(pose[:3, 3], calibration.quaternion_wxyz(pose), camera_axes="ros")
    else:
        arm = spec.parent.removesuffix("_tool0")
        camera = Camera(f"/World/{arm}/gripper/{spec.id}", resolution=(spec.width, spec.height))
        pose = calibration.TOOL0_IN_GRIPPER @ spec.pose
        camera.set_local_pose(pose[:3, 3], calibration.quaternion_wxyz(pose), camera_axes="ros")
    _set_lens(stage, camera.prim_path, spec)
    return camera


def _set_lens(stage, prim_path: str, spec: calibration.CameraSpec) -> None:
    """Pinhole fx, fy, cx, cy via USD's focal length, apertures and aperture offsets."""
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    aperture = 20.955  # mm; only its ratio to the focal length matters
    camera.GetFocalLengthAttr().Set(spec.fx * aperture / spec.width)
    camera.GetHorizontalApertureAttr().Set(aperture)
    camera.GetVerticalApertureAttr().Set(aperture * spec.height / spec.width * spec.fx / spec.fy)
    camera.GetHorizontalApertureOffsetAttr().Set((spec.cx - spec.width / 2) * aperture / spec.width)
    camera.GetVerticalApertureOffsetAttr().Set((spec.height / 2 - spec.cy) * aperture / spec.width)
    # The wrist cameras see their own fingers a few centimetres away.
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))


# ---- kinematics, as the real driver publishes it -----------------------------------
@functools.cache
def _joints() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each URDF joint's origin (4x4 in its parent link) and axis, by name."""
    joints = {}
    for joint in ET.parse(paths.TWIN_URDF).getroot().findall("joint"):
        origin = joint.find("origin")
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        joints[joint.get("name")] = (T, np.array([float(v) for v in joint.find("axis").get("xyz").split()]))
    return joints


def tool0(q: np.ndarray) -> np.ndarray:
    """tool0 in the arm's base frame for arm joints q, as the driver's FK gives it."""
    T = np.eye(4)
    for name, angle in zip(ARM_JOINTS, q):  # base -> link1 -> ... -> gripper, in order
        origin, axis = _joints()[name]
        spin = np.eye(4)
        spin[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
        T = T @ origin @ spin
    return T @ calibration.TOOL0_IN_GRIPPER


def open_finger(joint: str) -> np.ndarray:
    """A finger link's pose in the gripper link with the fingers open."""
    origin, axis = _joints()[joint]
    slide = np.eye(4)
    slide[:3, 3] = axis * FINGER_OPEN
    return origin @ slide
