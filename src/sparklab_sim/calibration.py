"""The rig as calibrated, read from sparklab_yam/config: the files the real launch and camera node use.

Every pose is a 4x4 in its parent frame, the parent and child names being the ROS
frames: left_base, right_base, <arm>_tool0 and the camera ids (optical frames).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .paths import YAM_ULTRA_CONFIG_DIR

ARMS = ("left", "right")

# The table top sits 1.3-1.8 cm below the arm bases (calibration board, 2026-10-14).
TABLE_TOP_Z = -0.015

# tool0 in the gripper link: turned half a turn about x, 0.1347 m out along the stock fingers.
# Must match sparklab_yam.model.kinematics.TOOL0_OFFSET_XYZ/QUAT; the wrist cameras hang off it.
TOOL0_IN_GRIPPER = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0, -0.1347],
                             [0.0, 0.0, 0.0, 1.0]])


@dataclass
class CameraSpec:
    """One camera: where it hangs, and its pinhole intrinsics at the captured resolution."""

    id: str
    parent: str
    pose: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def k(self) -> list[float]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]


def matrix(translation, rotation_xyzw) -> np.ndarray:
    """4x4 from a translation and a unit quaternion (x, y, z, w)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(rotation_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def quaternion_wxyz(T: np.ndarray) -> np.ndarray:
    """The rotation of a 4x4 as a unit quaternion (w, x, y, z)."""
    return np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)


def _config(name: str) -> dict:
    return yaml.safe_load((YAM_ULTRA_CONFIG_DIR / name).read_text())


def transforms() -> dict[str, tuple[str, np.ndarray]]:
    """Every calibrated frame, as {child: (parent, pose in parent)}."""
    return {t["child"]: (t["parent"], matrix(t["translation"], t["rotation"]))
            for t in _config("extrinsics.yaml")["transforms"]}


def arm_base(arm: str) -> np.ndarray:
    """An arm's base in left_base, the twin's world frame."""
    return np.eye(4) if arm == "left" else transforms()[f"{arm}_base"][1]


def cameras() -> dict[str, CameraSpec]:
    """The cameras in cameras.yaml, placed by extrinsics.yaml, with intrinsics.yaml's colour lens."""
    frames = transforms()
    lenses = {str(c["serial_number"]): c["streams"]["color"] for c in _config("intrinsics.yaml")["cameras"].values()}
    specs = {}
    for cam_id, cam in _config("cameras.yaml")["cameras"].items():
        lens = lenses[str(cam["serial"])]
        size = (lens["resolution"]["width"], lens["resolution"]["height"])
        if size != (cam["width"], cam["height"]):
            raise ValueError(f"intrinsics.yaml has {cam_id} at {size}, cameras.yaml captures "
                             f"{cam['width']}x{cam['height']}: re-read the intrinsics")
        parent, pose = frames[cam_id]
        specs[cam_id] = CameraSpec(cam_id, parent, pose, *size,
                                   lens["focal_length"]["fx"], lens["focal_length"]["fy"],
                                   lens["principal_point"]["ppx"], lens["principal_point"]["ppy"])
    return specs


def driver_parameter(name: str):
    """A yam_driver parameter from yam.yaml, as the real drivers load it."""
    return _config("yam.yaml")["/**/yam_driver"]["ros__parameters"][name]
