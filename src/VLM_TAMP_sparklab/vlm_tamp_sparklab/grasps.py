"""Top-down grasps from an object's bounding box, for things resting on the table.

The fingers close across the box's narrow horizontal side and reach down over the
object's top GRASP_DEPTH, but never so low that the fingertips meet the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from capx_sparklab.robot import FINGERTIP_DEPTH, GRIPPER_OPENING, TOOL0_DEPTH

# Fingertips stay this far above the table.
TIP_CLEARANCE = 0.008
# The grasp centre sits this far below the object's top, the gripper body still 2 cm above it.
GRASP_DEPTH = 0.05
# Room left in the opening around the object.
WIDTH_MARGIN = 0.01
# Along an object longer than this, grasps also ALONG either side of its middle, where
# the 2.6 cm-wide fingers still land on it. Not on a 7 cm can: they'd close on its edge.
LONG = 0.09
ALONG = 0.025


@dataclass
class Grasp:
    """A gripper-base pose (what the planner takes) and how much it is preferred."""

    position: np.ndarray
    rotation: Rotation
    score: float


def _fingers_down(closing: np.ndarray) -> Rotation:
    """Gripper base pointing down, the fingers closing along a horizontal direction."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([closing[0], closing[1], 0.0]) / np.linalg.norm(closing[:2])
    return Rotation.from_matrix(np.column_stack([np.cross(y, z), y, z]))


def top_down(points: np.ndarray, table_z: float) -> list[Grasp] | str:
    """Grasps from above, best first, or why there are none.

    points: the object's cloud in left_base. Its bottom is taken to be the table:
        the cameras see little of an object's underside.
    table_z: height of the table top.
    """
    xy = points[:, :2]
    mean = xy.mean(axis=0)
    # Horizontal principal axes, long then short.
    _, vectors = np.linalg.eigh(np.cov((xy - mean).T))
    long_axis, short_axis = vectors[:, 1], vectors[:, 0]
    along, across = (xy - mean) @ long_axis, (xy - mean) @ short_axis
    centre = mean + long_axis * (along.min() + along.max()) / 2 + short_axis * (across.min() + across.max()) / 2
    length, width = np.ptp(along), np.ptp(across)
    if width > GRIPPER_OPENING - WIDTH_MARGIN:
        return f"{width:.3f} m across at its narrowest; the fingers open to {GRIPPER_OPENING} m"

    top = points[:, 2].max()
    lowest = table_z + TIP_CLEARANCE + (FINGERTIP_DEPTH - TOOL0_DEPTH)
    grasp_z = max(lowest, top - GRASP_DEPTH)

    closings = [(short_axis, 1.0)]
    if length <= GRIPPER_OPENING - WIDTH_MARGIN:
        closings.append((long_axis, 0.8))  # round enough to take the other way too
    offsets = (0.0, -ALONG, ALONG) if length > LONG else (0.0,)
    grasps = []
    for closing, preference in closings:
        for offset in offsets:
            grasp_centre = np.array([*(centre + long_axis * offset), grasp_z])
            for sign in (1.0, -1.0):  # the wrist turns only so far; one of the two is in reach
                rotation = _fingers_down(sign * closing)
                position = grasp_centre - rotation.apply([0.0, 0.0, TOOL0_DEPTH])
                grasps.append(Grasp(position, rotation, preference - abs(offset)))
    return sorted(grasps, key=lambda g: -g.score)
