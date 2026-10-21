"""What the robot knows about the objects on the table: one tracked point cloud per name.

An object's points live in the robot API's cloud store, under its cloud id, so
grasping and attaching use exactly the points the scene graph reports. This module
cleans them, moves them with the gripper while held, and relates objects to each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

# DBSCAN radius: points closer than this belong to one blob.
BLOB_RADIUS = 0.012
# An object rests on another when its bottom is within this of the other's top.
ON_TOLERANCE = 0.03


@dataclass
class TrackedObject:
    """One named object. Its points are clouds[cloud_id]["points"] in the robot API."""

    name: str
    prompt: str
    cloud_id: str
    held_by: str | None = None
    seen_at: float = field(default_factory=time.monotonic)


def largest_blob(points: np.ndarray) -> np.ndarray:
    """The biggest connected cluster: drops the fins and specks a mask edge deprojects to."""
    if len(points) < 20:
        return points
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(eps=BLOB_RADIUS, min_points=8))
    if (labels < 0).all():
        return points
    return points[labels == np.bincount(labels[labels >= 0]).argmax()]


def box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds (low, high) in left_base."""
    return points.min(axis=0), points.max(axis=0)


class SceneGraph:
    """Named objects, their clouds and who holds them.

    clouds: the robot API's cloud store, {cloud_id: {"points": (n, 3), ...}}.
    """

    def __init__(self, clouds: dict) -> None:
        self._clouds = clouds
        self.objects: dict[str, TrackedObject] = {}

    def points(self, name: str) -> np.ndarray:
        return self._clouds[self.objects[name].cloud_id]["points"]

    def add(self, name: str, prompt: str, cloud_id: str) -> TrackedObject:
        """Track a freshly perceived cloud under name, keeping only its largest blob."""
        self._clouds[cloud_id]["points"] = largest_blob(self._clouds[cloud_id]["points"])
        held_by = self.objects[name].held_by if name in self.objects else None
        self.objects[name] = TrackedObject(name, prompt, cloud_id, held_by)
        return self.objects[name]

    def move(self, name: str, motion: np.ndarray) -> None:
        """Carry an object's points by a rigid motion (4x4), as the gripper holding it moved."""
        points = self.points(name)
        self._clouds[self.objects[name].cloud_id]["points"] = points @ motion[:3, :3].T + motion[:3, 3]

    def held_by(self, arm: str) -> str | None:
        return next((o.name for o in self.objects.values() if o.held_by == arm), None)

    def resting_on(self, name: str) -> str | None:
        """The object this one sits on or in: its centre over the other's footprint, its bottom near the other's top."""
        low, high = box(self.points(name))
        centre = (low + high) / 2
        for other in self.objects:
            if other == name or self.objects[other].held_by:
                continue
            other_low, other_high = box(self.points(other))
            over = np.all((centre[:2] > other_low[:2]) & (centre[:2] < other_high[:2]))
            if over and other_low[2] - 0.01 < low[2] < other_high[2] + ON_TOLERANCE:
                return other
        return None

    def describe(self) -> list[dict]:
        """Each object's position (box centre), size, top height, holder and support, in metres."""
        now = time.monotonic()
        rows = []
        for name, obj in self.objects.items():
            low, high = box(self.points(name))
            rows.append({"name": name, "position": np.round((low + high) / 2, 3).tolist(),
                         "size": np.round(high - low, 3).tolist(), "top_z": round(float(high[2]), 3),
                         "held_by": obj.held_by, "on": None if obj.held_by else self.resting_on(name),
                         "seen_s_ago": round(now - obj.seen_at, 1)})
        return rows
