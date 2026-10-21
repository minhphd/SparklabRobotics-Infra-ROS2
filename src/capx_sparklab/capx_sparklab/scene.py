"""Both arms and what cap-x's tools show, as ROS topics for Foxglove.

Takes the place of cap-x's viser SceneView: the same calls, published instead.

    /<arm>/robot_description   std_msgs/String  the arm with its gripper; link
                               names carry the arm, meshes are package:// URIs
    /tf                        every arm link, from the joint states
    /capx/perceived            sensor_msgs/PointCloud2  the last perceived object
    /capx/perceived_box        visualization_msgs/MarkerArray  its bounding box
    /capx/grasps               visualization_msgs/MarkerArray  grasp candidates, best one bright
    /capx/collision            visualization_msgs/MarkerArray  the planner's world, when shown
    /capx/<camera>/highlight   sensor_msgs/Image  the frame with the mask and grounding point

Everything 3D is in left_base. Given a viser server as ``viewer``, it also draws there
every perceived object's points and box, one per perceive prompt, and the grasps.
"""

from __future__ import annotations

import array
import xml.etree.ElementTree as ET
from typing import Any

import cv2
import numpy as np
from geometry_msgs.msg import Point, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from capx.utils.scene_view import gripper_outline
from capx.utils.tool_envelope import pose_in

from .robot import FINGER_OPEN, GRIPPER_OPENING, TOOL_FRAME, URDF

WORLD = "left_base"
LINK_HZ = 20.0
GLOW = (255, 190, 0)
BEST, OTHER = (0, 255, 128), (90, 90, 110)
CLOUD_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(("x", "y", "z", "rgb"))]


def _arm_model(arm: str) -> tuple[str, list[dict[str, Any]]]:
    """One arm's URDF with links prefixed by the arm and package:// meshes, and its joints.

    The planning frames (world, grasp_base, tool0) are left out: the driver
    publishes <arm>_tool0 itself, and a second parent for it would break TF.
    """
    root = ET.parse(URDF).getroot()
    planning = {"world", TOOL_FRAME, "tool0"}
    for link in list(root.findall("link")):
        if link.get("name") in planning:
            root.remove(link)
        else:
            link.set("name", f"{arm}_{link.get('name')}")
    joints = []
    for joint in list(root.findall("joint")):
        parent, child = joint.find("parent"), joint.find("child")
        if parent.get("link") in planning or child.get("link") in planning:
            root.remove(joint)
            continue
        parent.set("link", f"{arm}_{parent.get('link')}")
        child.set("link", f"{arm}_{child.get('link')}")
        origin, axis = joint.find("origin"), joint.find("axis")
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        joints.append({"name": joint.get("name"), "type": joint.get("type"), "origin": T,
                       "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None,
                       "parent": parent.get("link"), "child": child.get("link")})
    for mesh in root.iter("mesh"):
        mesh.set("filename", f"package://capx_sparklab/robot/{mesh.get('filename')}")
    return ET.tostring(root, encoding="unicode"), joints


def _color(rgb: tuple[int, int, int], alpha: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=rgb[0] / 255, g=rgb[1] / 255, b=rgb[2] / 255, a=alpha)


class RosScene:
    """Publishes the arms and cap-x's drawings from the env's ROS node.

    env: a YamEnv, for its node, joint states and camera frames.
    """

    live = True

    def __init__(self, env: Any) -> None:
        self._env = env
        self.viewer = None
        self._label = ""
        node = env._node
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._joints = {}
        for arm in env.arms:
            description, self._joints[arm] = _arm_model(arm)
            node.create_publisher(String, f"/{arm}/robot_description", latched).publish(String(data=description))
        self._tf = TransformBroadcaster(node)
        self._cloud = node.create_publisher(PointCloud2, "/capx/perceived", 1)
        self._box = node.create_publisher(MarkerArray, "/capx/perceived_box", 1)
        self._grasps = node.create_publisher(MarkerArray, "/capx/grasps", 1)
        self._collision = node.create_publisher(MarkerArray, "/capx/collision", 1)
        self._highlights = {c: node.create_publisher(Image, f"/capx/{c}/highlight", 1) for c in env.cameras}
        node.create_timer(1.0 / LINK_HZ, self._publish_links)

    # ---- the arms ----------------------------------------------------------------
    def _publish_links(self) -> None:
        stamp = self._env._node.get_clock().now().to_msg()
        out = []
        for arm, joints in self._joints.items():
            q, opening = self._env.joints(arm)
            finger = FINGER_OPEN * opening / GRIPPER_OPENING
            angles = {f"dof_joint{i + 1}": v for i, v in enumerate(q)}
            for joint in joints:
                T = joint["origin"].copy()
                if joint["type"] == "revolute":
                    T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(
                        joint["axis"] / np.linalg.norm(joint["axis"]) * angles[joint["name"]]).as_matrix()
                elif joint["type"] == "prismatic":
                    T[:3, 3] = T[:3, 3] + T[:3, :3] @ (joint["axis"] * finger)
                msg = TransformStamped(child_frame_id=joint["child"])
                msg.header.stamp, msg.header.frame_id = stamp, joint["parent"]
                msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = T[:3, 3]
                (msg.transform.rotation.x, msg.transform.rotation.y,
                 msg.transform.rotation.z, msg.transform.rotation.w) = Rotation.from_matrix(T[:3, :3]).as_quat()
                out.append(msg)
        self._tf.sendTransform(out)

    # ---- SceneView's calls ---------------------------------------------------------
    def show_on_camera(self, camera_id: str, mask: np.ndarray, pixel: tuple[int, int] | None,
                       label: str | None) -> None:
        """The camera's frame with the mask outlined and the grounding point marked."""
        self._env.perception_overlay = {"camera_id": camera_id, "mask": np.asarray(mask, dtype=bool),
                                        "pixel": pixel, "label": label or ""}
        # cap-x calls show_cloud next, for this prompt's object.
        self._label = label or ""
        frame = self._env._camera(camera_id)
        if frame is None or camera_id not in self._highlights:
            return
        rgb = frame["images"]["rgb"].copy()
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape == rgb.shape[:2]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, GLOW, 3)
        if pixel is not None:
            centre = (int(pixel[0]), int(pixel[1]))
            cv2.circle(rgb, centre, 9, (255, 255, 255), -1)
            cv2.circle(rgb, centre, 6, (255, 40, 40), -1)
            cv2.putText(rgb, label or "", (centre[0] + 12, centre[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        msg = Image(height=rgb.shape[0], width=rgb.shape[1], encoding="rgb8", step=rgb.shape[1] * 3)
        msg.header.stamp, msg.header.frame_id = self._env._node.get_clock().now().to_msg(), camera_id
        msg.data = array.array("B", rgb.tobytes())
        self._highlights[camera_id].publish(msg)

    def show_cloud(self, points: np.ndarray, obb: dict[str, Any]) -> None:
        """The perceived points in one colour, and their box."""
        stamp = self._env._node.get_clock().now().to_msg()
        cloud = np.empty(len(points), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        cloud["x"], cloud["y"], cloud["z"] = np.asarray(points, dtype=np.float32).T
        cloud["rgb"] = (GLOW[0] << 16) | (GLOW[1] << 8) | GLOW[2]
        msg = PointCloud2(height=1, width=len(cloud), fields=CLOUD_FIELDS, is_bigendian=False,
                          point_step=16, row_step=16 * len(cloud), is_dense=True)
        msg.header.stamp, msg.header.frame_id = stamp, WORLD
        msg.data = array.array("B", cloud.tobytes())
        self._cloud.publish(msg)

        box = self._marker("perceived", 0, Marker.CUBE, GLOW, 0.25)
        box.pose.position.x, box.pose.position.y, box.pose.position.z = (float(v) for v in obb["center"])
        (box.pose.orientation.x, box.pose.orientation.y,
         box.pose.orientation.z, box.pose.orientation.w) = (float(v) for v in obb["quaternion_xyzw"])
        box.scale.x, box.scale.y, box.scale.z = (max(float(e), 0.002) for e in obb["extent"])
        self._box.publish(MarkerArray(markers=[self._clear("perceived"), box]))
        if self.viewer is not None:
            name = "/perceived/" + (self._label.replace("/", "-") or "object")
            self.viewer.scene.add_point_cloud(f"{name}/points", points=np.asarray(points, dtype=np.float32),
                                              colors=GLOW, point_size=0.004)
            xyzw = np.asarray(obb["quaternion_xyzw"], dtype=float)
            self.viewer.scene.add_box(f"{name}/box", color=GLOW, dimensions=np.asarray(obb["extent"], dtype=float),
                                      wireframe=True, wxyz=np.roll(xyzw, 1), position=np.asarray(obb["center"]))

    def show_grasps(self, candidates: list[dict[str, Any]], width: float, reach: float, depth: float) -> None:
        """Every candidate as a gripper outline; the one the planner would take first stands out."""
        ranked = sorted(candidates, key=lambda c: (not c.get("plannable", False), -c["score"]))
        markers = [self._clear("grasps")]
        outlines = []
        for rank, candidate in enumerate(ranked):
            position, wxyz = pose_in(candidate["pose"])
            R = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
            lines = gripper_outline(candidate.get("width", width), reach, depth).reshape(-1, 3) @ R.T + position
            marker = self._marker("grasps", rank, Marker.LINE_LIST, BEST if rank == 0 else OTHER, 1.0)
            marker.scale.x = 0.005 if rank == 0 else 0.0015
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in lines]
            markers.append(marker)
            outlines.append(lines.reshape(-1, 2, 3))
        self._grasps.publish(MarkerArray(markers=markers))
        if self.viewer is not None:
            self.viewer.scene.remove_by_name("/grasps")
            if outlines:
                self.viewer.scene.add_line_segments("/grasps/best", outlines[0], BEST, line_width=4)
            if len(outlines) > 1:
                self.viewer.scene.add_line_segments("/grasps/others", np.concatenate(outlines[1:]), OTHER)

    def show_collision_world(self, surfaces: np.ndarray, spheres: np.ndarray, arm_spheres: int,
                             carrying: bool, voxel: float) -> None:
        """What the planner believes is there: the fused surface, and the arm's spheres."""
        field = self._marker("field", 0, Marker.CUBE_LIST, (120, 120, 255), 0.5)
        field.scale.x = field.scale.y = field.scale.z = voxel
        field.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in surfaces]
        markers = [self._clear("field"), self._clear("robot"), field]
        for rank, sphere in enumerate(spheres):
            colour = (255, 60, 60) if carrying and rank >= arm_spheres else (60, 200, 255)
            marker = self._marker("robot", rank, Marker.SPHERE, colour, 0.35)
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (float(v) for v in sphere[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = float(sphere[3]) * 2
            markers.append(marker)
        self._collision.publish(MarkerArray(markers=markers))

    def _marker(self, namespace: str, index: int, kind: int, rgb: tuple[int, int, int], alpha: float) -> Marker:
        marker = Marker(ns=namespace, id=index, type=kind, action=Marker.ADD, color=_color(rgb, alpha))
        marker.header.stamp = self._env._node.get_clock().now().to_msg()
        marker.header.frame_id = WORLD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _clear(namespace: str) -> Marker:
        return Marker(ns=namespace, action=Marker.DELETEALL)
