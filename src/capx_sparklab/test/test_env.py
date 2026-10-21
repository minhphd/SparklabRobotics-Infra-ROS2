"""YamEnv against two simulated drivers, the calibrated static TF and synthetic cameras.

Needs cap-x's stack: ./scripts/capx_python.sh -m pytest src/capx_sparklab/test
"""

import array
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import rclpy
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from capx_sparklab.env import STOCK_TOOL0, YamEnv
from capx_sparklab.robot import GRIPPER_OPENING
from sparklab_yam.ik.decoupled_ik import DecoupledIKSolver

K = [400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0]
K_TOP = [911.0, 0.0, 645.0, 0.0, 911.0, 363.0, 0.0, 0.0, 1.0]
WRIST_D = [-0.05, 0.06, 0.0, 0.0, -0.02]
# The rig's resolutions differ per camera, which cuRobo's fusion has to be told about.
SIZES = {"top": (1280, 720, K_TOP), "left_wrist": (640, 480, K), "right_wrist": (640, 480, K)}
EXTRINSICS = yaml.safe_load(
    (Path(get_package_share_directory("sparklab_yam")) / "config" / "extrinsics.yaml").read_text())["transforms"]


def _matrix(t: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(t["rotation"]).as_matrix()
    T[:3, 3] = t["translation"]
    return T


EXTRINSIC = {t["child"]: _matrix(t) for t in EXTRINSICS}


class Rig(Node):
    """The calibrated static TF and the rig's three cameras, each looking at a wall 1 m away."""

    def __init__(self) -> None:
        super().__init__("fake_rig")
        self._static = StaticTransformBroadcaster(self)
        self._static.sendTransform([self._tf(t) for t in EXTRINSICS])
        self._pubs = {c: (self.create_publisher(Image, f"/cameras/{c}/color/image_raw", 1),
                          self.create_publisher(Image, f"/cameras/{c}/depth/image_raw", 1),
                          self.create_publisher(CameraInfo, f"/cameras/{c}/color/camera_info", 1))
                      for c in ("top", "left_wrist", "right_wrist")}
        self._rgb, self._depth = {}, {}
        for cam, (width, height, _) in SIZES.items():
            rgb = np.zeros((height, width, 3), np.uint8)
            rgb[..., 0] = np.arange(width) % 256
            self._rgb[cam] = array.array("B", rgb.tobytes())
            self._depth[cam] = array.array("B", np.full((height, width), 1000, np.uint16).tobytes())
        self.create_timer(0.1, self._publish)

    def _tf(self, t: dict) -> TransformStamped:
        msg = TransformStamped()
        msg.header.frame_id, msg.child_frame_id = t["parent"], t["child"]
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = t["translation"]
        (msg.transform.rotation.x, msg.transform.rotation.y,
         msg.transform.rotation.z, msg.transform.rotation.w) = t["rotation"]
        return msg

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for cam, (color, depth, info) in self._pubs.items():
            width, height, k = SIZES[cam]
            d = WRIST_D if cam.endswith("wrist") else [0.0] * 5
            for pub, enc, step, data in ((color, "rgb8", width * 3, self._rgb[cam]),
                                         (depth, "16UC1", width * 2, self._depth[cam])):
                msg = Image(height=height, width=width, encoding=enc, step=step)
                msg.header.stamp, msg.header.frame_id = stamp, cam
                msg.data = data
                pub.publish(msg)
            camera = CameraInfo(width=width, height=height, k=k, d=d, distortion_model="plumb_bob")
            camera.header.stamp = stamp
            info.publish(camera)


@pytest.fixture(scope="module")
def env():
    # Domain 88: nothing else of ours lives there.
    os.environ.update(ROS_DOMAIN_ID="88", ROS_LOCALHOST_ONLY="1")
    driver = Path(get_package_prefix("sparklab_yam")) / "lib" / "sparklab_yam" / "driver"
    drivers = [subprocess.Popen([str(driver), "--ros-args", "-r", f"__ns:=/{arm}", "-r", "__node:=yam_driver",
                                 "-p", "sim:=true", "-p", "park_on_exit:=false"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for arm in ("left", "right")]
    rclpy.init()
    rig = Rig()
    executor = SingleThreadedExecutor()
    executor.add_node(rig)
    stop = threading.Event()
    thread = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    try:
        yam = YamEnv()
        time.sleep(0.5)
        yield yam
        yam.close()
    finally:
        stop.set()
        thread.join()
        executor.shutdown()
        rig.destroy_node()
        rclpy.try_shutdown()
        for p in drivers:
            p.terminate()
            p.wait(timeout=10)


def _grasp_base(arm: str, q: np.ndarray) -> np.ndarray:
    pos, wxyz = DecoupledIKSolver().fk(q)
    R = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
    base = EXTRINSIC["right_base"] if arm == "right" else np.eye(4)
    return base[:3, :3] @ (pos - R[:, 2] * STOCK_TOOL0) + base[:3, 3]


def test_observation_carries_joints_tool_and_every_camera(env):
    env.active_arm = "left"
    obs = env.get_observation()
    assert obs["robot_joint_pos"].shape == (7,)
    assert 0.0 <= obs["robot_joint_pos"][6] <= GRIPPER_OPENING
    assert np.allclose(obs["robot_cartesian_pos"][:3], _grasp_base("left", np.zeros(6)), atol=1e-3)
    for cam, (width, height, _) in SIZES.items():
        assert obs[cam]["images"]["rgb"].shape == (height, width, 3)
        assert obs[cam]["images"]["depth"].dtype == np.float32


def test_camera_poses_follow_the_calibration(env):
    obs = env.get_observation()
    assert np.allclose(obs["top"]["pose_mat"], EXTRINSIC["top"], atol=1e-4)
    tool0 = env._lookup("right_tool0")
    assert np.allclose(obs["right_wrist"]["pose_mat"], tool0 @ EXTRINSIC["right_wrist"], atol=1e-3)


def test_depth_in_metres_and_wrist_images_undistorted(env):
    obs = env.get_observation()
    assert np.all(obs["top"]["images"]["depth"] == pytest.approx(1.0))
    K3 = np.array(K).reshape(3, 3)
    map_x, _ = cv2.initUndistortRectifyMap(K3, np.array(WRIST_D), None, K3, (640, 480), cv2.CV_32FC1)
    # The synthetic red channel is the source column, so each pixel shows where it was sampled from.
    red = obs["left_wrist"]["images"]["rgb"][..., 0].astype(float)
    for v, u in ((10, 10), (240, 320), (400, 100)):
        assert abs(red[v, u] - map_x[v, u] % 256) < 1.5
    assert abs(map_x[10, 10] - 10) > 2  # shifted well past the 1.5 px tolerance above


def test_trajectory_moves_only_the_addressed_arm(env):
    target = np.array([0.1, 0.3, 0.3, -0.2, 0.1, 0.0])
    env.execute_trajectory("right", np.linspace(env.joints("right")[0], target, 40))
    assert np.allclose(env.joints("right")[0], target, atol=0.01)
    assert np.allclose(env.joints("left")[0], 0.0, atol=0.01)
    env.active_arm = "right"
    obs = env.get_observation()
    assert np.allclose(obs["robot_cartesian_pos"][:3], _grasp_base("right", target), atol=2e-3)


def test_move_to_joints_blocking_ramps_the_active_arm(env):
    env.active_arm = "left"
    env.move_to_joints_blocking(np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]))
    assert np.allclose(env.joints("left")[0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.0], atol=0.01)


def test_gripper_opening_in_metres(env):
    assert env.set_gripper("left", 0.0) == pytest.approx(0.0, abs=1e-3)
    assert env.set_gripper("left", 1.0) == pytest.approx(GRIPPER_OPENING, abs=1e-3)


def test_scene_publishes_both_arms_for_foxglove(env):
    descriptions = {}
    node = env._node
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for arm in env.arms:
        node.create_subscription(String, f"/{arm}/robot_description",
                                 lambda m, a=arm: descriptions.__setitem__(a, m.data), latched)
    time.sleep(1.0)
    assert set(descriptions) == {"left", "right"}
    assert 'link name="right_tip_left"' in descriptions["right"]
    assert "package://capx_sparklab/robot/assets/tip_left.stl" in descriptions["left"]
    for frame in ("left_link3", "left_tip_left", "right_tip_right"):
        assert env._tf.can_transform("left_base", frame, Time())
    # The tool frame each driver publishes and the fingers from the scene must agree on the hand.
    gripper, tool0 = env._lookup("left_gripper"), env._lookup("left_tool0")
    assert np.linalg.norm(gripper[:3, 3] - tool0[:3, 3]) == pytest.approx(STOCK_TOOL0, abs=1e-3)


def test_scene_draws_what_the_tools_show(env):
    got = {}
    node = env._node
    node.create_subscription(PointCloud2, "/capx/perceived", lambda m: got.__setitem__("cloud", m), 1)
    node.create_subscription(MarkerArray, "/capx/perceived_box", lambda m: got.__setitem__("box", m), 1)
    node.create_subscription(MarkerArray, "/capx/grasps", lambda m: got.__setitem__("grasps", m), 1)
    node.create_subscription(Image, "/capx/top/highlight", lambda m: got.__setitem__("highlight", m), 1)
    time.sleep(0.5)
    points = np.random.default_rng(0).uniform([0.4, -0.1, 0.0], [0.45, -0.05, 0.05], (500, 3))
    env.scene.show_cloud(points, {"center": [0.42, -0.07, 0.02], "extent": [0.05, 0.05, 0.05],
                                  "quaternion_xyzw": [0, 0, 0, 1]})
    grasp = {"pose": {"position": [0.42, -0.07, 0.25], "quaternion_xyzw": [0, 1, 0, 0]}, "score": 0.9,
             "plannable": True}
    env.scene.show_grasps([grasp, {**grasp, "score": 0.5, "plannable": False}], 0.094, 0.199, 0.05)
    mask = np.zeros((720, 1280), bool)
    mask[300:400, 600:700] = True
    env.scene.show_on_camera("top", mask, (650, 350), "the box")
    time.sleep(0.5)
    assert got["cloud"].width == 500 and got["cloud"].header.frame_id == "left_base"
    assert [m.type for m in got["box"].markers][-1] == 1  # a CUBE after the clear
    assert len(got["grasps"].markers) == 3 and got["grasps"].markers[1].scale.x > got["grasps"].markers[2].scale.x
    assert (got["highlight"].width, got["highlight"].height) == (1280, 720)
