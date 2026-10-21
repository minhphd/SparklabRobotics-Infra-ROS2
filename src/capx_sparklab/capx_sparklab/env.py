"""cap-x's low-level env on the YAM rig, over ROS 2.

One ROS node inside the cap-x process reads joint states, camera frames and TF
from the running rig (``ros2 launch sparklab_yam yam.launch.py``) and streams
joint targets to the drivers. The world frame is ``left_base``, the root of the
rig's TF tree. The observation is for the arm in ``active_arm``:

    robot_joint_pos      6 joints (rad), then the gripper opening (m)
    robot_cartesian_pos  grasp_base in world: x, y, z, qw, qx, qy, qz
    <camera_id>          {images: {rgb, depth (m)}, intrinsics, pose_mat (camera
                         to world), pose (x, y, z, qw, qx, qy, qz)}

Wrist images arrive undistorted, so cap-x's pinhole deprojection holds for them.
Both arms, and what cap-x's tools draw, go out on ROS for Foxglove (``scene``).
Both arms park when the process exits, Ctrl-C included.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from capx.envs.base import BaseEnv

from .robot import GRIPPER_OPENING
from .scene import RosScene

WORLD = "left_base"
ARMS = ("left", "right")
CAMERAS = ("top", "left_wrist", "right_wrist")
# sparklab_yam's tool0, which the drivers publish on TF, sits this far along grasp_base z.
STOCK_TOOL0 = 0.1347
# cuRobo's interpolation_dt default, which cap-x's Planner keeps.
TRAJECTORY_DT = 0.025
MAX_LAG = 0.15
"""rad between the streamed target and the measured joint before a move is stopped."""
SETTLE_TOL = 0.01
SETTLE_TIMEOUT = 2.0
RAMP_SPEED = 0.4
"""rad/s for single-target moves, which have no planner timing."""
GRIPPER_STILL = 0.005
GRIPPER_TIMEOUT = 3.0
PARK_TIMEOUT = 10.0
"""s to wait on the drivers' park, which blocks for its 5 s ramp."""


class YamEnv(BaseEnv):
    """The YAM rig as a cap-x low-level env.

    cameras: camera ids to observe, as in sparklab_yam's cameras.yaml.
    arms: arm namespaces; the first is active at start.
    viser_debug: accepted for cap-x's ReAct loop, which sets it; the scene is on
        ROS either way and there is no viser server.
    """

    def __init__(self, cameras: tuple[str, ...] = CAMERAS, arms: tuple[str, ...] = ARMS,
                 viser_debug: bool = False) -> None:
        if not rclpy.ok():
            # rclpy's own handler would shut ROS down on Ctrl-C before close() could park.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.arms = tuple(arms)
        self.cameras = tuple(cameras)
        self.active_arm = self.arms[0]
        self.viser_server = None
        self.perception_overlay: dict[str, Any] | None = None

        node = self._node = Node("capx_yam_env")
        self._joints: dict[str, np.ndarray] = {}
        self._color: dict[str, dict[Any, Image]] = {c: {} for c in self.cameras}
        self._depth: dict[str, dict[Any, Image]] = {c: {} for c in self.cameras}
        self._info: dict[str, CameraInfo] = {}
        self._undistort: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        for arm in self.arms:
            node.create_subscription(JointState, f"/{arm}/joint_states",
                                     lambda m, a=arm: self._joints.__setitem__(a, np.array(m.position)), 10)
        for cam in self.cameras:
            node.create_subscription(Image, f"/cameras/{cam}/color/image_raw",
                                     lambda m, c=cam: self._keep(self._color[c], m), 3)
            node.create_subscription(Image, f"/cameras/{cam}/depth/image_raw",
                                     lambda m, c=cam: self._keep(self._depth[c], m), 3)
            node.create_subscription(CameraInfo, f"/cameras/{cam}/color/camera_info",
                                     lambda m, c=cam: self._info.__setitem__(c, m), 1)
        self._arm_pubs = {a: node.create_publisher(Float64MultiArray, f"/{a}/arm/command", 10) for a in self.arms}
        self._gripper_pubs = {a: node.create_publisher(Float64, f"/{a}/gripper/command", 10) for a in self.arms}
        self._park_clients = {a: node.create_client(Trigger, f"/{a}/park") for a in self.arms}
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._stop = threading.Event()
        self._spinner = threading.Thread(target=self._spin, daemon=True)
        self._spinner.start()
        atexit.register(self.close)
        self._wait(lambda: all(a in self._joints for a in self.arms), 5.0,
                   f"no joint_states from {[a for a in self.arms if a not in self._joints]} — is the rig up?")
        frames = [*(f"{a}_tool0" for a in self.arms), *self.cameras]
        self._wait(lambda: all(self._tf.can_transform(WORLD, f, Time()) for f in frames), 5.0,
                   f"no TF from {WORLD} to {[f for f in frames if not self._tf.can_transform(WORLD, f, Time())]}"
                   " — is the launch (static extrinsics) up?")
        self._wait(lambda: all(self._camera(c) is not None for c in self.cameras), 5.0,
                   f"no frames from {[c for c in self.cameras if self._camera(c) is None]} — is the camera node up?")
        self.viser_debug = False
        self.scene = RosScene(self)

    def _spin(self) -> None:
        while not self._stop.is_set():
            # A failing callback must not stop the joint states the motion checks read.
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception as exc:
                self._node.get_logger().error(f"callback failed: {type(exc).__name__}: {exc}",
                                              throttle_duration_sec=5.0)

    # ---- gym / cap-x interface -------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.get_observation(), {}

    def step(self, action: Any):
        return self.get_observation(), 0.0, False, False, {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = self._camera(self.cameras[0])
        return frame["images"]["rgb"] if frame else np.zeros((480, 640, 3), np.uint8)

    def get_observation(self) -> dict[str, Any]:
        q, gripper = self.joints(self.active_arm)
        obs: dict[str, Any] = {
            "robot_joint_pos": np.append(q, gripper),
            "robot_cartesian_pos": self.tool_pose(self.active_arm),
        }
        for cam in self.cameras:
            frame = self._camera(cam)
            if frame is not None:
                obs[cam] = frame
        return obs

    def move_to_joints_blocking(self, joints: np.ndarray) -> None:
        """Ramp the active arm to a joint target at RAMP_SPEED, then settle."""
        start, _ = self.joints(self.active_arm)
        goal = np.asarray(joints, dtype=float)[:6]
        steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / RAMP_SPEED / TRAJECTORY_DT)))
        self.execute_trajectory(self.active_arm, start + np.outer(np.arange(1, steps + 1) / steps, goal - start))

    def _set_gripper(self, fraction: float) -> None:
        self.set_gripper(self.active_arm, fraction)

    def _step_once(self) -> None:
        time.sleep(TRAJECTORY_DT)

    def close(self) -> None:
        """Park both arms, then leave ROS. Runs at exit; later calls do nothing."""
        if self._stop.is_set():
            return
        parks = [self._park_clients[a].call_async(Trigger.Request()) for a in self.arms]
        try:
            self._wait(lambda: all(p.done() for p in parks), PARK_TIMEOUT, "the drivers did not confirm the park")
        finally:
            self._stop.set()
            self._spinner.join()
            self._executor.shutdown()
            self._node.destroy_node()

    # ---- state ---------------------------------------------------------------
    def joints(self, arm: str) -> tuple[np.ndarray, float]:
        """Arm joints (6,) in rad and the gripper opening in m."""
        position = self._joints[arm]
        # The driver's gripper reads 1 open, 0 closed (seen on the rig).
        return position[:6].copy(), float(np.clip(position[6], 0.0, 1.0) * GRIPPER_OPENING)

    def base_pose(self, arm: str) -> np.ndarray:
        """The arm's base in world (4x4), from the rig's calibrated TF."""
        return np.eye(4) if f"{arm}_base" == WORLD else self._lookup(f"{arm}_base")

    def tool_pose(self, arm: str) -> np.ndarray:
        """grasp_base in world as x, y, z, qw, qx, qy, qz."""
        T = self._lookup(f"{arm}_tool0")
        position = T[:3, 3] - T[:3, 2] * STOCK_TOOL0
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        return np.array([*position, w, x, y, z])

    # ---- commands ------------------------------------------------------------
    def execute_trajectory(self, arm: str, waypoints: np.ndarray, dt: float = TRAJECTORY_DT) -> None:
        """Stream joint waypoints to one arm, then block until it settles on the last.

        Raises RuntimeError, with the arm held where it is, if it falls MAX_LAG
        behind the stream: blocked, in zero gravity, or its driver gone.
        """
        waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 6)
        tick = time.monotonic()
        for target in waypoints:
            self._arm_pubs[arm].publish(Float64MultiArray(data=target.tolist()))
            lag = np.abs(self.joints(arm)[0] - target)
            if lag.max() > MAX_LAG:
                self.hold(arm)
                raise RuntimeError(f"{arm} arm lags its trajectory by {lag.max():.2f} rad at joint {lag.argmax() + 1} "
                                   "and was stopped: blocked, in zero gravity, or its driver is down")
            tick += dt
            time.sleep(max(0.0, tick - time.monotonic()))
        self._wait(lambda: np.max(np.abs(self.joints(arm)[0] - waypoints[-1])) < SETTLE_TOL,
                   SETTLE_TIMEOUT, None)

    def hold(self, arm: str) -> None:
        """Command the arm to stay where it measures now."""
        self._arm_pubs[arm].publish(Float64MultiArray(data=self.joints(arm)[0].tolist()))

    def set_gripper(self, arm: str, fraction: float) -> float:
        """Open (1) or close (0) the gripper and wait for it to stop moving.

        Returns: the opening it settled at, in m.
        """
        self._gripper_pubs[arm].publish(Float64(data=float(np.clip(fraction, 0.0, 1.0))))
        deadline = time.monotonic() + GRIPPER_TIMEOUT
        last = self.joints(arm)[1]
        time.sleep(0.3)
        while time.monotonic() < deadline:
            now = self.joints(arm)[1]
            if abs(now - last) < GRIPPER_STILL:
                break
            last = now
            time.sleep(0.2)
        return self.joints(arm)[1]

    # ---- cameras -------------------------------------------------------------
    @staticmethod
    def _keep(frames: dict[Any, Image], msg: Image) -> None:
        frames[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = msg
        while len(frames) > 3:
            frames.pop(next(iter(frames)))

    def _camera(self, cam: str) -> dict[str, Any] | None:
        """The newest colour frame whose depth has arrived, with that depth, or None."""
        info = self._info.get(cam)
        if info is None:
            return None
        for stamp in reversed(list(self._color[cam])):
            color, depth = self._color[cam].get(stamp), self._depth[cam].get(stamp)
            if color is not None and depth is not None:
                break
        else:
            return None
        rgb = np.frombuffer(color.data, np.uint8).reshape(color.height, color.width, 3)
        metres = np.frombuffer(depth.data, np.uint16).reshape(depth.height, depth.width).astype(np.float32) * 1e-3
        K = np.array(info.k).reshape(3, 3)
        if cam not in self._undistort:
            D = np.array(info.d)
            self._undistort[cam] = (cv2.initUndistortRectifyMap(K, D, None, K, (info.width, info.height), cv2.CV_32FC1)
                                    if np.any(D) else None)
        maps = self._undistort[cam]
        if maps is not None:
            rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
            metres = cv2.remap(metres, *maps, cv2.INTER_NEAREST)
        T = self._lookup(cam, color.header.stamp)
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        return {"images": {"rgb": np.ascontiguousarray(rgb), "depth": metres},
                "intrinsics": K, "pose_mat": T, "pose": np.array([*T[:3, 3], w, x, y, z])}

    def _lookup(self, frame: str, stamp: Any = None) -> np.ndarray:
        """frame in WORLD (4x4), at stamp when TF still holds it, else the latest."""
        try:
            msg = self._tf.lookup_transform(WORLD, frame, Time.from_msg(stamp) if stamp else Time())
        except TransformException:
            msg = self._tf.lookup_transform(WORLD, frame, Time())
        t, r = msg.transform.translation, msg.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _wait(self, ready, timeout: float, error: str | None) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            if time.monotonic() > deadline:
                if error:
                    raise RuntimeError(error)
                return
            time.sleep(0.02)
