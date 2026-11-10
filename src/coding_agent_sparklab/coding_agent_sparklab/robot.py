"""The robot API the coding agent's scripts call: both YAM arms and the three cameras over ROS 2.

    from coding_agent_sparklab.robot import (get_joint_angles, get_ee_pose, fk, ik, command, get_image,
                                             get_depth, get_intrinsics, EXTRINSICS, FINGERTIPS)

Poses are tool0, the stock gripper's grasp site, in that arm's base frame (x
forward, y left, z up, m); quaternions are w, x, y, z. fk, ik and
EXTRINSICS need nothing running; the rest need the rig, or a sim, on ROS.
"""

import atexit
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray

from sparklab_yam.ik.solver import IKSolver
from sparklab_yam.model.kinematics import DEFAULT_Q_REST

ARMS = ("left", "right")
CAMERAS = ("top", "left_wrist", "right_wrist")
FINGERTIPS = 0.075
"""m past tool0 along its z axis where the long fin-ray fingertips end: they are 14.5 cm,
on the sliders that carry the stock 8 cm fingers, whose tips end 1 cm past tool0."""
SPEED = 0.4
"""rad/s, the fastest any joint moves under command()."""
DT = 0.025
MAX_LAG = 0.15
"""rad between the ramp and the measured joint before command() stops the arm."""
SETTLE_TOL = 0.01
SETTLE_TIMEOUT = 2.0
GRIPPER_STILL = 0.05
GRIPPER_TIMEOUT = 3.0
STALE = 0.5
"""s after which a joint state means the driver is down."""
CONNECT_TIMEOUT = 3.0
POS_TOL, ROT_TOL = 1e-3, 1e-2
IK_ITERS, IK_RESTARTS = 500, 8

# The step cap keeps a far start from overshooting; teleop's caps are per tick and much smaller.
_solver = IKSolver(max_dq_per_joint=[0.3] * 6)
LIMITS = _solver.joint_limits


def _extrinsics() -> dict[tuple[str, str], np.ndarray]:
    out = {}
    path = Path(get_package_share_directory("sparklab_yam")) / "config" / "extrinsics.yaml"
    for t in yaml.safe_load(path.read_text())["transforms"]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(t["rotation"]).as_matrix()
        T[:3, 3] = t["translation"]
        out[t["parent"], t["child"]] = T
    return out


EXTRINSICS = _extrinsics()
"""(parent, child) -> the child frame's pose in the parent (4x4): the rig's calibration,
as sparklab_yam's extrinsics.yaml gives it. Camera frames are optical."""


class _Rig(Node):
    def __init__(self) -> None:
        super().__init__("coding_agent_robot")
        self.states: dict[str, tuple[float, np.ndarray]] = {}
        for arm in ARMS:
            self.create_subscription(
                JointState, f"/{arm}/joint_states",
                lambda m, a=arm: self.states.__setitem__(a, (time.monotonic(), np.array(m.position))), 10)
        self.arm_pubs = {a: self.create_publisher(Float64MultiArray, f"/{a}/arm/command", 10) for a in ARMS}
        self.gripper_pubs = {a: self.create_publisher(Float64, f"/{a}/gripper/command", 10) for a in ARMS}
        self.frames: dict[tuple[str, str], tuple[float, Image]] = {}
        self.infos: dict[str, CameraInfo] = {}
        self.watched: set[tuple[str, str]] = set()

    def watch(self, camera: str, stream: str) -> None:
        """Subscribe to a camera's color, depth or info on first use, so a script receives only what it asks for."""
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {CAMERAS}, not {camera!r}")
        if (camera, stream) in self.watched:
            return
        self.watched.add((camera, stream))
        if stream == "info":
            self.create_subscription(CameraInfo, f"/cameras/{camera}/color/camera_info",
                                     lambda m: self.infos.__setitem__(camera, m), 1)
        else:
            self.create_subscription(Image, f"/cameras/{camera}/{stream}/image_raw",
                                     lambda m: self.frames.__setitem__((camera, stream), (time.monotonic(), m)), 1)


_rig: _Rig | None = None


def _node() -> _Rig:
    global _rig
    if _rig is None:
        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        _rig = _Rig()
        executor = SingleThreadedExecutor()
        executor.add_node(_rig)
        stop = threading.Event()
        thread = threading.Thread(
            target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)], daemon=True)
        thread.start()

        def close() -> None:
            stop.set()
            thread.join()
            executor.shutdown()
            _rig.destroy_node()
            rclpy.try_shutdown()
        atexit.register(close)
    return _rig


def _wait_for(value, error: str):
    deadline = time.monotonic() + CONNECT_TIMEOUT
    while (v := value()) is None:
        if time.monotonic() > deadline:
            raise RuntimeError(error)
        time.sleep(0.01)
    return v


def get_joint_angles(arm: str) -> np.ndarray:
    """The arm's measured state: joints 1-6 in rad, then the gripper, 1 open .. 0 closed.

    arm: "left" or "right".
    Returns: (7,).
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    rig = _node()

    def fresh():
        stamp, position = rig.states.get(arm, (-np.inf, None))
        return position.copy() if time.monotonic() - stamp < STALE else None
    return _wait_for(fresh, f"no joint states from the {arm} arm: is its driver running?")


def _frame(camera: str, stream: str) -> Image:
    rig = _node()
    rig.watch(camera, stream)
    asked = time.monotonic()

    def newer():
        stamp, msg = rig.frames.get((camera, stream), (-np.inf, None))
        return msg if stamp > asked else None
    return _wait_for(newer, f"no {stream} frames from the {camera} camera: is the camera node running?")


def get_image(camera: str) -> np.ndarray:
    """A colour frame received after the call, as the camera sends it: wrist frames are not undistorted.

    camera: "top", "left_wrist" or "right_wrist".
    Returns: (H, W, 3) uint8 RGB.
    """
    msg = _frame(camera, "color")
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()


def get_depth(camera: str) -> np.ndarray:
    """A depth frame received after the call, aligned pixel for pixel with get_image's frames.

    Each pixel is the distance along the camera's z axis, 0 where the camera got no reading.
    Returns: (H, W) float32, m.
    """
    msg = _frame(camera, "depth")
    return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 1e-3


def get_intrinsics(camera: str) -> tuple[np.ndarray, np.ndarray]:
    """The camera's pinhole intrinsics for get_image's frames, from its camera_info.

    Returns: K (3, 3), distortion coefficients k1, k2, p1, p2, k3 (5,).
    """
    rig = _node()
    rig.watch(camera, "info")
    info = _wait_for(lambda: rig.infos.get(camera), f"no camera_info from the {camera} camera: "
                     "is the camera node running?")
    return np.array(info.k).reshape(3, 3), np.array(info.d)


def get_ee_pose(arm: str) -> tuple[np.ndarray, np.ndarray]:
    """The arm's measured tool0 pose, as fk of its measured joints.

    Returns: position (3,), quaternion wxyz (4,).
    """
    return fk(get_joint_angles(arm)[:6])


def fk(joint_angles) -> tuple[np.ndarray, np.ndarray]:
    """tool0's pose for joints 1-6; the arms share one model.

    Returns: position (3,), quaternion wxyz (4,).
    """
    return _solver.fk(np.asarray(joint_angles, dtype=float)[:6])


def ik(position, quat_wxyz, seed=None) -> np.ndarray:
    """Joints 1-6 that put tool0 at a pose, within the joint limits.

    Raises ValueError when no solution comes within 1 mm and 0.6 degrees.

    seed: joints to start from; pass the arm's present ones to stay near them.
        Defaults to [0, 0.5, 1.0, -1.0, 0, 0], the gripper in front of the base pointing forward and down.
    Returns: (6,).
    """
    position = np.asarray(position, dtype=float)
    quat = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
    rng = np.random.default_rng(0)
    first = DEFAULT_Q_REST if seed is None else np.asarray(seed, dtype=float)[:6]
    closest = (np.inf, 0.0, 0.0)
    for q in [first, *(rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(IK_RESTARTS))]:
        for _ in range(IK_ITERS):
            step = _solver.solve(position, quat, q)
            done = np.max(np.abs(step - q)) < 1e-7
            q = step
            if done:
                break
        reached, reached_quat = _solver.fk(q)
        pos_err = float(np.linalg.norm(reached - position))
        rot_err = 2 * float(np.arccos(min(1.0, abs(float(reached_quat @ quat)))))
        if pos_err < POS_TOL and rot_err < ROT_TOL:
            return q
        closest = min(closest, (pos_err / POS_TOL + rot_err / ROT_TOL, pos_err, rot_err))
    raise ValueError(f"no joint angles reach that pose; the closest is {closest[1] * 1000:.0f} mm "
                     f"and {np.degrees(closest[2]):.0f} deg off")


def command(arm: str, joint_angles) -> np.ndarray:
    """Move the arm to joints 1-6 and wait until it is there; a 7th value then sets the gripper.

    The joints ramp together in a straight line, none faster than 0.4 rad/s. The
    gripper goes 1 open .. 0 closed and stops early on whatever it closes on.
    Raises ValueError for a joint outside its limits, and RuntimeError, with the
    arm held where it is, when the arm falls 0.15 rad behind: blocked, or its driver down.
    Returns: the measured state afterwards, as get_joint_angles.
    """
    target = np.asarray(joint_angles, dtype=float)
    if target.shape not in ((6,), (7,)):
        raise ValueError(f"command takes 6 joint angles, or 7 with the gripper, not shape {target.shape}")
    outside = (target[:6] < LIMITS[:, 0]) | (target[:6] > LIMITS[:, 1])
    if outside.any():
        j = int(np.argmax(outside))
        raise ValueError(f"joint {j + 1} target {target[j]:.3f} rad is outside its limits "
                         f"[{LIMITS[j, 0]:.3f}, {LIMITS[j, 1]:.3f}]")
    rig = _node()
    start = get_joint_angles(arm)[:6]
    steps = max(1, int(np.ceil(np.max(np.abs(target[:6] - start)) / SPEED / DT)))
    tick = time.monotonic()
    for waypoint in start + np.outer(np.arange(1, steps + 1) / steps, target[:6] - start):
        rig.arm_pubs[arm].publish(Float64MultiArray(data=waypoint.tolist()))
        measured = get_joint_angles(arm)[:6]
        lag = np.abs(measured - waypoint)
        if lag.max() > MAX_LAG:
            rig.arm_pubs[arm].publish(Float64MultiArray(data=measured.tolist()))
            raise RuntimeError(f"the {arm} arm fell {lag.max():.2f} rad behind at joint {lag.argmax() + 1} "
                               "and was stopped: blocked, or its driver is down")
        tick += DT
        time.sleep(max(0.0, tick - time.monotonic()))
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while np.max(np.abs(get_joint_angles(arm)[:6] - target[:6])) > SETTLE_TOL and time.monotonic() < deadline:
        time.sleep(0.02)
    if len(target) == 7:
        rig.gripper_pubs[arm].publish(Float64(data=float(target[6])))
        deadline = time.monotonic() + GRIPPER_TIMEOUT
        last = get_joint_angles(arm)[6]
        time.sleep(0.3)
        while time.monotonic() < deadline:
            now = get_joint_angles(arm)[6]
            if abs(now - last) < GRIPPER_STILL:
                break
            last = now
            time.sleep(0.2)
    return get_joint_angles(arm)
