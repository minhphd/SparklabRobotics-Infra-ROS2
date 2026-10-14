"""Hand-eye calibration: where a camera is relative to an arm, from a ChArUco board.

    ros2 run sparklab_yam calibrate --ros-args -p mode:=eye_in_hand -p arm:=left -p camera:=left_wrist
    ros2 run sparklab_yam calibrate --ros-args -p mode:=eye_to_hand -p arm:=left -p camera:=top
    ros2 run sparklab_yam calibration_board          # printable board PNG

eye_in_hand: the camera rides on the gripper and the board lies still on the
table; solves the camera in `<arm>_tool0`. eye_to_hand: the camera is fixed and
the board is clamped to the gripper; solves the camera in `<arm>_base`.

The arm goes into zero-gravity mode. Move it by hand, hold still, press Enter to
capture; every capture re-solves and prints the running error. `u` + Enter
undoes the last capture, `s` saves, `q` quits and the arm holds where it is.
The live detection is drawn on /calibration/image.
"""

from __future__ import annotations

import argparse
import array
import collections
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import SetBool

MODES = ("eye_in_hand", "eye_to_hand")
# The lab's board: 24 squares across, 17 down. 204 markers, so the 5x5 dictionary
# needs its 250-marker set; its first markers are the same in every size.
BOARD = {"squares_x": 24, "squares_y": 17, "square_length": 0.015, "marker_length": 0.011,
         "dictionary": "DICT_5X5_250"}
MIN_CORNERS = 6
# The arm counts as still when tool0 moved less than this over the last half second.
STILL_MM, STILL_DEG = 1.0, 0.2


def make_board(squares_x, squares_y, square_length, marker_length, dictionary):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)


def _T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _mean_pose(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean translation and chordal-mean rotation."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return _T(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def solve(mode: str, gripper: list[np.ndarray], board: list[np.ndarray]) -> np.ndarray:
    """AX = XB with OpenCV's Park method.

    gripper: tool0 in the arm base, per capture (4x4).
    board: the board in the camera, per capture (4x4).
    Returns: the camera in tool0 (eye_in_hand) or in the arm base (eye_to_hand).
    """
    A = gripper if mode == "eye_in_hand" else [np.linalg.inv(T) for T in gripper]
    R, t = cv2.calibrateHandEye([a[:3, :3] for a in A], [a[:3, 3] for a in A],
                                [b[:3, :3] for b in board], [b[:3, 3] for b in board],
                                method=cv2.CALIB_HAND_EYE_PARK)
    return _T(R, t)


def residuals(mode: str, X: np.ndarray, gripper: list[np.ndarray], board: list[np.ndarray],
              obj: list[np.ndarray], img: list[np.ndarray], K: np.ndarray, D: np.ndarray):
    """How well X explains each capture.

    The board never moves (in the base for eye_in_hand, on tool0 for
    eye_to_hand), so every capture predicts the same board pose through X.
    Returns: per capture, the distance (mm) and angle (deg) of its prediction
    from the mean, and the reprojection RMS (px) of the mean board through X.
    """
    if mode == "eye_in_hand":
        fixed = [A @ X @ B for A, B in zip(gripper, board)]
        to_cam = [np.linalg.inv(A @ X) for A in gripper]
    else:
        fixed = [np.linalg.inv(A) @ X @ B for A, B in zip(gripper, board)]
        to_cam = [np.linalg.inv(X) @ A for A in gripper]
    F = _mean_pose(fixed)
    mm = np.array([1e3 * np.linalg.norm(T[:3, 3] - F[:3, 3]) for T in fixed])
    deg = np.array([_angle_deg(T[:3, :3] @ F[:3, :3].T) for T in fixed])
    px = []
    for C, o, i in zip(to_cam, obj, img):
        B = C @ F
        proj, _ = cv2.projectPoints(o, cv2.Rodrigues(B[:3, :3])[0], B[:3, 3], K, D)
        px.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1))))
    return mm, deg, np.array(px)


def _pose_to_T(msg: PoseStamped) -> np.ndarray:
    p, q = msg.pose.position, msg.pose.orientation
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array([q.w, q.x, q.y, q.z]))
    return _T(R.reshape(3, 3), [p.x, p.y, p.z])


def _seconds(stamp) -> float:
    return Time.from_msg(stamp).nanoseconds * 1e-9


class CalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("calibration")
        self.mode = self.declare_parameter("mode", "eye_in_hand").value
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.arm = self.declare_parameter("arm", "left").value
        self.camera = self.declare_parameter("camera", "left_wrist").value
        self.output = Path(self.declare_parameter("output", f"calibration_{self.camera}.yaml").value)
        self.board_cfg = {k: self.declare_parameter(k, v).value for k, v in BOARD.items()}
        self.board = make_board(**self.board_cfg)
        self._detector = cv2.aruco.CharucoDetector(self.board)

        self._poses = collections.deque(maxlen=300)  # (seconds, tool0 in base)
        self._joints = collections.deque(maxlen=300)  # (seconds, joint positions)
        self._K = self._D = None
        self._detection = None
        self._last_draw = 0.0
        self.samples: list[dict] = []
        self.result: dict | None = None

        self.create_subscription(PoseStamped, f"/{self.arm}/ee_pose",
                                 lambda m: self._poses.append((_seconds(m.header.stamp), _pose_to_T(m))), 10)
        self.create_subscription(JointState, f"/{self.arm}/joint_states",
                                 lambda m: self._joints.append((_seconds(m.header.stamp), list(m.position))), 10)
        self.create_subscription(CameraInfo, f"/cameras/{self.camera}/color/camera_info", self._on_info, 1)
        self.create_subscription(Image, f"/cameras/{self.camera}/color/image_raw", self._on_image, 1)
        self._image_pub = self.create_publisher(Image, "~/image", 1)
        self._zero_g = self.create_client(SetBool, f"/{self.arm}/zero_gravity")

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k).reshape(3, 3)
        self._D = np.array(msg.d)

    def _pose_at(self, t: float) -> np.ndarray | None:
        """tool0 in the base at time t, if a pose within 50 ms of it is known."""
        poses = list(self._poses)  # appended to from the ROS thread
        if not poses:
            return None
        stamp, T = min(poses, key=lambda p: abs(p[0] - t))
        return T if abs(stamp - t) < 0.05 else None

    def _still(self, t: float) -> bool:
        recent = [T for s, T in self._poses if t - 0.5 <= s <= t]
        if len(recent) < 2:
            return False
        last = recent[-1]
        return all(1e3 * np.linalg.norm(T[:3, 3] - last[:3, 3]) < STILL_MM
                   and _angle_deg(T[:3, :3] @ last[:3, :3].T) < STILL_DEG for T in recent)

    def _on_image(self, msg: Image) -> None:
        # Detection is ~10-20 ms on a 720p frame; 10 Hz is plenty to aim the arm by.
        if self._K is None or time.monotonic() - self._last_draw < 0.1:
            return
        self._last_draw = time.monotonic()
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        t = _seconds(msg.header.stamp)
        corners, ids, _, _ = self._detector.detectBoard(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        n = 0 if ids is None else len(ids)
        status = f"board: {n} corners"
        if n >= MIN_CORNERS:
            obj, img = self.board.matchImagePoints(corners, ids)
            ok, rvec, tvec = cv2.solvePnP(obj, img, self._K, self._D)
            if ok:
                still = self._still(t)
                proj, _ = cv2.projectPoints(obj, rvec, tvec, self._K, self._D)
                fit = float(np.sqrt(np.mean(np.sum((proj - img).reshape(-1, 2) ** 2, axis=1))))
                self._detection = {"t": t, "obj": obj, "img": img, "still": still, "fit": fit,
                                   "board": _T(cv2.Rodrigues(rvec)[0], tvec)}
                status += "  HOLD STILL" if not still else "  still - press Enter"
                cv2.aruco.drawDetectedCornersCharuco(rgb, corners, ids)
                cv2.drawFrameAxes(rgb, self._K, self._D, rvec, tvec, self.board_cfg["square_length"] * 2)
        lines = [status, f"captures: {len(self.samples)}"]
        if self.result:
            lines.append("error: {mm:.1f} mm  {deg:.2f} deg  {px:.2f} px".format(**self.result))
        scale = msg.width / 1280
        for k, line in enumerate(lines):
            org = (int(10 * scale), int((35 + 35 * k) * scale))
            cv2.putText(rgb, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(5 * scale) + 1)
            cv2.putText(rgb, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), int(2 * scale) + 1)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding="rgb8",
                    step=msg.width * 3)
        out.data = array.array("B", rgb.tobytes())
        self._image_pub.publish(out)

    def set_zero_gravity(self, on: bool) -> None:
        if not self._zero_g.wait_for_service(timeout_sec=3.0):
            print(f"  ! /{self.arm}/zero_gravity not available — is the driver running?")
            return
        future = self._zero_g.call_async(SetBool.Request(data=on))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if future.done():
            print(f"  {'' if future.result().success else '! '}{future.result().message}")

    def capture(self) -> str:
        det = self._detection
        if det is None or time.time() - det["t"] > 0.5:
            return "no board in view"
        if not det["still"]:
            return "the arm is moving — hold it still"
        gripper = self._pose_at(det["t"])
        if gripper is None:
            return f"no fresh /{self.arm}/ee_pose — is the driver running?"
        joints = min(self._joints, key=lambda j: abs(j[0] - det["t"]), default=(0.0, None))[1]
        self.samples.append({"gripper": gripper, "joints": joints, **det})
        return self._report()

    def undo(self) -> str:
        if self.samples:
            self.samples.pop()
        return self._report()

    def _report(self) -> str:
        n = len(self.samples)
        # The board alone through the intrinsics: well under 1 px, or the camera side is off.
        fit = f"board fit {self.samples[-1]['fit']:.2f} px" if self.samples else ""
        if n < 3:
            self.result = None
            return f"#{n} captured — need at least 3 | {fit}"
        g = [s["gripper"] for s in self.samples]
        X = solve(self.mode, g, [s["board"] for s in self.samples])
        mm, deg, px = residuals(self.mode, X, g, [s["board"] for s in self.samples],
                                [s["obj"] for s in self.samples], [s["img"] for s in self.samples],
                                self._K, self._D)
        spread = max(_angle_deg(a[:3, :3] @ b[:3, :3].T) for a in g for b in g)
        rpy = cv2.RQDecomp3x3(X[:3, :3])[0]  # already degrees
        self.result = {"X": X, "mm": float(mm.mean()), "deg": float(deg.mean()), "px": float(px.mean()),
                       "spread": spread}
        worst = int(np.argmax(px))
        xyz = ", ".join(f"{v:.1f}" for v in 1e3 * X[:3, 3])
        line = (f"#{n}  camera at [{xyz}] mm, rpy [{', '.join(f'{v:.1f}' for v in rpy)}] deg | "
                f"error {mm.mean():.1f} mm {deg.mean():.2f} deg {px.mean():.2f} px "
                f"(worst #{worst + 1}: {px[worst]:.2f} px) | rotation spread {spread:.0f} deg | {fit}")
        if spread < 30.0:
            line += "\n    tilt and turn the gripper more: rotation spread under 30 deg is unreliable"
        return line

    def save(self) -> str:
        if self.result is None:
            return "nothing to save — need at least 3 captures"
        X = self.result["X"]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, X[:3, :3].flatten())
        parent = f"{self.arm}_tool0" if self.mode == "eye_in_hand" else f"{self.arm}_base"
        data = {
            "mode": self.mode, "parent_frame": parent, "child_frame": self.camera,
            "translation_m": X[:3, 3].tolist(),
            "quaternion_xyzw": [float(q[1]), float(q[2]), float(q[3]), float(q[0])],
            "matrix": X.tolist(),
            "error": {k: self.result[k] for k in ("mm", "deg", "px", "spread")},
            "captures": len(self.samples), "board": self.board_cfg,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": [{"tool0_in_base": s["gripper"].tolist(), "board_in_camera": s["board"].tolist(),
                         "joint_pos": s["joints"], "board_fit_px": s["fit"],
                         "board_points": s["obj"].reshape(-1, 3).tolist(),
                         "image_points": s["img"].reshape(-1, 2).tolist()}
                        for s in self.samples],
        }
        self.output.write_text(yaml.safe_dump(data, sort_keys=False))
        return f"saved {self.output.resolve()}"


def main() -> None:
    # drawFrameAxes warns on every frame where an axis tip leaves the image.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CalibrationNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.1) for _ in iter(stop.is_set, True)])
    spinner.start()
    print(f"{node.mode}: /{node.arm} with camera {node.camera} — live view on /calibration/image")
    node.set_zero_gravity(True)
    print("move the arm, hold still, Enter = capture | u = undo | s = save | q = quit")
    try:
        while (line := input("> ").strip().lower()) != "q":
            print("  " + {"": node.capture, "u": node.undo, "s": node.save}.get(
                line, lambda: "Enter, u, s or q")())
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        node.set_zero_gravity(False)
        stop.set()
        spinner.join()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


def make_board_image() -> None:
    ap = argparse.ArgumentParser(description="Write a printable ChArUco board PNG.")
    ap.add_argument("--squares-x", type=int, default=BOARD["squares_x"])
    ap.add_argument("--squares-y", type=int, default=BOARD["squares_y"])
    ap.add_argument("--square-mm", type=float, default=1e3 * BOARD["square_length"])
    ap.add_argument("--marker-mm", type=float, default=1e3 * BOARD["marker_length"])
    ap.add_argument("--dictionary", default=BOARD["dictionary"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="charuco_board.png")
    args = ap.parse_args()
    board = make_board(args.squares_x, args.squares_y, args.square_mm / 1e3, args.marker_mm / 1e3,
                       args.dictionary)
    px_per_mm = args.dpi / 25.4
    margin = round(10 * px_per_mm)
    size = (round(args.squares_x * args.square_mm * px_per_mm) + 2 * margin,
            round(args.squares_y * args.square_mm * px_per_mm) + 2 * margin)
    cv2.imwrite(args.out, board.generateImage(size, marginSize=margin))
    print(f"wrote {args.out}: {args.squares_x}x{args.squares_y} squares of {args.square_mm} mm, "
          f"{args.dpi} dpi. Print at 100%, measure a square, and pass it as square_length.")


if __name__ == "__main__":
    main()
