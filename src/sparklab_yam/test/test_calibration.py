"""Hand-eye calibration against synthetic captures with a known answer."""

import time

import cv2
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import CameraInfo, Image

from sparklab_yam.calibration import BOARD, CalibrationNode, _T, _angle_deg, make_board, residuals, solve

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
D = np.zeros(5)
BOARD_OBJ = make_board(**BOARD).getChessboardCorners().astype(np.float64)
CENTER = BOARD_OBJ.mean(axis=0)


def _look_at(eye, target, roll):
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    R = np.column_stack((x, np.cross(z, x), z)) @ cv2.Rodrigues(np.array([0, 0, roll]))[0]
    return _T(R, eye)


def _views(rng, board_in_world, n):
    """Camera poses (in the world) looking at the board from varied angles."""
    center = board_in_world[:3, :3] @ CENTER + board_in_world[:3, 3]
    normal = -board_in_world[:3, 2]  # the printed side faces -z of the board frame
    cams = []
    for _ in range(n):
        d = normal + rng.uniform(-0.6, 0.6, 3)
        cams.append(_look_at(center + 0.45 * d / np.linalg.norm(d), center, rng.uniform(-0.5, 0.5)))
    return cams


def _observe(rng, board_in_cam):
    """Noisy detected corners and the PnP board pose, as the node would get them."""
    img, _ = cv2.projectPoints(BOARD_OBJ, cv2.Rodrigues(board_in_cam[:3, :3])[0], board_in_cam[:3, 3], K, D)
    img = img + rng.normal(0, 0.3, img.shape)
    _, rvec, tvec = cv2.solvePnP(BOARD_OBJ, img, K, D)
    return BOARD_OBJ, img, _T(cv2.Rodrigues(rvec)[0], tvec)


def _check(mode, X_true, gripper, board, obj, img):
    X = solve(mode, gripper, board)
    assert 1e3 * np.linalg.norm(X[:3, 3] - X_true[:3, 3]) < 2.0
    assert _angle_deg(X[:3, :3] @ X_true[:3, :3].T) < 0.3
    mm, deg, px = residuals(mode, X, gripper, board, obj, img, K, D)
    assert mm.mean() < 2.0 and deg.mean() < 0.5 and px.mean() < 1.5


def test_eye_in_hand_recovers_the_camera_on_the_gripper():
    rng = np.random.default_rng(0)
    X_true = _T(cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0], [0.03, -0.05, -0.08])
    board_in_base = _T(cv2.Rodrigues(np.array([np.pi, 0, 0]))[0], [0.4, 0.0, 0.0])
    cams = _views(rng, board_in_base, 15)
    gripper = [C @ np.linalg.inv(X_true) for C in cams]
    obs = [_observe(rng, np.linalg.inv(C) @ board_in_base) for C in cams]
    _check("eye_in_hand", X_true, gripper, *map(list, zip(*[(o[2], o[0], o[1]) for o in obs])))


def test_eye_to_hand_recovers_the_fixed_camera():
    rng = np.random.default_rng(1)
    X_true = _T(cv2.Rodrigues(np.array([np.pi - 0.3, 0.1, 0.0]))[0], [0.5, 0.1, 0.8])
    board_on_tool = _T(cv2.Rodrigues(np.array([0.0, 0.2, 0.1]))[0], [0.0, 0.0, 0.05])
    board_in_base = []
    for C in _views(rng, np.eye(4), 15):  # board poses seen from the fixed camera
        board_in_base.append(X_true @ np.linalg.inv(C))
    gripper = [P @ np.linalg.inv(board_on_tool) for P in board_in_base]
    obs = [_observe(rng, np.linalg.inv(X_true) @ P) for P in board_in_base]
    _check("eye_to_hand", X_true, gripper, *map(list, zip(*[(o[2], o[0], o[1]) for o in obs])))


# ---- the node, end to end, on rendered images ------------------------------
def _renderer():
    """Render the printed board as a 640x480 camera would see it from any pose."""
    board = make_board(**BOARD)
    flat = board.generateImage((1400, 1000), marginSize=60)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(flat)
    obj, img = board.matchImagePoints(corners, ids)
    metric_to_flat, _ = cv2.findHomography(obj[:, 0, :2], img[:, 0])

    def render(board_in_cam):
        R, t = board_in_cam[:3, :3], board_in_cam[:3, 3]
        metric_to_cam = K @ np.column_stack((R[:, 0], R[:, 1], t))
        gray = cv2.warpPerspective(flat, metric_to_cam @ np.linalg.inv(metric_to_flat), (640, 480),
                                   borderValue=128)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return render


@pytest.fixture
def node():
    rclpy.init(args=["--ros-args", "-p", "mode:=eye_in_hand"])
    n = CalibrationNode()
    info = CameraInfo(width=640, height=480, k=K.flatten().tolist(), d=D.tolist())
    n._on_info(info)
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def _stamp(t: float) -> TimeMsg:
    return TimeMsg(sec=int(t), nanosec=int((t % 1) * 1e9))


def test_node_captures_rendered_views_and_solves(node):
    rng = np.random.default_rng(2)
    render = _renderer()
    X_true = _T(cv2.Rodrigues(np.array([0.05, -0.1, 0.02]))[0], [0.02, -0.04, -0.07])
    board_in_base = _T(cv2.Rodrigues(np.array([np.pi, 0, 0]))[0], [0.4, 0.0, 0.0])
    for C in _views(rng, board_in_base, 8):
        A = C @ np.linalg.inv(X_true)
        t = time.time()
        node._poses.clear()
        for k in range(6):  # half a second of a still arm at 100 Hz, thinned
            node._poses.append((t - 0.5 + 0.1 * k, A))
        rgb = render(np.linalg.inv(C) @ board_in_base)
        msg = Image(height=480, width=640, encoding="rgb8", step=640 * 3, data=rgb.tobytes())
        msg.header.stamp = _stamp(t)
        node._last_draw = 0.0
        node._on_image(msg)
        print(node.capture())
    X = node.result["X"]
    assert len(node.samples) == 8
    assert 1e3 * np.linalg.norm(X[:3, 3] - X_true[:3, 3]) < 3.0
    assert _angle_deg(X[:3, :3] @ X_true[:3, :3].T) < 0.5
    assert node.result["px"] < 1.5


def test_node_refuses_a_moving_arm(node):
    render = _renderer()
    board_in_cam = _T(np.eye(3), [-0.14, -0.1, 0.45])
    t = time.time()
    node._poses.extend([(t - 0.3, _T(np.eye(3), [0.3, 0, 0.2])), (t, _T(np.eye(3), [0.31, 0, 0.2]))])
    msg = Image(height=480, width=640, encoding="rgb8", step=640 * 3, data=render(board_in_cam).tobytes())
    msg.header.stamp = _stamp(t)
    node._on_image(msg)
    assert "moving" in node.capture()
    assert not node.samples
