"""The robot API: kinematics and calibration alone, then against two simulated drivers and a fake camera."""

import array
import threading
import time

import numpy as np
import pytest
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from coding_agent_sparklab import robot


def test_ik_inverts_fk():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.uniform(robot.LIMITS[:, 0], robot.LIMITS[:, 1])
        pos, quat = robot.fk(q)
        try:
            solved = robot.ik(pos, quat)
        except ValueError:
            continue
        reached, reached_quat = robot.fk(solved)
        assert np.linalg.norm(reached - pos) < robot.POS_TOL
        assert abs(reached_quat @ quat) > np.cos(robot.ROT_TOL / 2)


def test_ik_reaches_down_onto_the_table_and_refuses_the_far_side_of_the_room():
    down = [0.0, 0.0, 1.0, 0.0]
    pos, _ = robot.fk(robot.ik([0.35, 0.0, 0.05], down))
    assert np.allclose(pos, [0.35, 0.0, 0.05], atol=robot.POS_TOL)
    with pytest.raises(ValueError, match="no joint angles"):
        robot.ik([2.0, 0.0, 0.3], down)


def test_extrinsics_are_the_rigs_calibration():
    T = robot.EXTRINSICS["left_base", "right_base"]
    np.testing.assert_allclose(T[:3, 3], [0.0196, -0.5478, 0.0080])
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert {("left_base", "top"), ("left_tool0", "left_wrist"), ("right_tool0", "right_wrist")} <= set(robot.EXTRINSICS)


def test_command_moves_one_arm_and_sets_the_gripper(drivers):
    target = np.array([0.3, 0.8, 0.9, -0.2, 0.3, 0.4])
    other = robot.get_joint_angles("right")
    start = time.monotonic()
    state = robot.command("left", [*target, 0.0])
    assert time.monotonic() - start > np.max(np.abs(target)) / robot.SPEED
    assert state.shape == (7,)
    assert np.max(np.abs(state[:6] - target)) < robot.SETTLE_TOL
    assert state[6] < 0.2
    assert np.allclose(robot.get_joint_angles("right"), other, atol=0.01)
    pos, quat = robot.get_ee_pose("left")
    expected = robot.fk(robot.get_joint_angles("left")[:6])
    assert np.allclose(pos, expected[0], atol=1e-3) and np.allclose(quat, expected[1], atol=1e-2)


def test_command_refuses_targets_outside_the_limits(drivers):
    before = robot.get_joint_angles("left")
    with pytest.raises(ValueError, match="joint 3"):
        robot.command("left", [0.0, 1.0, 3.2, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="6 joint angles"):
        robot.command("left", [0.0, 1.0])
    with pytest.raises(ValueError, match="arm must be"):
        robot.get_joint_angles("Left")
    assert np.allclose(robot.get_joint_angles("left")[:6], before[:6], atol=0.01)


@pytest.fixture
def top_camera(drivers):
    robot._node()
    node = Node("fake_top_camera")
    image = node.create_publisher(Image, "/cameras/top/color/image_raw", 1)
    depth = node.create_publisher(Image, "/cameras/top/depth/image_raw", 1)
    info = node.create_publisher(CameraInfo, "/cameras/top/color/camera_info", 1)
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[:, 32:] = (200, 10, 10)
    mm = np.full((48, 64), 912, np.uint16)
    mm[0, 0] = 0
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            image.publish(Image(height=48, width=64, encoding="rgb8", step=64 * 3,
                                data=array.array("B", rgb.tobytes())))
            depth.publish(Image(height=48, width=64, encoding="16UC1", step=64 * 2,
                                data=array.array("B", mm.tobytes())))
            info.publish(CameraInfo(k=[50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0], d=[0.1, 0.0, 0.0, 0.0, 0.0]))
            time.sleep(0.05)
    thread = threading.Thread(target=publish)
    thread.start()
    yield rgb
    stop.set()
    thread.join()
    node.destroy_node()


def test_a_camera_gives_a_fresh_frame_and_its_intrinsics(top_camera):
    np.testing.assert_array_equal(robot.get_image("top"), top_camera)
    metres = robot.get_depth("top")
    assert metres.shape == (48, 64) and metres[0, 0] == 0.0 and metres[5, 40] == pytest.approx(0.912)
    K, distortion = robot.get_intrinsics("top")
    assert K[0, 2] == 32.0 and K[1, 1] == 50.0 and distortion[0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="camera must be"):
        robot.get_image("front")
