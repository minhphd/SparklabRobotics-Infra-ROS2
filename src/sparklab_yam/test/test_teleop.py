import threading
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Joy

from sparklab_yam.teleop import (
    DEFAULT_R_CALIB,
    QuestTeleop,
    QuestTeleopConfig,
    _headset_relative_calibration,
)


@pytest.fixture
def ros():
    rclpy.init()
    node, fake_quest = Node("teleop_under_test"), Node("fake_quest")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(fake_quest)
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    yield node, fake_quest
    stop.set()
    thread.join()
    executor.shutdown()
    rclpy.try_shutdown()


def test_single_hand_builds_only_one_arm(ros):
    teleop = QuestTeleop(QuestTeleopConfig(), ros[0], hands=("right",))
    action = teleop.step()
    assert len(action) == 7 and all(k.startswith("right_") for k in action)


def test_stow_ramp_advances_with_a_stale_pose(ros):
    teleop = QuestTeleop(QuestTeleopConfig(rest_ramp_duration_s=1.0), ros[0], hands=("right",))
    arm = teleop._arms["right"]
    arm["qpos"][:6] = 1.0
    teleop._stow_arm("right", arm)
    arm["ramp_start_t"] = time.perf_counter() - 0.5

    assert teleop._advance_ramp("right", arm, np.zeros(3), np.array([1.0, 0, 0, 0]), pose_is_fresh=False)
    assert arm["ramp_active"]
    np.testing.assert_allclose(arm["qpos"][:6], 0.5 * (1.0 + arm["ramp_target_q"]), atol=0.03)


def test_stow_completion_does_not_reanchor_from_a_stale_pose(ros):
    teleop = QuestTeleop(QuestTeleopConfig(rest_ramp_duration_s=1.0), ros[0], hands=("right",))
    arm = teleop._arms["right"]
    arm["engaged"] = True
    teleop._stow_arm("right", arm)
    arm["ramp_start_t"] = time.perf_counter() - 2.0

    teleop._advance_ramp("right", arm, np.zeros(3), np.array([1.0, 0, 0, 0]), pose_is_fresh=False)
    assert not arm["ramp_active"]
    assert arm["needs_reanchor"]


def test_headset_yaw_maps_head_forward_to_robot_forward():
    yaw = np.pi / 2
    calibrated = _headset_relative_calibration(
        np.asarray(DEFAULT_R_CALIB), np.array([0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)]))
    np.testing.assert_allclose(calibrated @ np.array([-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(calibrated @ calibrated.T, np.eye(3), atol=1e-12)


def test_clutch_and_trigger_from_the_quest_topics_drive_only_that_arm(ros):
    node, fake_quest = ros
    teleop = QuestTeleop(QuestTeleopConfig(), node)
    pose_pub = fake_quest.create_publisher(PoseStamped, "/quest/right/pose", 10)
    joy_pub = fake_quest.create_publisher(Joy, "/quest/right/joy", 10)

    def send(z: float, grip: bool, trigger: float) -> dict:
        pose = PoseStamped()
        pose.header.stamp = fake_quest.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = 0.2, 1.2, z
        pose.pose.orientation.w = 1.0
        joy = Joy(buttons=[int(trigger > 0.5), int(grip), 0, 0, 0, 0],
                  axes=[trigger, float(grip), 0.0, 0.0, 0.0, 0.0])
        pose_pub.publish(pose)
        joy_pub.publish(joy)
        time.sleep(0.01)
        return teleop.step()

    home = send(-0.3, grip=False, trigger=0.0)
    send(-0.3, grip=True, trigger=0.0)                       # clutch in
    for i in range(50):                                      # 5 cm forward, trigger pulled
        action = send(-0.3 - 0.001 * (i + 1), grip=True, trigger=1.0)

    right = [f"right_joint_{j}.pos" for j in range(1, 7)]
    left = [f"left_joint_{j}.pos" for j in range(1, 7)]
    assert max(abs(action[k] - home[k]) for k in right) > 0.05
    assert all(action[k] == home[k] for k in left)
    assert action["right_gripper.pos"] == 1.0
    assert teleop.xr_snapshot()["engaged"] == {"left": False, "right": True}
