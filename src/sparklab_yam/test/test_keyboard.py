"""The keyboard node against a simulated left arm."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from sparklab_yam.driver import YamDriver
from sparklab_yam.keyboard import Keyboard


@pytest.fixture
def rig():
    rclpy.init(args=["--ros-args", "-p", "sim:=true", "-p", "park_on_exit:=false", "-r", "__ns:=/left"])
    driver, keyboard, probe = YamDriver(), Keyboard(), Node("probe")
    executor = SingleThreadedExecutor()
    for n in (driver, keyboard, probe):
        executor.add_node(n)
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    states: list[JointState] = []
    probe.create_subscription(JointState, "/left/joint_states", states.append, 10)
    time.sleep(0.3)
    yield keyboard, probe, states
    stop.set()
    thread.join()
    executor.shutdown()
    driver.close()
    rclpy.try_shutdown()


def _settle(keyboard, *keys):
    for k in keys:
        keyboard.press(k)
        time.sleep(0.05)
    time.sleep(0.3)


def test_keys_jog_joints_within_limits_and_drive_the_gripper(rig):
    keyboard, _, states = rig
    _settle(keyboard, "q", "q", "q", "f", "s", "z")
    q = states[-1].position
    # Joint 2 starts on its lower limit (0), so "s" stays there.
    assert list(q[:6]) == pytest.approx([0.06, 0.0, 0.0, -0.02, 0.0, 0.0], abs=1e-6)
    assert q[6] == pytest.approx(1.0)  # z opens: the driver's gripper is 1 open
    _settle(keyboard, "x")
    assert states[-1].position[6] == pytest.approx(0.0)


def test_space_switches_arm(rig):
    keyboard, _, _ = rig
    assert keyboard.press(" ") == "now driving the right arm"
    assert keyboard.press(" ") == "now driving the left arm"


def test_restarts_from_the_measured_pose_when_something_else_moved_the_arm(rig):
    keyboard, probe, states = rig
    _settle(keyboard, "q")
    probe.create_publisher(Float64MultiArray, "/left/arm/command", 10).publish(
        Float64MultiArray(data=[0.1, 0.0, 0.0, 0.5, 0.0, 0.0]))
    time.sleep(0.3)
    _settle(keyboard, "r")
    # max_step clamps the 0.5 command to 0.15; the next press starts from there, not from 0.
    assert states[-1].position[3] == pytest.approx(0.15 + 0.02, abs=1e-6)
