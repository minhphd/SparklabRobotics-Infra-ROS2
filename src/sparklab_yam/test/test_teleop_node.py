"""The teleop node against a simulated right arm."""

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from sparklab_yam.driver import YamDriver
from sparklab_yam.model.kinematics import DEFAULT_Q_REST
from sparklab_yam.teleop_node import TeleopNode


@pytest.fixture
def rig():
    rclpy.init(args=["--ros-args", "-r", "__ns:=/right", "-p", "sim:=true", "-p", "park_on_exit:=false",
                     "-p", "hands:=[right]", "-p", "park_duration_s:=1.0", "-p", "rest_on_start:=true"])
    driver, teleop, probe = YamDriver(), TeleopNode(), Node("probe")
    # Multi-threaded, as run() spins it: a home ramp blocks the node's tick while the driver keeps moving.
    executor = MultiThreadedExecutor()
    for n in (driver, teleop, probe):
        executor.add_node(n)
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    states: list[JointState] = []
    probe.create_subscription(JointState, "/right/joint_states", states.append, 10)
    yield teleop, states
    stop.set()
    thread.join()
    executor.shutdown()
    driver.close()
    rclpy.try_shutdown()


def test_starts_disarmed_with_home_at_the_rest_pose_the_driver_ramps_to(rig):
    teleop, states = rig
    time.sleep(2.0)
    assert teleop._teleop is not None and not teleop._armed
    np.testing.assert_allclose(teleop._home["right"], DEFAULT_Q_REST)
    assert np.max(np.abs(np.asarray(states[-1].position[:6]) - DEFAULT_Q_REST)) < 0.03
