"""The GELLO node over real topics and services, with a scripted reader."""

import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from sparklab_gello.guard import DISARMED, FAULT, GuardLimits
from sparklab_gello.node import GelloNode
from sparklab_gello.reader import GelloInputError
from sparklab_yam.driver import JOINT_NAMES, YamDriver

POSE = np.array([0.1, 0.5, 0.6, -0.2, 0.3, -0.4, 0.2])


class FakeReader:
    def __init__(self) -> None:
        self.q, self.gripper = POSE[:6].copy(), float(POSE[6])
        self.error: Exception | None = None

    def read(self):
        if self.error is not None:
            raise self.error
        return self.q.copy(), self.gripper

    def disconnect(self) -> None:
        pass


class Rig:
    """GELLO node, a probe publishing joint_states, and recorders for both arms' commands."""

    def __init__(self, arm: str, sim_driver: bool = False) -> None:
        rclpy.init(args=["--ros-args", "-p", f"arm:={arm}", "-r", "yam_driver:__ns:=/left",
                         "-p", "sim:=true", "-p", "park_on_exit:=false"])
        self.arm, self.reader = arm, FakeReader()
        self.node, self.probe = GelloNode(self.reader), Node("probe")
        self.present = POSE.copy()
        self.publish_state = True
        self.cmds = {h: [] for h in ("left", "right")}
        self.grippers = {h: [] for h in ("left", "right")}
        for h in ("left", "right"):
            self.probe.create_subscription(Float64MultiArray, f"/{h}/arm/command",
                                           lambda m, h=h: self.cmds[h].append(list(m.data)), 10)
            self.probe.create_subscription(Float64, f"/{h}/gripper/command",
                                           lambda m, h=h: self.grippers[h].append(m.data), 10)
        self.driver = YamDriver() if sim_driver else None
        if self.driver is None:
            pub = self.probe.create_publisher(JointState, f"/{arm}/joint_states", 10)
            self.probe.create_timer(0.01, lambda: self.publish_state and pub.publish(
                JointState(name=JOINT_NAMES, position=self.present.tolist())))
        self.executor = MultiThreadedExecutor()
        for n in (self.node, self.probe) + ((self.driver,) if self.driver else ()):
            self.executor.add_node(n)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=lambda: [self.executor.spin_once(timeout_sec=0.05)
                            for _ in iter(self._stop.is_set, True)])
        self._thread.start()
        time.sleep(0.3)

    def call(self, name: str) -> Trigger.Response:
        client = self.probe.create_client(Trigger, f"/gello/{name}")
        assert client.wait_for_service(timeout_sec=2.0)
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result()

    def clear(self) -> None:
        for h in ("left", "right"):
            self.cmds[h].clear()
            self.grippers[h].clear()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.executor.shutdown()
        self.node.close()
        if self.driver is not None:
            self.driver.close()
        for n in (self.node, self.probe) + ((self.driver,) if self.driver else ()):
            n.destroy_node()
        rclpy.try_shutdown()


@pytest.fixture
def rig():
    r = Rig("left")
    yield r
    r.close()


def test_starts_disarmed_and_publishes_nothing(rig):
    time.sleep(0.3)
    assert rig.node._guard.state == DISARMED
    assert rig.cmds == {"left": [], "right": []}


@pytest.mark.parametrize("arm", ["left", "right"])
def test_arm_drives_only_the_chosen_arm_in_joint_order(arm):
    r = Rig(arm)
    try:
        response = r.call("arm")
        assert response.success, response.message
        time.sleep(0.3)
        other = "right" if arm == "left" else "left"
        assert r.cmds[other] == [] and r.grippers[other] == []
        assert r.cmds[arm] and all(len(c) == 6 for c in r.cmds[arm])
        np.testing.assert_allclose(r.cmds[arm][-1], POSE[:6])
        assert r.grippers[arm][-1] == pytest.approx(POSE[6])
    finally:
        r.close()


def test_arm_is_refused_when_the_leader_does_not_match(rig):
    rig.reader.q[2] += 0.5
    response = rig.call("arm")
    assert not response.success
    assert "joint3" in response.message
    time.sleep(0.2)
    assert rig.node._guard.state == DISARMED and rig.cmds["left"] == []


def test_arm_is_refused_while_another_node_publishes_commands(rig):
    rig.probe.create_publisher(Float64MultiArray, "/left/arm/command", 10)
    time.sleep(0.3)
    response = rig.call("arm")
    assert not response.success and "another node" in response.message


def test_rate_limit_is_time_based_for_joints_and_gripper(rig):
    assert rig.call("arm").success
    rig.clear()
    rig.reader.q[0] += 1.0
    rig.reader.gripper = 1.0
    time.sleep(0.3)
    limits = GuardLimits()
    # The probe never moves, so every command is one step from the same pose.
    assert max(abs(c[0] - POSE[0]) for c in rig.cmds["left"]) <= limits.max_joint_speed * limits.max_dt + 1e-9
    assert max(abs(g - POSE[6]) for g in rig.grippers["left"]) <= limits.max_gripper_speed * limits.max_dt + 1e-9


def test_disarm_holds_the_measured_pose_once_then_stops(rig):
    assert rig.call("arm").success
    rig.present = POSE + 0.01
    time.sleep(0.1)
    response = rig.call("disarm")
    assert response.success and rig.node._guard.state == DISARMED
    time.sleep(0.05)
    rig.clear()
    time.sleep(0.3)
    assert rig.cmds["left"] == []


def test_input_error_faults_and_holds_the_measured_pose(rig):
    assert rig.call("arm").success
    time.sleep(0.1)
    rig.clear()
    rig.reader.error = GelloInputError("GELLO read failed")
    time.sleep(0.3)
    assert rig.node._guard.state == FAULT
    assert len(rig.cmds["left"]) == 1
    np.testing.assert_allclose(rig.cmds["left"][0], POSE[:6])
    assert rig.grippers["left"] == [pytest.approx(POSE[6])]
    rig.reader.error = None
    time.sleep(0.2)
    assert rig.node._guard.state == FAULT and len(rig.cmds["left"]) == 1
    assert rig.call("arm").success


def test_silent_joint_states_fault_without_commanding_the_stale_pose(rig):
    assert rig.call("arm").success
    time.sleep(0.1)
    rig.clear()
    rig.publish_state = False
    time.sleep(0.4)
    assert rig.node._guard.state == FAULT
    assert "joint_states silent" in rig.node._guard.reason
    # At most commands sent before the timeout; never a hold built from the stale pose.
    assert len(rig.cmds["left"]) <= 12
    response = rig.call("arm")
    assert not response.success and "no fresh" in response.message


def test_end_to_end_with_the_sim_driver():
    r = Rig("left", sim_driver=True)
    try:
        r.reader.q, r.reader.gripper = np.zeros(6), 0.0
        response = r.call("arm")
        assert response.success, response.message
        target = np.array([0.3, 0.4, 0.5, -0.2, 0.25, -0.3])
        r.reader.q, r.reader.gripper = target.copy(), 1.0
        time.sleep(1.0)
        measured = np.asarray(r.driver._robot.get_joint_pos())
        np.testing.assert_allclose(measured[:6], target, atol=1e-6)
        assert measured[6] == pytest.approx(1.0)
        qpos = r.driver._robot._data.qpos
        assert qpos[6] > 0.0 and qpos[7] == pytest.approx(qpos[6])
    finally:
        r.close()
