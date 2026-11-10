"""The pair node against two scripted per-arm GELLO services."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from sparklab_gello.pair import GelloPair

# Never the rig's domain: on it, /gello/<arm>/arm would arm a running launch's GELLO nodes.
DOMAIN = 91


class FakeArm:
    """One GELLO node's ~/arm and ~/disarm, accepting or refusing as told."""

    def __init__(self, node: Node, hand: str, arms: bool = True) -> None:
        self.arms, self.armed = arms, False
        node.create_service(Trigger, f"/gello/{hand}/arm", self._arm)
        node.create_service(Trigger, f"/gello/{hand}/disarm", self._disarm)

    def _arm(self, _request, response):
        self.armed = response.success = self.arms
        response.message = "ARMED" if self.arms else "arm rejected: joint3 is off"
        return response

    def _disarm(self, _request, response):
        self.armed = False
        response.success, response.message = True, "DISARMED"
        return response


@pytest.fixture
def ros():
    rclpy.init(domain_id=DOMAIN)
    fake, probe, pair = Node("fake_gello"), Node("probe"), GelloPair()
    executor = MultiThreadedExecutor()
    for n in (fake, probe, pair):
        executor.add_node(n)
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    yield fake, probe
    stop.set()
    thread.join()
    executor.shutdown()
    for n in (fake, probe, pair):
        n.destroy_node()
    rclpy.try_shutdown()


def _call(probe: Node, op: str) -> Trigger.Response:
    client = probe.create_client(Trigger, f"/gello/{op}")
    assert client.wait_for_service(timeout_sec=2.0)
    future = client.call_async(Trigger.Request())
    deadline = time.monotonic() + 5.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return future.result()


def test_arm_arms_both_and_disarm_stops_both(ros):
    fake, probe = ros
    left, right = FakeArm(fake, "left"), FakeArm(fake, "right")
    time.sleep(0.3)
    assert _call(probe, "arm").success and left.armed and right.armed
    assert _call(probe, "disarm").success and not left.armed and not right.armed


def test_one_refusal_disarms_both(ros):
    fake, probe = ros
    left, right = FakeArm(fake, "left"), FakeArm(fake, "right", arms=False)
    time.sleep(0.3)
    reply = _call(probe, "arm")
    assert not reply.success and not left.armed and not right.armed
    assert "right: arm rejected: joint3 is off" in reply.message and "disarmed both again" in reply.message


def test_a_missing_gello_node_is_a_refusal(ros):
    fake, probe = ros
    left = FakeArm(fake, "left")
    time.sleep(0.3)
    reply = _call(probe, "arm")
    assert not reply.success and not left.armed and "no reply from /gello/right/arm" in reply.message
