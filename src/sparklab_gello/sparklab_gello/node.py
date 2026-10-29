"""GELLO joint teleop of one YAM arm through the YAM driver's command topics.

    ros2 run sparklab_gello gello --ros-args -p arm:=left
    ros2 service call /gello/arm std_srvs/srv/Trigger
    ros2 service call /gello/disarm std_srvs/srv/Trigger

    /<arm>/joint_states     in   measured pose; silent for state_timeout_s faults
    /<arm>/arm/command      out  6 joint targets (rad), joint_1..6
    /<arm>/gripper/command  out  gripper target, 0 closed .. 1 open
    ~/arm, ~/disarm         srv  std_srvs/Trigger

Starts DISARMED and publishes nothing until ``~/arm``; the next GELLO reading
must then match the measured pose. Leaving ARMED publishes the measured pose
once, so the driver holds there. If this node dies, the driver keeps its last target.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from sparklab_yam.constants import ARM_JOINTS, HANDS
from sparklab_yam.run import run

from .guard import ARMED, FAULT, ArmGuard, GuardLimits
from .reader import GelloInputError, GelloReader

ARM_TIMEOUT_S = 1.0


class GelloNode(Node):
    """reader: a connected GelloReader; built from parameters and connected when None."""

    def __init__(self, reader: GelloReader | None = None) -> None:
        super().__init__("gello")
        self._arm = self.declare_parameter("arm", "left").value
        if self._arm not in HANDS:
            raise ValueError(f"arm must be one of {HANDS}, got {self._arm!r}")
        rate_hz = self.declare_parameter("rate_hz", 100.0).value
        self._state_timeout = self.declare_parameter("state_timeout_s", 0.1).value
        defaults = GuardLimits()
        limits = GuardLimits(**{k: float(self.declare_parameter(k, getattr(defaults, k)).value)
                                for k in GuardLimits.__dataclass_fields__})
        if reader is None:
            config = Path(get_package_share_directory("sparklab_gello")) / "config"
            reader = GelloReader(
                self.declare_parameter(
                    "calibration_path", str(config / "yam_active_leader_eeprom.json")).value,
                self.declare_parameter("mapping_path", str(config / "gello_to_yam.yaml")).value,
                port=self.declare_parameter("port", "").value,
                read_retries=self.declare_parameter("read_retries", 1).value,
                max_read_latency_s=self.declare_parameter("max_read_latency_s", 0.1).value,
                max_input_speed_deg_s=self.declare_parameter("max_input_speed_deg_s", 720.0).value,
                range_margin_deg=self.declare_parameter("range_margin_deg", 10.0).value)
            port = reader.connect()
            self.get_logger().info(f"GELLO on {port} ({reader.calibration_path.name})")
        self._reader = reader
        self._guard = ArmGuard(ARM_JOINTS, limits)
        self._lock = threading.Lock()
        self._present: np.ndarray | None = None
        self._present_t = 0.0

        self._arm_topic = f"/{self._arm}/arm/command"
        self._gripper_topic = f"/{self._arm}/gripper/command"
        self._arm_pub = self.create_publisher(Float64MultiArray, self._arm_topic, 10)
        self._gripper_pub = self.create_publisher(Float64, self._gripper_topic, 10)
        # The arm service waits for a tick, so the tick needs its own group.
        group = ReentrantCallbackGroup()
        self.create_subscription(JointState, f"/{self._arm}/joint_states", self._on_state, 10,
                                 callback_group=group)
        self.create_service(Trigger, "~/arm", self._on_arm, callback_group=group)
        self.create_service(Trigger, "~/disarm", self._on_disarm, callback_group=group)
        self.create_timer(1.0 / rate_hz, self._tick,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"DISARMED; driving the {self._arm} arm after "
                               f"`ros2 service call {self.get_fully_qualified_name()}/arm std_srvs/srv/Trigger`")

    # ---- state ------------------------------------------------------------
    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            self._present = np.asarray(msg.position[:ARM_JOINTS + 1], dtype=float)
            self._present_t = time.monotonic()

    def _fresh(self) -> bool:
        return (self._present is not None
                and time.monotonic() - self._present_t <= self._state_timeout)

    def _publish(self, q: np.ndarray) -> None:
        self._arm_pub.publish(Float64MultiArray(data=q[:ARM_JOINTS].tolist()))
        self._gripper_pub.publish(Float64(data=float(q[ARM_JOINTS])))

    def _left_armed(self, before: str, hold: bool) -> None:
        """After a transition: on leaving ARMED, command the hold once. Call with the lock held."""
        g = self._guard
        if before == g.state:
            return
        self.get_logger().warning(f"{before} -> {g.state}: {g.reason}")
        if before != ARMED:
            return
        if hold:
            self._publish(g.hold_target)
        else:
            self.get_logger().error("no fresh joint_states to hold; the driver keeps its last target")

    def _fault(self, reason: str) -> None:
        """Fault, holding the measured pose if it is fresh. Call with the lock held."""
        g, before, fresh = self._guard, self._guard.state, self._fresh()
        if g.pending_since is not None:
            g.arm_result = f"arm rejected: {reason}"
        # A stale pose is recorded as the hold target but never commanded.
        g.fault(reason, self._present)
        self._left_armed(before, hold=fresh)
        self.get_logger().error(reason, throttle_duration_sec=1.0)

    # ---- loop -------------------------------------------------------------
    def _tick(self) -> None:
        with self._lock:
            if self._present is None:
                return
        try:
            q, gripper = self._reader.read()
        except GelloInputError as e:
            with self._lock:
                self._fault(str(e))
            return
        with self._lock:
            if not self._fresh():
                if self._guard.state == ARMED or self._guard.pending_since is not None:
                    self._fault(f"/{self._arm}/joint_states silent for "
                                f"{time.monotonic() - self._present_t:.2f} s")
                return
            before = self._guard.state
            self._guard.check_timeout(self._present)
            cmd, execute = self._guard.filter(np.append(q, gripper), self._present)
            if execute:
                self._publish(cmd)
            self._left_armed(before, hold=True)

    # ---- services ---------------------------------------------------------
    def _on_arm(self, _request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            others = [t for t in (self._arm_topic, self._gripper_topic)
                      if self.count_publishers(t) > 1]
            if others:
                self._guard.arm_result = (f"arm rejected: another node publishes {others}; "
                                          "stop it (Quest teleop, keyboard) first")
            elif not self._fresh():
                self._guard.arm_result = f"arm rejected: no fresh /{self._arm}/joint_states"
            else:
                self._guard.request_arm(self._present)
        deadline = time.monotonic() + ARM_TIMEOUT_S
        while self._guard.pending_since is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        with self._lock:
            # Never leave a request pending past the reply: it could arm unannounced later.
            if self._guard.pending_since is not None:
                self._guard.pending_since = None
                self._guard.arm_result = f"arm rejected: no GELLO reading within {ARM_TIMEOUT_S} s"
            response.success = self._guard.state == ARMED
            response.message = self._guard.arm_result
        # rclpy refuses two severities from one call site.
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warning(response.message)
        return response

    def _on_disarm(self, _request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._disarm("disarm requested")
            response.success = True
            response.message = f"{self._guard.state}: {self._guard.reason}"
        return response

    def _disarm(self, reason: str) -> None:
        """Call with the lock held."""
        before = self._guard.state
        if self._present is not None:
            self._guard.disarm(reason, self._present)
        self._left_armed(before, hold=self._fresh())

    def close(self) -> None:
        with self._lock:
            self._disarm("node shutdown")
        self._reader.disconnect()


def main() -> None:
    run(GelloNode)


if __name__ == "__main__":
    main()
