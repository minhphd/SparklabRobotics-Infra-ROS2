"""Quest teleoperation of the YAM-Ultra arms, as a ROS 2 node.

    ros2 run sparklab_yam teleop
    ros2 run sparklab_yam teleop --ros-args -p hands:="[right]"

The pose each arm has when its joint_states first arrive is HOME. Starts
DISARMED. B/Y on either hand arms it, seeded from the measured pose; B/Y again
disarms and ramps the arms home. Grip clutches, trigger drives the gripper, A/X
is precision, thumbstick ramps that arm home. SIGINT/SIGTERM ramps home first.
"""

from __future__ import annotations

import time

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from sparklab_quest import buttons

from .constants import ARM_JOINTS
from .run import run
from .teleop import QuestTeleop, QuestTeleopConfig


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("yam_teleop")
        self._hands = tuple(self.declare_parameter("hands", ["left", "right"]).value)
        rate_hz = self.declare_parameter("rate_hz", 100.0).value
        self._home_ramp_s = self.declare_parameter("home_ramp_s", 5.0).value
        self._home_on_exit = self.declare_parameter("home_on_exit", True).value
        dq_pos = self.declare_parameter("max_dq_pos", 0.04).value
        dq_rot = self.declare_parameter("max_dq_rot", 0.15).value
        self._cfg = QuestTeleopConfig(
            max_dq_per_joint=[dq_pos] * 3 + [dq_rot] * 3,
            max_dq_per_joint_scalar_pos=dq_pos,
            max_dq_per_joint_scalar_rot=dq_rot,
            scale_translation=self.declare_parameter("scale_translation", 0.8).value,
            scale_rotation=self.declare_parameter("scale_rotation", 0.8).value,
            rest_ramp_duration_s=self.declare_parameter("rest_ramp_s", 4.0).value,
        )

        self._state: dict[str, JointState] = {}
        self._arm_pubs, self._gripper_pubs = {}, {}
        for h in self._hands:
            self.create_subscription(
                JointState, f"/{h}/joint_states", lambda m, h=h: self._state.__setitem__(h, m), 10)
            self._arm_pubs[h] = self.create_publisher(Float64MultiArray, f"/{h}/arm/command", 10)
            self._gripper_pubs[h] = self.create_publisher(Float64, f"/{h}/gripper/command", 10)

        self._teleop: QuestTeleop | None = None
        self._home: dict[str, np.ndarray] = {}
        self._gripper: dict[str, float] = {}
        self._armed = False
        self._last_handoff = False
        # Its own group: a disarm ramp blocks the tick, and joint_states must
        # keep arriving meanwhile.
        self.create_timer(1.0 / rate_hz, self._tick,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"waiting for joint_states from {', '.join(self._hands)}")

    def _obs(self) -> dict[str, float]:
        obs = {}
        for h in self._hands:
            q = self._state[h].position
            obs.update({f"{h}_joint_{j + 1}.pos": q[j] for j in range(ARM_JOINTS)})
            obs[f"{h}_gripper.pos"] = q[ARM_JOINTS]
        return obs

    def _publish(self, h: str, arm: np.ndarray, gripper: float) -> None:
        self._arm_pubs[h].publish(Float64MultiArray(data=[float(v) for v in arm]))
        self._gripper_pubs[h].publish(Float64(data=float(gripper)))

    def _seed(self) -> None:
        obs = self._obs()
        self._teleop.seed(obs)
        self._gripper = {h: obs[f"{h}_gripper.pos"] for h in self._hands}

    def _start(self) -> None:
        """Capture HOME and build the teleop once every arm has reported."""
        self._home = {h: np.asarray(self._state[h].position[:ARM_JOINTS]) for h in self._hands}
        for h, q in self._home.items():
            self.get_logger().info(f"{h} HOME (pose at startup): {np.round(q, 3).tolist()}")
            setattr(self._cfg, f"rest_qpos_{h}", q.tolist())
        self._teleop = QuestTeleop(self._cfg, self, self._hands)
        self._seed()
        self.get_logger().info("DISARMED, holding HOME — press B/Y to arm")

    def _move_home(self) -> None:
        """Smoothstep every arm to HOME over home_ramp_s, holding the gripper. Blocks."""
        start = {h: np.asarray(self._state[h].position[:ARM_JOINTS]) for h in self._hands}
        steps = max(1, int(self._home_ramp_s * 50))
        for i in range(steps + 1):
            a = i / steps
            a = a * a * (3.0 - 2.0 * a)
            for h in self._hands:
                self._publish(h, (1.0 - a) * start[h] + a * self._home[h], self._gripper[h])
            time.sleep(self._home_ramp_s / steps)

    def _tick(self) -> None:
        if self._teleop is None:
            if all(h in self._state for h in self._hands):
                self._start()
            return

        action = self._teleop.step()
        handoff = self._teleop.quest.pressed(buttons.B_Y)
        if handoff and not self._last_handoff:
            self._last_handoff = True
            if not self._armed:
                self._seed()
                self._armed = True
                self.get_logger().info("ARMED — grip to engage; B/Y to disarm")
            else:
                self._armed = False
                self.get_logger().info(f"DISARMING: ramping home over {self._home_ramp_s:.1f}s")
                self._move_home()
                self._seed()
                self.get_logger().info("DISARMED, holding HOME — B/Y to re-arm")
            # The action in hand predates the seed — never command it.
            return
        self._last_handoff = handoff

        if self._armed:
            for h in self._hands:
                self._gripper[h] = action[f"{h}_gripper.pos"]
                arm = [action[f"{h}_joint_{j}.pos"] for j in range(1, ARM_JOINTS + 1)]
                self._publish(h, np.asarray(arm), self._gripper[h])

    def close(self) -> None:
        if self._teleop is not None and self._home_on_exit:
            self.get_logger().info("ramping home before exit")
            self._move_home()


def main() -> None:
    run(TeleopNode)


if __name__ == "__main__":
    main()
