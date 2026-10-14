"""Latest Quest headset and controller state, from the bridge's ROS topics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Joy

from . import buttons


@dataclass
class Controller:
    """One controller in Quest ``local-floor`` (Y-up), as of its last pose."""

    pos: np.ndarray
    quat_xyzw: np.ndarray
    buttons: list[int]
    axes: list[float]
    age_s: float


def _pose(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    p, q = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])


class QuestInput:
    """Subscribes ``/quest/{head,<hand>}/pose`` and ``/quest/<hand>/joy`` on ``node``."""

    def __init__(self, node: Node, hands: tuple[str, ...] = ("left", "right")) -> None:
        self._node = node
        self._hands = hands
        self._msgs: dict[str, object] = {}
        topics = [("head/pose", PoseStamped)]
        topics += [(f"{h}/{kind}", t) for h in hands for kind, t in (("pose", PoseStamped), ("joy", Joy))]
        for name, msg_type in topics:
            node.create_subscription(
                msg_type, f"/quest/{name}", lambda m, k=name: self._msgs.__setitem__(k, m), 10)

    def head(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Headset (position, quat xyzw), or None before the first frame."""
        msg = self._msgs.get("head/pose")
        return None if msg is None else _pose(msg)

    def controller(self, hand: str) -> Controller | None:
        pose = self._msgs.get(f"{hand}/pose")
        if pose is None:
            return None
        joy = self._msgs.get(f"{hand}/joy")
        pos, quat = _pose(pose)
        return Controller(pos, quat,
                          list(joy.buttons) if joy else [], list(joy.axes) if joy else [],
                          (self._node.get_clock().now() - Time.from_msg(pose.header.stamp)).nanoseconds * 1e-9)

    def pressed(self, index: int) -> bool:
        """Whether button `index` is held on any subscribed controller."""
        return any(c is not None and buttons.pressed(c.buttons, index)
                   for c in map(self.controller, self._hands))
