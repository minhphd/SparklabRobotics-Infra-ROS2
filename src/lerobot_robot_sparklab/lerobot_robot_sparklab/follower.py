"""The bimanual YAM-Ultra as a LeRobot ``Robot`` — ``--robot.type=yam_ultra_bimanual``.

A ROS 2 client of the sparklab_yam drivers and the camera node; it opens no
device itself, so start the rig first::

    ros2 launch sparklab_yam yam.launch.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from sparklab_yam.constants import ARM_JOINTS, DEFAULT_MAX_RELATIVE_TARGET, HANDS

from . import ros

logger = logging.getLogger(__name__)

# A driver whose control loop died stops publishing, and a stale frame shows
# the policy a scene that no longer exists; both are refused past this age.
STALE_S = 0.5


@cache
def _rig_cameras() -> dict[str, tuple[int, int, int]]:
    """id -> (H, W, 3) after crop, from the rig file the camera node also reads."""
    rig = Path(get_package_share_directory("sparklab_yam")) / "config" / "cameras.yaml"
    cams = yaml.safe_load(rig.read_text())["cameras"] or {}
    return {c: (min(cfg.get("crop_height") or cfg.get("height", 480), cfg.get("height", 480)),
                cfg.get("width", 640), 3)
            for c, cfg in cams.items()}


@RobotConfig.register_subclass("yam_ultra_bimanual")
@dataclass
class YamUltraFollowerConfig(RobotConfig):
    """``--robot.type=yam_ultra_bimanual`` in any LeRobot CLI."""

    max_relative_target: float | list[float] | None = field(
        default_factory=lambda: list(DEFAULT_MAX_RELATIVE_TARGET)
    )
    # Ramp the arms to the zero pose on disconnect, where torque is safe to cut.
    park_on_disconnect: bool = True
    # Ids from the rig's cameras.yaml, each read from /cameras/<id>/color/image_raw.
    # Not `cameras`: RobotConfig reserves that for LeRobot CameraConfigs.
    camera_ids: list[str] = field(default_factory=lambda: list(_rig_cameras()))
    connect_timeout_s: float = 10.0


class YamUltraFollower(Robot):
    """Two YAM-Ultra arms + N cameras as one LeRobot Robot."""

    config_class = YamUltraFollowerConfig
    name = "yam_ultra_bimanual"

    _ARM_KEYS = {h: tuple(f"{h}_joint_{j}.pos" for j in range(1, ARM_JOINTS + 1)) for h in HANDS}
    _GRIPPER_KEYS = {h: f"{h}_gripper.pos" for h in HANDS}

    def __init__(self, config: YamUltraFollowerConfig):
        super().__init__(config)
        self.config = config
        # LeRobot sizes its image writers by len(robot.cameras).
        self.cameras = {c: _rig_cameras()[c] for c in config.camera_ids}
        mrt = config.max_relative_target
        self._caps = (None if mrt is None
                      else np.broadcast_to(np.asarray(mrt, dtype=float), (ARM_JOINTS,)).copy())
        self._node: Node | None = None
        self._state: dict[str, JointState] = {}
        self._images: dict[str, Image] = {}
        # send_action seeds from this, so a joint the action omits holds.
        self._last_sent: dict[str, np.ndarray] = {}
        self._last_clamp_warn: dict[str, float] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self.action_features, **self.cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {k: float for h in HANDS for k in (*self._ARM_KEYS[h], self._GRIPPER_KEYS[h])}

    # ---- connection ---------------------------------------------------------
    def _age(self, msg) -> float:
        return (self._node.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9

    def _stale(self) -> list[str]:
        """Topics with no message, or none within STALE_S."""
        out = [f"/{h}/joint_states" for h in HANDS
               if h not in self._state or self._age(self._state[h]) > STALE_S]
        out += [f"/cameras/{c}/color/image_raw" for c in self.cameras
                if c not in self._images or self._age(self._images[c]) > STALE_S]
        return out

    @property
    def is_connected(self) -> bool:
        return self._node is not None and not self._stale()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:  # i2rt homes its own encoders on connect
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._node is not None:
            raise RuntimeError(f"{self} is already connected")
        ros.init()
        node = self._node = Node("yam_follower")
        self._arm_pubs, self._gripper_pubs, self._park = {}, {}, {}
        for h in HANDS:
            node.create_subscription(JointState, f"/{h}/joint_states",
                                     lambda m, h=h: self._state.__setitem__(h, m), 10)
            self._arm_pubs[h] = node.create_publisher(Float64MultiArray, f"/{h}/arm/command", 10)
            self._gripper_pubs[h] = node.create_publisher(Float64, f"/{h}/gripper/command", 10)
            self._park[h] = node.create_client(Trigger, f"/{h}/park")
        for c in self.cameras:
            node.create_subscription(Image, f"/cameras/{c}/color/image_raw",
                                     lambda m, c=c: self._images.__setitem__(c, m), 1)
        self._close_node = ros.spin_in_background(node)

        deadline = time.monotonic() + self.config.connect_timeout_s
        while (missing := self._stale()) and time.monotonic() < deadline:
            time.sleep(0.05)
        if missing:
            self.disconnect()
            raise DeviceNotConnectedError(
                f"{self}: nothing fresh on {', '.join(missing)} after "
                f"{self.config.connect_timeout_s:.0f}s. Start the rig first:\n"
                f"    ros2 launch sparklab_yam yam.launch.py")
        for h in HANDS:
            # Two drivers on one arm would both act on every command.
            if (n := node.count_publishers(f"/{h}/joint_states")) > 1:
                self.disconnect()
                raise RuntimeError(f"{n} publishers on /{h}/joint_states — stop the extra driver")
            logger.info("%s pose at connect: %s (park target is zero)",
                        h, np.round(self._state[h].position[:ARM_JOINTS], 3).tolist())
        logger.info("%s connected (%d cameras).", self, len(self.cameras))

    def disconnect(self) -> None:
        """Park if configured, then drop the node. The drivers keep running."""
        if self._node is None:
            return
        # A dead camera must not stop the arms parking; a dead driver cannot park.
        arms_live = not any(t.endswith("joint_states") for t in self._stale())
        if self.config.park_on_disconnect and arms_live:
            self.park()
        self._close_node()
        self._node = None
        self._state.clear()
        self._images.clear()
        logger.info("%s disconnected.", self)

    def _require_fresh(self, where: str) -> None:
        if self._node is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")
        if missing := self._stale():
            raise RuntimeError(
                f"{self}: nothing fresh on {', '.join(missing)} (at {where}). A silent "
                "joint_states means the driver stopped or its CAN control loop died — "
                "check its log for 'loss communication' and the E-stop.")

    # ---- I/O ----------------------------------------------------------------
    def get_observation(self) -> dict:
        self._require_fresh("get_observation")
        obs: dict[str, object] = {}
        for h in HANDS:
            q = self._state[h].position
            obs.update(zip(self._ARM_KEYS[h], q[:ARM_JOINTS]))
            obs[self._GRIPPER_KEYS[h]] = q[ARM_JOINTS]
        for c in self.cameras:
            msg = self._images[c]
            obs[c] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        return obs

    def send_action(self, action: dict) -> dict:
        """Command both arms.

        The driver clamps every command to its max_step from the measured pose;
        clamping here too, the same way, makes the returned action the one the
        arm was actually told to take.

        action: `.pos` keys; omitted joints hold their last commanded value.
        Returns: the Δq-clamped action sent, in the same keys.
        """
        self._require_fresh("send_action")
        sent: dict[str, float] = {}
        for h in HANDS:
            present = np.asarray(self._state[h].position[:ARM_JOINTS])
            target = self._last_sent.get(h, present).copy()
            for i, key in enumerate(self._ARM_KEYS[h]):
                if key in action:
                    target[i] = float(action[key])
            if self._caps is not None:
                step = target - present
                target = present + np.clip(step, -self._caps, self._caps)
                self._warn_clamp(h, np.abs(step) - self._caps)
            self._last_sent[h] = target
            self._arm_pubs[h].publish(Float64MultiArray(data=target.tolist()))
            sent.update(zip(self._ARM_KEYS[h], target.tolist()))

            gkey = self._GRIPPER_KEYS[h]
            sent[gkey] = float(action.get(gkey, self._state[h].position[ARM_JOINTS]))
            if gkey in action:
                self._gripper_pubs[h].publish(Float64(data=sent[gkey]))
        return sent

    def _warn_clamp(self, hand: str, overshoot: np.ndarray) -> None:
        # Rate-limited: sustained clamping is normal under teleop.
        now = time.monotonic()
        if np.any(overshoot > 0) and now - self._last_clamp_warn.get(hand, 0.0) > 1.0:
            self._last_clamp_warn[hand] = now
            j = int(np.argmax(overshoot))
            logger.warning("%s joint_%d exceeded the Δq cap %.3f rad by %.3f — clamped "
                           "(suppressed for 1s)", hand, j + 1, self._caps[j], overshoot[j])

    def park(self) -> None:
        """Ramp both arms to the zero pose and hold, where torque is safe to cut.

        Blocks for the drivers' park_duration_s; both start before either is awaited.
        """
        logger.info("parking both arms to zero")
        futures = {h: self._park[h].call_async(Trigger.Request()) for h in HANDS}
        deadline = time.monotonic() + 30.0
        while not all(f.done() for f in futures.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        for h, f in futures.items():
            if not f.done() or not f.result().success:
                logger.warning("%s arm did not park — it drops when torque is cut", h)
        self._last_sent.clear()
