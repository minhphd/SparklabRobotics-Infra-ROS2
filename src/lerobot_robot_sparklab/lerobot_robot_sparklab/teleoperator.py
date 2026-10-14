"""Quest teleoperation as a LeRobot ``Teleoperator`` — ``--teleop.type=bi_quest_teleop``.

A thin adapter over ``sparklab_yam.teleop.QuestTeleop``; every QuestTeleopConfig
knob is a ``--teleop.<name>`` flag. Needs the Quest bridge running.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
from rclpy.node import Node

from sparklab_yam.constants import ARM_JOINTS, HANDS
from sparklab_yam.teleop import QuestTeleop, QuestTeleopConfig

from . import ros


@TeleoperatorConfig.register_subclass("bi_quest_teleop")
@dataclass
class BiQuestTeleoperatorConfig(TeleoperatorConfig, QuestTeleopConfig):
    pass


class BiQuestTeleoperator(Teleoperator):
    config_class = BiQuestTeleoperatorConfig
    name = "bi_quest_teleop"

    def __init__(self, config: BiQuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._core: QuestTeleop | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{h}_{k}.pos": float
                for h in HANDS for k in [*(f"joint_{j}" for j in range(1, ARM_JOINTS + 1)), "gripper"]}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._core is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        ros.init()
        self._node = Node("quest_teleop")
        self._core = QuestTeleop(self.config, self._node, HANDS)
        self._close_node = ros.spin_in_background(self._node)

    def disconnect(self) -> None:
        self._close_node()
        self._core = None

    def get_action(self) -> dict[str, float]:
        return self._core.step()

    def send_feedback(self, feedback: dict) -> None:
        pass

    @property
    def quest(self):
        """The raw Quest input (``sparklab_quest.input.QuestInput``)."""
        return self._core.quest

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Re-sync the IK to `obs` and disengage. See DESIGN.md#teleoperation."""
        self._core.seed(obs)

    def xr_snapshot(self) -> dict | None:
        return self._core.xr_snapshot()
