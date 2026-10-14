"""LeRobot plugin for SparkLab's rigs, over ROS 2.

    --robot.type=yam_ultra_bimanual   the real arms, via the sparklab_yam drivers
    --robot.type=yam_ultra_sim        the Isaac twin, over HTTP
    --teleop.type=bi_quest_teleop     Quest teleop, via the sparklab_quest bridge

The ``lerobot_robot_`` distribution prefix and these re-exports are both
load-bearing for plugin discovery — see DESIGN.md before changing either.
"""

from .follower import YamUltraFollower, YamUltraFollowerConfig
from .sim_follower import YamUltraSim, YamUltraSimConfig
from .teleoperator import BiQuestTeleoperator, BiQuestTeleoperatorConfig

__all__ = ["BiQuestTeleoperator", "BiQuestTeleoperatorConfig", "YamUltraFollower",
           "YamUltraFollowerConfig", "YamUltraSim", "YamUltraSimConfig"]
