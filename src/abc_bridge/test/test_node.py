"""The episode loop against a fake policy and rig: no ROS, no server."""

from types import SimpleNamespace

import numpy as np
import pytest

from abc_bridge import node


class Rig:
    """Stands in for Bridge: both arms at zero, recording what it is told to publish."""

    def __init__(self):
        self.sent = []

    def get_logger(self):
        return SimpleNamespace(info=print, warning=print)

    def observation(self):
        return {"state": np.zeros(14, np.float32), "prompt": "test", "images": {}}

    def state(self):
        return np.zeros(14, np.float32)

    def publish(self, action):
        self.sent.append(action)

    def record_request(self, obs):
        pass

    def record_reply(self, actions):
        pass


def test_a_target_far_from_the_measured_joint_stops_the_episode_before_it_is_sent(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    actions = np.zeros((30, 14), np.float32)
    actions[node.PREFIX:, 6] = 1.0                   # the left gripper opening is not an arm joint
    actions[node.PREFIX + 3, 2] = node.MAX_LEAD + 0.1  # left joint 3, on the fourth step
    rig = Rig()
    with pytest.raises(RuntimeError, match="left joint 3"):
        node.run_episode(rig, SimpleNamespace(infer=lambda obs: actions))
    assert len(rig.sent) == 3
