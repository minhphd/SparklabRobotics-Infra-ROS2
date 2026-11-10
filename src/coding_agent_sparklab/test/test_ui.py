"""The page's Stop against two simulated drivers: it kills the agents, then parks both arms."""

import threading
import time

import numpy as np
from fastapi.testclient import TestClient
from rclpy.executors import SingleThreadedExecutor

from coding_agent_sparklab import robot
from coding_agent_sparklab.agents import Chat
from coding_agent_sparklab.ui import Rig, serve


class Recorder:
    label = "fake"

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_stop_kills_every_agent_and_parks_both_arms(drivers, tmp_path):
    robot.command("left", [0.3, 0.6, 0.6, 0.0, 0.2, 0.0])
    robot.command("right", [-0.3, 0.6, 0.6, 0.0, -0.2, 0.0])
    rig = Rig()
    executor = SingleThreadedExecutor()
    executor.add_node(rig)
    stop = threading.Event()
    thread = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    try:
        chat, agents = Chat(), {"a": Recorder(), "b": Recorder()}
        time.sleep(0.5)
        assert TestClient(serve(rig, chat, agents, tmp_path)).post("/api/stop").json() == {"ok": True}
    finally:
        stop.set()
        thread.join()
        executor.shutdown()
        rig.destroy_node()
    assert all(a.stopped for a in agents.values())
    assert [i["text"] for i in chat.items] == ["Stopped. Parking both arms", "Both arms parked"]
    assert not chat.busy
    for arm in robot.ARMS:
        np.testing.assert_allclose(robot.get_joint_angles(arm)[:6], 0.0, atol=0.03)
