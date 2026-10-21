"""The shipped cap-x config builds a session on two simulated arms, as cap-x's agent loop does."""

import os
import subprocess
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.executors import SingleThreadedExecutor

import capx.integrations.franka.libero_sparklab_v2 as surface
from capx.agent.simple_loop import load_config
from capx.envs.configs.instantiate import instantiate
from capx_sparklab import CONFIG
from capx_sparklab.api import _Printed
from test_env import Rig


@pytest.fixture(scope="module")
def session():
    os.environ.update(ROS_DOMAIN_ID="90", ROS_LOCALHOST_ONLY="1")
    driver = Path(get_package_prefix("sparklab_yam")) / "lib" / "sparklab_yam" / "driver"
    drivers = [subprocess.Popen([str(driver), "--ros-args", "-r", f"__ns:=/{arm}", "-r", "__node:=yam_driver",
                                 "-p", "sim:=true", "-p", "park_on_exit:=false"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for arm in ("left", "right")]
    rclpy.init()
    rig = Rig()
    executor = SingleThreadedExecutor()
    executor.add_node(rig)
    stop = threading.Event()
    thread = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    try:
        time.sleep(1.0)
        factory, multi_turn, servers = load_config(str(CONFIG))
        factory["cfg"]["low_level"]["viser_debug"] = True  # as cap-x's ReAct loop sets it
        with mock.patch.object(surface, "Voice", _Printed):
            env = instantiate(factory)
        yield env, multi_turn, servers
        env.low_level_env.close()
    finally:
        stop.set()
        thread.join()
        executor.shutdown()
        rig.destroy_node()
        rclpy.try_shutdown()
        for p in drivers:
            p.terminate()
            p.wait(timeout=10)


def test_session_prompt_documents_both_arms_and_handover(session):
    env, multi_turn, servers = session
    obs, _ = env.reset()
    prompt = obs["full_prompt"][1]["content"][0]["text"]
    assert "THE TWO ARMS" in prompt and "handover" in prompt and "arm_id" in prompt
    assert "{executed_code}" in multi_turn
    assert servers[0]["port"] == 8114


def test_session_runs_agent_code_against_the_tools(session):
    env, _, _ = session
    env.reset()
    _, _, _, _, info = env.step("state = get_state()\nprint(sorted(state['data']['arms']))")
    assert "['left', 'right']" in info["stdout"]


def test_browser_panel_shows_a_rig_camera(session):
    from capx.agent import flask_ui
    from capx_sparklab.ui import panel_encoder

    obs = session[0].low_level_env.get_observation()
    assert flask_ui.encode_frame(obs) is None  # cap-x's own looks for LIBERO's agentview
    assert panel_encoder("top")(obs).startswith("data:image/jpeg;base64,")


def test_react_loop_draws_into_the_ros_scene(session):
    import capx_sparklab.agent  # noqa: F401  (the ReAct launcher imports)

    env = session[0]
    assert env._apis["YamSparkLabApi"]._view is env.low_level_env.scene
