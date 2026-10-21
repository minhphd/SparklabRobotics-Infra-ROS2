"""YamSparkLabApi on two simulated arms: arm addressing, planned motion, attach and handover.

Planning is real cuRobo on the GPU; perception and GraspGenX are stubbed, since
they need their servers. Needs cap-x's stack:
./scripts/capx_python.sh -m pytest src/capx_sparklab/test
"""

import array
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.executors import SingleThreadedExecutor
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

import capx.integrations.franka.libero_sparklab_v2 as surface
from capx_sparklab.env import YamEnv
from test_env import EXTRINSIC, SIZES, Rig

TABLE_Z = -0.015


class TableRig(Rig):
    """The calibrated TF and a top camera that sees a bare table."""

    def __init__(self) -> None:
        super().__init__()
        self._pubs = {"top": self._pubs["top"]}
        width, height, k = SIZES["top"]
        K3 = np.array(k).reshape(3, 3)
        v, u = np.mgrid[0:height, 0:width]
        rays = np.stack([(u - K3[0, 2]) / K3[0, 0], (v - K3[1, 2]) / K3[1, 1], np.ones_like(u, float)], -1)
        T = EXTRINSIC["top"]
        down = rays @ T[:3, :3].T
        depth = (TABLE_Z - T[2, 3]) / down[..., 2]
        self._depth["top"] = array.array("B", np.round(np.clip(depth, 0, 6) * 1000).astype(np.uint16).tobytes())


def _fingers_down(position):
    # Pitched straight down from the parked orientation; the half-turn-rolled pose is out of the wrist's range.
    x, y, z, w = Rotation.from_euler("y", np.pi).as_quat()
    return {"position": list(position), "quaternion_xyzw": [x, y, z, w]}


@pytest.fixture(scope="module")
def api():
    os.environ.update(ROS_DOMAIN_ID="89", ROS_LOCALHOST_ONLY="1")
    driver = Path(get_package_prefix("sparklab_yam")) / "lib" / "sparklab_yam" / "driver"
    drivers = [subprocess.Popen([str(driver), "--ros-args", "-r", f"__ns:=/{arm}", "-r", "__node:=yam_driver",
                                 "-p", "sim:=true", "-p", "park_on_exit:=false"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for arm in ("left", "right")]
    rclpy.init()
    rig = TableRig()
    executor = SingleThreadedExecutor()
    executor.add_node(rig)
    stop = threading.Event()
    thread = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    try:
        env = YamEnv(cameras=("top",))
        time.sleep(0.5)
        from capx_sparklab.api import YamSparkLabApi

        # Small batches: the GPU is shared with other work. Read when each planner is built.
        surface.PLAN_BATCH = 8
        yam = YamSparkLabApi(env, speak=False)
        yield yam
        env.close()
    finally:
        stop.set()
        thread.join()
        executor.shutdown()
        rig.destroy_node()
        rclpy.try_shutdown()
        for p in drivers:
            p.terminate()
            p.wait(timeout=10)


def test_motion_tools_need_an_arm(api):
    tools = api.functions()
    assert tools["go_to_pose"](pose=_fingers_down([0.35, 0.0, 0.25]))["error_code"] == "ARM_REQUIRED"
    assert tools["home"](arm_id="middle")["error_code"] == "UNKNOWN_ARM"


def test_go_to_pose_moves_only_the_named_arm(api):
    tools = api.functions()
    right_before = api._env.joints("right")[0]
    result = tools["go_to_pose"](arm_id="left", pose=_fingers_down([0.35, 0.0, 0.25]))
    assert result["status"] == "ok", result
    assert np.allclose(result["data"]["final_pose"]["position"], [0.35, 0.0, 0.25], atol=0.01)
    assert np.allclose(api._env.joints("right")[0], right_before, atol=0.01)


def test_get_state_reports_each_arm(api):
    state = api.functions()["get_state"]()["data"]["arms"]
    assert set(state) == {"left", "right"}
    assert np.allclose(state["left"]["joint_positions"], api._env.joints("left")[0], atol=1e-3)
    assert np.allclose(state["right"]["joint_positions"], api._env.joints("right")[0], atol=1e-3)


def test_grasps_stop_where_the_shoulder_strains(api):
    points = np.random.default_rng(0).uniform([0.44, -0.02, 0.0], [0.46, 0.02, 0.03], (200, 3))
    api._clouds["cloud_far"] = {"points": points, "camera_id": "top"}
    result = api.functions()["get_grasp_poses"](cloud_id="cloud_far", arm_id="left")
    assert result["error_code"] == "OUT_OF_REACH", result

def test_handover_passes_a_held_object(api, monkeypatch):
    tools = api.functions()
    right_base = EXTRINSIC["right_base"][:3, 3]
    start = right_base + [0.35, 0.0, 0.30]
    assert tools["go_to_pose"](arm_id="right", pose=_fingers_down(start))["status"] == "ok"

    # A 4 x 4 x 16 cm bar held by its top 2 cm, so there is length below the fingers to take.
    with api._arm("right"):
        centre = api._grasp_centre(api._tool_matrix())
    rng = np.random.default_rng(0)
    points = centre + rng.uniform([-0.02, -0.02, -0.14], [0.02, 0.02, 0.02], (2000, 3))
    api._clouds["cloud_bar"] = {"points": points, "camera_id": "top"}
    assert tools["update_collision_world"](attach_cloud_id="cloud_bar", arm_id="right")["status"] == "ok"

    def level_grasps(object_points, scene_points, top_k):
        # Level grasps closing across the bar from either side, both rolls, on each half.
        mean = object_points.mean(axis=0)
        axis = np.linalg.svd(object_points - mean)[2][0]
        along = (object_points - mean) @ axis
        side = np.cross(axis, [0.0, 0.0, 1.0])
        side /= np.linalg.norm(side)
        poses = []
        for half in (along > 0, along <= 0):
            target = object_points[half].mean(axis=0)
            for approach in (side, -side):
                for x in (axis, -axis):
                    T = np.eye(4)
                    T[:3, :3] = np.column_stack([x, np.cross(approach, x), approach])
                    T[:3, 3] = target - approach * api._reach
                    poses.append(T)
        return np.array(poses[:top_k]), np.linspace(0.9, 0.8, len(poses[:top_k]))

    monkeypatch.setattr(api, "_propose_grasps", level_grasps)
    real_set_gripper = api._env.set_gripper
    # The simulated fingers meet on nothing; report them stopping on a 4 cm bar.
    monkeypatch.setattr(api._env, "set_gripper",
                        lambda arm, fraction: 0.04 if fraction == 0.0 else real_set_gripper(arm, fraction))

    result = tools["handover"](from_arm="right", to_arm="left")
    assert result["status"] == "ok", result
    assert result["data"]["holder"] == "left" and result["data"]["retreated"]
    assert api._attached == {"left": "cloud_bar"}
