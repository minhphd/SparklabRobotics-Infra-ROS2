"""The generated robot description against the arm model the rest of the stack uses.

Needs cap-x's stack: ./scripts/capx_python.sh -m pytest src/capx_sparklab/test
"""

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from capx.integrations.motion.curobo_v2 import Planner
from capx_sparklab.robot import TOOL0_DEPTH, curobo_config
from sparklab_yam.ik.decoupled_ik import DecoupledIKSolver

STOCK_TOOL0 = 0.1347  # sparklab_yam's tool0, the stock gripper's grasp site


def _placement(xyz, rpy_deg) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


@pytest.fixture(scope="module", params=["left", "right"])
def arm(request):
    base = np.eye(4) if request.param == "left" else _placement([0.02, -0.548, 0.008], [-1.5, -0.3, 3.0])
    return base, Planner(robot=curobo_config(base), batch=4)


def test_tool_frame_matches_the_stack_model(arm):
    base, planner = arm
    ik = DecoupledIKSolver()
    rng = np.random.default_rng(0)
    lower, upper = ik.joint_limits[:, 0], ik.joint_limits[:, 1]
    for q in rng.uniform(lower, upper, (20, 6)):
        pos, quat_wxyz = ik.fk(q)
        R = Rotation.from_quat([*quat_wxyz[1:], quat_wxyz[0]]).as_matrix()
        # sparklab_yam's tool0 shares grasp_base's axes, STOCK_TOOL0 further along z.
        expected = base[:3, :3] @ (pos - R[:, 2] * STOCK_TOOL0) + base[:3, 3]
        got, got_wxyz = planner.tool_pose(q)
        got_R = Rotation.from_quat([*got_wxyz[1:], got_wxyz[0]]).as_matrix()
        assert np.linalg.norm(got - expected) < 1e-3
        assert np.degrees(Rotation.from_matrix(got_R.T @ base[:3, :3] @ R).magnitude()) < 0.1


def test_plans_out_of_the_parked_pose(arm):
    _, planner = arm
    # The planners' frame is the world (left_base) for both arms; the table top sits 1.5 cm below it.
    planner.set_world([{"name": "table", "dims": [2.0, 2.0, 0.1], "pose": [0, 0, -0.065, 1, 0, 0, 0]}])
    path = planner.plan_to_joints(np.array([0.0, 1.2, 1.2, -0.6, 0.0, 0.0]), np.zeros(6))
    assert path is not None and path.shape[1] == 6
    assert np.allclose(path[-1], [0.0, 1.2, 1.2, -0.6, 0.0, 0.0], atol=0.02)


def test_collision_spheres_reach_the_fingertips(arm):
    _, planner = arm
    q = np.zeros(6)
    pos, wxyz = planner.tool_pose(q)
    axis = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()[:, 2]
    spheres = planner.collision_spheres(q)
    reach = np.max((spheres[:, :3] - pos) @ axis + spheres[:, 3])
    # The fingertips end 1 cm past the grasp centre.
    assert TOOL0_DEPTH < reach < TOOL0_DEPTH + 0.03
