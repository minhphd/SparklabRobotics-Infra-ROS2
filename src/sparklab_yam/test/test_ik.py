"""The full IK solver on the arm model; no ROS."""

import numpy as np
from scipy.spatial.transform import Rotation

from sparklab_yam.ik.solver import IKSolver


def _rotated(quat_wxyz, rotvec):
    R = Rotation.from_rotvec(rotvec) * Rotation.from_quat(np.roll(quat_wxyz, -1))
    return np.roll(R.as_quat(), 1)


def _iterate(solver, pos, quat, q, steps=500):
    for _ in range(steps):
        q = solver.solve(pos, quat, q)
    return q


def test_iterating_reaches_poses_near_the_seed():
    solver = IKSolver()
    rng = np.random.default_rng(0)
    lo, hi = solver.joint_limits.T
    for _ in range(20):
        truth = rng.uniform(lo + 0.3, hi - 0.3)
        pos, quat = solver.fk(truth)
        q = _iterate(solver, pos, quat, truth + rng.normal(0, 0.15, 6))
        reached, reached_quat = solver.fk(q)
        assert np.linalg.norm(reached - pos) < 1e-4
        assert abs(reached_quat @ quat) > np.cos(1e-3 / 2)


def test_a_twist_next_to_the_wrist_singularity_takes_a_small_step():
    solver = IKSolver()
    q88 = np.array([0.2, 1.4, 1.6, 0.3, np.radians(88.0), 0.2])
    pos, quat = solver.fk(q88)
    for axis in np.eye(3):
        step = solver.solve(pos, _rotated(quat, np.radians(1.0) * axis), q88) - q88
        assert np.max(np.abs(step)) < np.radians(5.0)
    assert solver.last_singularity_proximity > 0.85


def test_a_joint_at_its_limit_leaves_the_position_to_the_others():
    solver = IKSolver(max_dq_per_joint=[0.06] * 3 + [0.15] * 3)
    q0 = np.array([0.2, 1.3, 1.5, 0.2, 1.50, 0.2])
    pos, quat = solver.fk(q0)
    solver._fk(q0)
    body = solver.model.jnt_bodyid[4]
    joint5_axis = solver.data.xmat[body].reshape(3, 3) @ solver.model.jnt_axis[4]
    q = _iterate(solver, pos, _rotated(quat, np.radians(15.0) * joint5_axis), q0, steps=300)
    assert q[4] == solver.joint_limits[4, 1]
    # Clamping after an unconstrained step gives up 32 mm here.
    assert np.linalg.norm(solver.fk(q)[0] - pos) < 0.01


def test_out_of_reach_stays_within_the_limits_and_says_how_far():
    solver = IKSolver()
    q = _iterate(solver, [2.0, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], np.array([0.0, 1.57, 1.57, 0.0, 0.0, 0.0]))
    lo, hi = solver.joint_limits.T
    assert np.all(np.isfinite(q)) and np.all((lo <= q) & (q <= hi))
    assert solver.last_pos_err_norm > 1.0


def test_the_step_cap_shrinks_the_whole_step():
    free, capped = IKSolver(), IKSolver(max_dq_per_joint=[0.01] * 6)
    q0 = np.array([0.0, 1.2, 1.4, 0.0, 0.3, 0.0])
    pos, quat = free.fk(q0)
    target = pos + np.array([0.0, 0.1, 0.0])
    step, small = free.solve(target, quat, q0) - q0, capped.solve(target, quat, q0) - q0
    assert np.isclose(np.max(np.abs(small)), 0.01)
    np.testing.assert_allclose(small, step * 0.01 / np.max(np.abs(step)), atol=1e-9)
