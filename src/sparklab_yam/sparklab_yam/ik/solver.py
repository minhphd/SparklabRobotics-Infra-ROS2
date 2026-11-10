"""Full IK for the YAM-Ultra: one damped least-squares step on position and orientation together.

Every joint works on the whole 6-D error. A joint the step would push past a
limit is pinned there and the others re-solve, so they make up what it cannot.
`solve()` is one step from the caller's qpos: call it every control tick, or
iterate it to convergence. See DESIGN.md#inverse-kinematics.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..model.kinematics import build_model_with_tool0_site

ARM_DOFS = 6


class IKSolver:
    """Full IK for one YAM-Ultra arm, tool0 as the end effector, in the arm base frame.

    lam, lam0, sigma0: damping, λ² = lam² + lam0²·ramp², ramp = max(0, 1 - σ/sigma0),
        σ the weighted Jacobian's smallest singular value (m/rad).
    rot_weight: m per rad of orientation error, what a radian of twist costs
        against position when the pose cannot be met.
    max_dq_per_joint: per-joint cap on one step (6,); the whole step shrinks to fit.
    """

    def __init__(
        self,
        arm_xml_path: Path | None = None,
        lam: float = 0.002,
        lam0: float = 0.05,
        sigma0: float = 0.02,
        rot_weight: float = 0.1,
        max_dq_per_joint: list[float] | None = None,
    ) -> None:
        self.lam, self.lam0, self.sigma0 = float(lam), float(lam0), float(sigma0)
        self.rot_weight = float(rot_weight)
        self.max_dq_per_joint = (None if max_dq_per_joint is None
                                 else np.asarray(max_dq_per_joint, dtype=float).reshape(ARM_DOFS).copy())
        self.model, self.data = build_model_with_tool0_site(arm_xml_path)
        self.joint_limits = self.model.jnt_range[:ARM_DOFS].copy()
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
        self.j4_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "j4_anchor")
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Set by each solve(); the teleop turns them into controller vibration.
        self.last_limit_pressure = 0.0
        self.last_pos_err_norm = 0.0
        self.last_singularity_proximity = 0.0

    def _fk(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[:ARM_DOFS] = np.asarray(qpos, dtype=float)[:ARM_DOFS]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """tool0 in the arm base for joints 1-6.

        Returns: position (3,), quaternion wxyz (4,).
        """
        self._fk(qpos)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return self.data.site_xpos[self.site_id].copy(), quat

    def j4_anchor_xpos(self) -> np.ndarray:
        """The j4_anchor site, on the forearm past joint 4, at the last fk: the teleop's rotation pivot."""
        return self.data.site_xpos[self.j4_site_id].copy()

    def solve(self, target_pos: np.ndarray, target_quat_wxyz: np.ndarray, qpos_seed: np.ndarray) -> np.ndarray:
        """One damped least-squares step toward a tool0 pose. Always returns joints within the limits.

        target_pos: tool0 position in the arm base (3,).
        target_quat_wxyz: tool0 orientation (4,).
        qpos_seed: the joints to step from, at least 6 long.
        Returns: joints 1-6 (6,).
        """
        q = np.asarray(qpos_seed, dtype=float)[:ARM_DOFS].copy()
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        self._fk(q)
        pos_err = np.asarray(target_pos, dtype=float) - self.data.site_xpos[self.site_id]
        target_R = np.zeros(9)
        mujoco.mju_quat2Mat(target_R, np.asarray(target_quat_wxyz, dtype=float))
        # World-frame rotation vector of R_target·R_currentᵀ, the frame mj_jacSite's jacr maps into.
        current_R = self.data.site_xmat[self.site_id].reshape(3, 3)
        err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(err_quat, (target_R.reshape(3, 3) @ current_R.T).ravel())
        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
        err = np.concatenate([pos_err, self.rot_weight * rot_err])
        self.last_pos_err_norm = float(np.linalg.norm(pos_err))

        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        J = np.vstack([self._jacp[:, :ARM_DOFS], self.rot_weight * self._jacr[:, :ARM_DOFS]])
        sigma = float(np.linalg.svd(J, compute_uv=False)[-1])
        ramp = max(0.0, 1.0 - sigma / self.sigma0)
        self.last_singularity_proximity = ramp
        lam2 = self.lam ** 2 + (self.lam0 * ramp) ** 2

        dq = np.zeros(ARM_DOFS)
        free = np.ones(ARM_DOFS, dtype=bool)
        for i in range(ARM_DOFS):
            Jf = J[:, free]
            dq[free] = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam2 * np.eye(6), err - J[:, ~free] @ dq[~free])
            reached = q + dq
            over = free & ((reached < lo) | (reached > hi))
            if i == 0:
                self.last_limit_pressure = float(np.linalg.norm(reached - np.clip(reached, lo, hi)))
            if not over.any():
                break
            dq[over] = np.clip(reached, lo, hi)[over] - q[over]
            free &= ~over

        if self.max_dq_per_joint is not None:
            dq *= min(1.0, float(np.min(self.max_dq_per_joint / np.maximum(np.abs(dq), 1e-12))))
        return np.clip(q + dq, lo, hi)
