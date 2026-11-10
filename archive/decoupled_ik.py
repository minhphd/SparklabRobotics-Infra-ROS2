"""Decoupled IK for the bimanual YAM-Ultra — strict joint decoupling.

Joints 1-3 satisfy position, joints 4-6 orientation; each is one
damped-least-squares step per call, warm-started from the caller's qpos.

`solve()` always returns a valid qpos6. Workspace edge, joint limits, wrist
gimbal lock and near-antipodal demands are handled in-line so the arm degrades
gracefully instead of freezing — see DESIGN.md for each case.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

try:
    from ..model.kinematics import build_model_with_tool0_site
except ImportError:  # allow running this file directly as a script
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from sparklab_yam.model.kinematics import build_model_with_tool0_site


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


class DecoupledIKSolver:
    """Decoupled IK for the bimanual YAM-Ultra.

    Holds a mujoco model with the tool0 site plus cached site-id lookups.
    `solve()` is the main entry point.

    lam_pos, lam0, w0   Position sub-solve damping: base, ramp amplitude near
        the joints-1-3 singularity, and the manipulability threshold where the
        ramp starts. λ² = lam_pos² + lam0²·ramp², ramp = max(0, 1 - w/w0),
        w = |det(J_pos_arm)|.
    mu                  Tikhonov stiffness pulling joints 1-3 toward q_rest.
    lam_rot, lam0_rot, w0_rot   Same recipe for the wrist orientation sub-solve;
        w = |det(J_rot)| ≈ |cos θ5|, so the 0.5 default ramps in at θ5 ≈ 60°.
    rot_err_hold        Park the wrist past this orientation error (rad).
    q_rest              Rest pose for joints 1-3 that the Tikhonov term targets.
    max_dq_per_joint    Per-joint Δq cap, length 6.
    """

    def __init__(
        self,
        arm_xml_path: Path | None = None,
        lam_pos: float = 0.05,
        lam0: float = 0.15,
        w0: float = 0.05,
        mu: float = 0.02,
        lam_rot: float = 0.05,
        lam0_rot: float = 0.4,
        w0_rot: float = 0.5,
        rot_err_hold: float = 2.2,
        q_rest: np.ndarray | None = None,
        max_dq_per_joint: list[float] | None = None,
    ) -> None:
        self.lam_pos = float(lam_pos)
        self.lam0 = float(lam0)
        self.w0 = float(w0)
        self.mu = float(mu)
        self.lam_rot = float(lam_rot)
        self.lam0_rot = float(lam0_rot)
        self.w0_rot = float(w0_rot)
        self.rot_err_hold = float(rot_err_hold)
        self.q_rest_123 = (
            np.array([0.0, np.pi / 2, np.pi / 2], dtype=float)
            if q_rest is None else np.asarray(q_rest, dtype=float)[:3].copy()
        )
        self.max_dq_per_joint = (
            None if max_dq_per_joint is None
            else np.asarray(max_dq_per_joint, dtype=float).reshape(6).copy()
        )

        self.model, self.data = build_model_with_tool0_site(arm_xml_path)
        # Straight from the compiled MJCF — nothing transcribed by hand.
        self.joint_limits = self.model.jnt_range[:6].copy()

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0"
        )
        # Position-task anchor: 10 cm past joint 4 along link3→link4, so it is
        # wrist-invariant.
        self.j4_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "j4_anchor"
        )
        if -1 in (self.site_id, self.j4_site_id):
            raise RuntimeError("DecoupledIKSolver: required site missing from model")

        # Set by each solve(); fed into the controller-haptic mix downstream.
        self.last_limit_pressure: float = 0.0
        self.last_pos_err_norm: float = 0.0
        self.last_singularity_proximity: float = 0.0
        self.last_wrist_gimbal_proximity: float = 0.0

    def _fk(self, qpos: np.ndarray) -> None:
        """Run kinematics + comPos (required for the site Jacobians in solve())."""
        self.data.qpos[: self.model.nq] = 0
        self.data.qpos[: len(qpos)] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # for jacobians

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns (tool0 world pos, tool0 world quat wxyz)."""
        self._fk(qpos)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def j4_anchor_xpos(self) -> np.ndarray:
        """World position of the j4_anchor site at the most recent FK.

        The wrist-invariant point the position task targets; the pose mapper
        uses it as the rotation pivot so pure rotations leave joints 1-3 at rest.
        """
        return self.data.site_xpos[self.j4_site_id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        qpos_seed: np.ndarray,
    ) -> np.ndarray:
        """One step of decoupled IK. Always returns a valid 6-vector.

        target_pos: tool0 position in arm base (3,).
        target_quat_wxyz: tool0 orientation (4,).
        qpos_seed: warm start, at least 6 long.
        Returns: clamped joint positions (6,).
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        R_target = _quat_wxyz_to_R(target_quat_wxyz)

        # Step 1: FK at the seed; read the current (tool0 → j4) vector in tool0's
        # frame. Re-derived each tick so a re-anchor gives pos_err = 0 exactly.
        self._fk(qpos_seed)
        current_tool0 = self.data.site_xpos[self.site_id].copy()
        current_R_tool0 = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        current_J4 = self.data.site_xpos[self.j4_site_id].copy()
        ee_to_j4_in_tool0 = current_R_tool0.T @ (current_J4 - current_tool0)

        # ----- Step 2: target J4 anchor in arm-base; pos_err for Newton step -----
        target_J4 = target_pos + R_target @ ee_to_j4_in_tool0
        pos_err = target_J4 - current_J4

        self.last_pos_err_norm = float(np.linalg.norm(pos_err))

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.j4_site_id)
        J_pos_arm = jacp[:, :3]                  # 3x3, joints 1-3 only

        # Manipulability-adaptive damping: climbs as w → 0, bounding joint
        # velocities near the shoulder singularity without affecting tracking.
        w = abs(float(np.linalg.det(J_pos_arm)))
        ramp = max(0.0, 1.0 - w / max(self.w0, 1e-12))
        self.last_singularity_proximity = float(ramp)
        lam2 = self.lam_pos ** 2 + (self.lam0 ** 2) * (ramp ** 2)
        mu2 = self.mu ** 2
        A = J_pos_arm.T @ J_pos_arm + (lam2 + mu2) * np.eye(3)
        b = J_pos_arm.T @ pos_err + mu2 * (self.q_rest_123 - qpos_seed[:3])
        dq_arm = np.linalg.solve(A, b)

        new_q123 = qpos_seed[:3] + dq_arm

        # ----- Step 3: FK at (new_q123, seed_q456); read tool0 orientation -----
        qpos_after_arm = qpos_seed.copy()
        qpos_after_arm[:3] = new_q123
        self._fk(qpos_after_arm)
        R_cur = self.data.site_xmat[self.site_id].reshape(3, 3)

        # Step 4: orientation error as a world-frame rotation vector, so
        # R_target = exp([e_rot])·R_cur — the frame mj_jacSite maps into.
        R_err = R_target @ R_cur.T
        q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, np.ascontiguousarray(R_err).ravel())
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)

        # Step 5: damped LS on the wrist, same recipe as the arm. w = |det(J_rot)|
        # ≈ |cos θ5| hits 0 at gimbal lock; damping keeps J4/J6 steps bounded.
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J_rot = jacr[:, 3:6]                     # 3x3, wrist joints 4-6
        w_rot = abs(float(np.linalg.det(J_rot)))
        ramp_rot = max(0.0, 1.0 - w_rot / max(self.w0_rot, 1e-12))
        self.last_wrist_gimbal_proximity = float(ramp_rot)
        # Antipode gate: past `rot_err_hold` the shortest-way direction of e_rot
        # flips under jitter, so park the wrist rather than chase it.
        wrist_parked = float(np.linalg.norm(e_rot)) > self.rot_err_hold
        if wrist_parked:
            dq_wrist = np.zeros(3)
        else:
            lam2_rot = self.lam_rot ** 2 + (self.lam0_rot ** 2) * (ramp_rot ** 2)
            A_rot = J_rot.T @ J_rot + lam2_rot * np.eye(3)
            dq_wrist = np.linalg.solve(A_rot, J_rot.T @ e_rot)
        new_q456 = qpos_seed[3:6] + dq_wrist

        # ----- Step 6: joint-limit clamp; limit-pressure metric -----
        qpos6 = np.concatenate([new_q123, new_q456])
        qpos6_reachable = np.clip(qpos6, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # L2 distance from unclamped to clamped step: how hard this tick pushes
        # into the stops. Excludes the rate caps below, which are speed limits.
        self.last_limit_pressure = float(
            np.linalg.norm(qpos6 - qpos6_reachable)
        )
        if wrist_parked:
            self.last_limit_pressure = max(self.last_limit_pressure, 0.35)

        # ----- Step 7: per-joint Δq cap (operator-safety velocity bound) -----
        qpos6 = qpos6_reachable
        dq_total = qpos6 - qpos_seed[:6]
        if self.max_dq_per_joint is not None:
            dq_total = np.clip(dq_total, -self.max_dq_per_joint, self.max_dq_per_joint)
        qpos6 = qpos_seed[:6] + dq_total

        # ----- Step 8: clamp into joint limits (don't reject) -----
        qpos6 = np.clip(qpos6, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return qpos6


# ---------- self-test ----------

def _self_test() -> None:
    """Round-trip: pick known qpos → FK → solve from that target → check."""
    solver = DecoupledIKSolver(max_dq_per_joint=[0.05] * 6)

    test_set = [
        ("home elbow-up",   np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0])),
        ("slight rotation", np.array([0.2, np.pi/2, np.pi/2, 0.1, 0.1, 0.1])),
        ("forward reach",   np.array([0.0, 1.0,    1.0,    0.0, 0.0, 0.0])),
        ("yaw + pitch",     np.array([0.3, 1.4,    1.6,   -0.2, 0.3, 0.4])),
    ]

    print(f"{'pose':22s}  Δqpos  |  pos_err  |  rot_err_frob")
    print("-" * 70)
    for label, q in test_set:
        # FK to get target.
        solver._fk(q)
        target_pos = solver.data.site_xpos[solver.site_id].copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, solver.data.site_xmat[solver.site_id])

        # Equals q only for the first case; the rest exercise one capped step.
        seed = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = solver.solve(target_pos, target_quat, seed)

        full_q = np.zeros(8)
        full_q[:6] = result
        solver._fk(full_q)
        actual_pos = solver.data.site_xpos[solver.site_id].copy()
        actual_R = solver.data.site_xmat[solver.site_id].reshape(3, 3)
        target_R = _quat_wxyz_to_R(target_quat)
        rot_err = float(np.linalg.norm(actual_R - target_R, ord='fro'))
        pos_err = float(np.linalg.norm(actual_pos - target_pos))
        diff = float(np.linalg.norm(result - q[:6]))
        print(f"{label:22s}  {diff:.4f}  |  {pos_err*1000:5.1f} mm  |  {rot_err:.3f}")

    # Boundary handling: all should return a valid 6-vector, no exception.
    print("\nboundary checks (target placed deliberately bad):")
    seed = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    far_pos = np.array([2.0, 0.0, 0.4])   # way past reach
    out = solver.solve(far_pos, np.array([1.0, 0.0, 0.0, 0.0]), seed)
    print(f"  far-out-of-reach: qpos6 finite={bool(np.all(np.isfinite(out)))}  "
          f"|Δq|={float(np.linalg.norm(out - seed[:6])):.4f}")
    # Force a near-gimbal target by setting θ5 near ±π/2.
    q_gimbal = np.array([0.0, np.pi/2, np.pi/2, 0.0, np.pi/2 - 1e-4, 0.0])
    solver._fk(q_gimbal)
    g_pos = solver.data.site_xpos[solver.site_id].copy()
    g_quat = np.zeros(4)
    mujoco.mju_mat2Quat(g_quat, solver.data.site_xmat[solver.site_id])
    out = solver.solve(g_pos, g_quat, seed)
    print(f"  wrist near gimbal: qpos6 finite={bool(np.all(np.isfinite(out)))}  "
          f"θ5={out[4]:.4f}")

    # A 1° twist near θ5 = 88° must not demand a large J4/J6 step; the
    # closed-form extraction this replaced demanded ~27°.
    raw = DecoupledIKSolver()   # no Δq cap — observe the raw step
    q88 = np.array([0.2, 1.4, 1.6, 0.3, np.radians(88.0), 0.2, 0.0, 0.0])
    pos88, quat88 = raw.fk(q88)
    R88 = _quat_wxyz_to_R(quat88)
    worst = 0.0
    for axis in np.eye(3):
        for sign in (1.0, -1.0):
            ang = np.radians(1.0) * sign
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            R_t = R88 @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K))
            q_t = np.zeros(4)
            mujoco.mju_mat2Quat(q_t, np.ascontiguousarray(R_t).ravel())
            o = raw.solve(pos88, q_t, q88)
            worst = max(worst, abs(o[3] - q88[3]), abs(o[5] - q88[5]))
    ok = worst < np.radians(5.0)
    print(f"  gimbal conditioning @θ5=88°: worst J4/J6 step for 1° twist = "
          f"{np.degrees(worst):.2f}°  [{'ok' if ok else 'FAIL'}]")
    assert ok, "wrist step near gimbal not bounded — damping broken?"

    # Iterating the differential wrist on a fixed reachable target must drive
    # the orientation error to ~zero.
    q_t = np.array([0.3, 1.4, 1.6, -0.2, 0.3, 0.4])
    raw._fk(q_t)
    t_pos = raw.data.site_xpos[raw.site_id].copy()
    t_quat = np.zeros(4)
    mujoco.mju_mat2Quat(t_quat, raw.data.site_xmat[raw.site_id])
    cur = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        cur[:6] = raw.solve(t_pos, t_quat, cur)
    _, q_reached = raw.fk(cur)
    R_a, R_b = _quat_wxyz_to_R(q_reached), _quat_wxyz_to_R(t_quat)
    cos_err = (float(np.trace(R_a @ R_b.T)) - 1.0) / 2.0
    rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_err, -1.0, 1.0))))
    pos_err_mm = float(np.linalg.norm(raw.data.site_xpos[raw.site_id] - t_pos)) * 1000
    ok = rot_err_deg < 1.0 and pos_err_mm < 15.0
    print(f"  convergence (100 iters): rot_err={rot_err_deg:.3f}°  "
          f"pos_err={pos_err_mm:.1f} mm  [{'ok' if ok else 'FAIL'}]")
    assert ok, "iterated solve did not converge to the target pose"

    # ~150° away must park the wrist (zero step, saturating pressure) while
    # ~100° still tracks — guards the bang-bang oscillation seen on hardware.
    q0 = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.3, 0.0, 0.0, 0.0])
    p0, quat0 = raw.fk(q0)
    R0 = _quat_wxyz_to_R(quat0)

    def _twisted_target(deg: float) -> np.ndarray:
        ang = np.radians(deg)
        K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        R_t = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)) @ R0
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R_t).ravel())
        return qt

    out150 = raw.solve(p0, _twisted_target(150.0), q0)
    parked = bool(np.all(out150[3:] == q0[3:6]))
    pressure_at_park = raw.last_limit_pressure
    out100 = raw.solve(p0, _twisted_target(100.0), q0)
    tracks = float(np.linalg.norm(out100[3:] - q0[3:6])) > 1e-3
    ok = parked and pressure_at_park >= 0.35 and tracks
    print(f"  antipode gate: parked@150°={parked} (pressure={pressure_at_park:.2f})  "
          f"tracks@100°={tracks}  [{'ok' if ok else 'FAIL'}]")
    assert ok, "antipode gate broken (wrist should park at 150°, track at 100°)"


if __name__ == "__main__":
    _self_test()
