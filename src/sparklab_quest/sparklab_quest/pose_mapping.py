"""Clutch-relative pose mapping: WebXR controller pose -> EE target in arm base.

`ClutchPoseMapper` captures the controller and EE poses on the rising clutch
edge. While held, the EE target is that engaged pose composed with a controller
delta rotated into the arm base. Passing the arm's current EE pose into
`target()` enables the absorbing reach limits; omitting it gives the legacy
absolute delta-since-engage mapping. Disengaged, `target()` returns None.

Sanity tests: python -m sparklab_quest.pose_mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


# ------- small quaternion helpers (wxyz convention, matching mujoco) -------

def quat_mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(qa, float), np.asarray(qb, float))
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(q, float))
    return out


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(R, float).ravel())
    return out


def quat_pow(q: np.ndarray, k: float) -> np.ndarray:
    """Raise a quaternion to a scalar power: keep the axis, scale the angle by k."""
    w = float(q[0])
    v = np.asarray(q[1:], dtype=float)
    half_angle = float(np.arctan2(float(np.linalg.norm(v)), w))
    if half_angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / np.sin(half_angle)
    new_half = k * half_angle
    s = float(np.sin(new_half))
    return np.array([float(np.cos(new_half)), s * axis[0], s * axis[1], s * axis[2]])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis · angle, rad) of a wxyz quaternion, shortest way."""
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, np.asarray(q, float), 1.0)
    return out


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a rotation vector (axis · angle, rad)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(v, float) / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])


# ------- mapper -------

@dataclass
class ClutchPoseMapper:
    """One-handed clutch-relative controller→EE mapping.

    R: 3x3 taking Quest world vectors to arm base, v_armbase = R @ v_quest.
    scale: linear gain on translation, 1.0 = 1:1.
    scale_rotation: gain on rotation delta, 1.0 = 1:1.
    rotation_pivot: arm-base 3-vector to rotate the EE about, or None for
        in-place rotation. Captured per-engage by the caller.
    rot_reach_limit: max angle (rad) the orientation target may lead the arm's
        current orientation. Requires `ee_quat_armbase` in `target()`, and makes
        the orientation channel incremental with the excess absorbed. 0/None
        disables. See DESIGN.md for the absorbing-clutch semantics.
    pos_reach_limit: max distance (m) the position target may lead the current
        EE position. Requires `ee_pos_armbase`; same absorbing semantics.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    scale_rotation: float = 1.0
    rotation_pivot: np.ndarray | None = None
    rot_reach_limit: float | None = 0.6
    pos_reach_limit: float | None = 0.25

    def __post_init__(self) -> None:
        self._engaged: bool = False
        self._ctrl_engage_pos: np.ndarray | None = None
        self._ctrl_engage_quat: np.ndarray | None = None
        self._ee_engage_pos: np.ndarray | None = None
        self._ee_engage_quat: np.ndarray | None = None
        self._R_quat: np.ndarray = mat_to_quat(np.asarray(self.R, float))
        self._R_quat_conj: np.ndarray = quat_conj(self._R_quat)
        # Incremental reach-limit state: previous-tick controller pose (Quest)
        # and accumulated effective deltas (arm base). Reset on every engage.
        self._ctrl_prev_quat: np.ndarray | None = None
        self._ctrl_prev_pos: np.ndarray | None = None
        self._d_quat_eff: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff: np.ndarray = np.zeros(3)

    @property
    def engaged(self) -> bool:
        return self._engaged

    def set_R(self, R: np.ndarray) -> None:
        """Replace the rotation used for delta mapping.

        Typically called per engage with a yaw-corrected R, so the operator can
        turn their body and still have controller-forward mean robot-forward.
        """
        self.R = np.asarray(R, dtype=float).copy()
        self._R_quat = mat_to_quat(self.R)
        self._R_quat_conj = quat_conj(self._R_quat)

    def engage(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray,
        ee_quat_armbase: np.ndarray,
        pivot_armbase: np.ndarray | None = None,
    ) -> None:
        """Capture the engage frame. Call on the rising clutch edge.

        pivot_armbase: point to pivot rotation deltas about; None pivots at the EE.
        """
        self._ctrl_engage_pos = np.array(controller_pos_quest, float, copy=True)
        self._ctrl_engage_quat = np.array(controller_quat_quest, float, copy=True)
        self._ee_engage_pos = np.array(ee_pos_armbase, float, copy=True)
        self._ee_engage_quat = np.array(ee_quat_armbase, float, copy=True)
        self.rotation_pivot = (
            None if pivot_armbase is None else np.array(pivot_armbase, float, copy=True)
        )
        self._ctrl_prev_quat = self._ctrl_engage_quat.copy()
        self._ctrl_prev_pos = self._ctrl_engage_pos.copy()
        self._d_quat_eff = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff = np.zeros(3)
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    def target(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray | None = None,
        ee_quat_armbase: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute the current EE target in arm-base frame, or None if disengaged.

        ee_pos_armbase, ee_quat_armbase: the arm's current EE pose. Passing them
            enables the reach limits; without them the legacy absolute mapping
            applies unchanged.
        Returns: (target_pos, target_quat_wxyz) or None.
        """
        if not self._engaged:
            return None
        assert self._ctrl_engage_pos is not None
        assert self._ctrl_engage_quat is not None
        assert self._ee_engage_pos is not None
        assert self._ee_engage_quat is not None

        # Vectors commute, so per-tick increments compose to exactly the absolute
        # scaled delta until the reach limit absorbs.
        p_now = np.asarray(controller_pos_quest, float)
        pos_limited = ee_pos_armbase is not None and bool(self.pos_reach_limit)
        if pos_limited:
            assert self._ctrl_prev_pos is not None
            self._d_pos_eff = self._d_pos_eff + self.R @ (self.scale * (p_now - self._ctrl_prev_pos))
            d_pos_arm = self._d_pos_eff
        else:
            d_pos_arm = self.R @ (self.scale * (p_now - self._ctrl_engage_pos))
        self._ctrl_prev_pos = p_now.copy()

        q_now = np.asarray(controller_quat_quest, float)
        rot_limited = ee_quat_armbase is not None and bool(self.rot_reach_limit)
        if rot_limited:
            # Per-tick increments are never direction-ambiguous. At
            # scale_rotation=1 they telescope to the absolute delta; see DESIGN.md.
            assert self._ctrl_prev_quat is not None
            if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                q_now = -q_now  # hemisphere-align: q and -q are the same rotation
            inc = quat_mul(q_now, quat_conj(self._ctrl_prev_quat))
            self._ctrl_prev_quat = q_now.copy()
            if self.scale_rotation != 1.0:
                inc = quat_pow(inc, self.scale_rotation)
            inc_arm = quat_mul(quat_mul(self._R_quat, inc), self._R_quat_conj)
            d_quat_arm = quat_mul(inc_arm, self._d_quat_eff)
            d_quat_arm /= np.linalg.norm(d_quat_arm)
            self._d_quat_eff = d_quat_arm
        else:
            # Absolute delta (now · engage⁻¹) conjugated into arm base. The
            # incremental state is kept fresh so a live toggle sees no stale prev.
            if self._ctrl_prev_quat is not None:
                if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                    q_now = -q_now
                self._ctrl_prev_quat = q_now.copy()
            d_quat_quest = quat_mul(q_now, quat_conj(self._ctrl_engage_quat))
            if self.scale_rotation != 1.0:
                d_quat_quest = quat_pow(d_quat_quest, self.scale_rotation)
            d_quat_arm = quat_mul(quat_mul(self._R_quat, d_quat_quest), self._R_quat_conj)

        target_quat = quat_mul(d_quat_arm, self._ee_engage_quat)

        if rot_limited:
            # Clamp to within rot_reach_limit of the current orientation and absorb
            # the excess — absorbed twist does not return when the operator reverses.
            e = quat_to_rotvec(quat_mul(target_quat, quat_conj(np.asarray(ee_quat_armbase, float))))
            e_norm = float(np.linalg.norm(e))
            if e_norm > self.rot_reach_limit:
                e *= self.rot_reach_limit / e_norm
                target_quat = quat_mul(rotvec_to_quat(e), np.asarray(ee_quat_armbase, float))
                self._d_quat_eff = quat_mul(target_quat, quat_conj(self._ee_engage_quat))
                d_quat_arm = self._d_quat_eff

        # With a pivot set, rotation swings the EE in an arc around it; without
        # one the offset term vanishes and rotation is in-place.
        if self.rotation_pivot is not None:
            offset = self._ee_engage_pos - self.rotation_pivot
            rotated_offset = np.zeros(3)
            mujoco.mju_rotVecQuat(rotated_offset, offset, d_quat_arm)
            target_pos = self.rotation_pivot + rotated_offset + d_pos_arm
        else:
            target_pos = self._ee_engage_pos + d_pos_arm

        # Mouse-at-screen-edge: overshoot is absorbed, so reversal moves the
        # target immediately instead of after retracing it.
        if pos_limited:
            ee_p = np.asarray(ee_pos_armbase, float)
            dp = target_pos - ee_p
            dp_norm = float(np.linalg.norm(dp))
            if dp_norm > self.pos_reach_limit:
                clamped = ee_p + dp * (self.pos_reach_limit / dp_norm)
                self._d_pos_eff = self._d_pos_eff + (clamped - target_pos)
                target_pos = clamped

        return target_pos, target_quat


# ------- sanity tests -------

def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.allclose(a, b, atol=tol))


def main() -> None:
    print("=== ClutchPoseMapper sanity tests ===\n")

    ee_engage_pos = np.array([0.50, 0.00, 0.42])
    ee_engage_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
    ctrl_engage_pos = np.array([0.0, 1.4, -0.3])
    ctrl_engage_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # T1: identity R, no controller motion → target == engage
    m = ClutchPoseMapper()
    m.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    assert m.engaged
    out = m.target(ctrl_engage_pos, ctrl_engage_quat)
    assert out is not None
    p, q = out
    print(f"T1 (no motion):                target=({p.round(4)}, {q.round(4)})  "
          f"[{'ok' if _close(p, ee_engage_pos) and _close(q, ee_engage_quat) else 'FAIL'}]")

    # T2: identity R, controller moves +5 cm in X → EE target += (0.05, 0, 0)
    p, q = m.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.05, 0, 0])
    print(f"T2 (controller +5cm X, R=I):   target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) and _close(q, ee_engage_quat) else 'FAIL'}]")

    # T3: identity R, scale 0.5 → 5cm controller motion → 2.5cm EE motion
    m_scaled = ClutchPoseMapper(scale=0.5)
    m_scaled.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m_scaled.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.025, 0, 0])
    print(f"T3 (scale 0.5):                target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    # T4: non-trivial R (90° about Z swaps X<->Y, with sign): Quest +X → arm base +Y
    R_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # R_z90 @ (1,0,0) = (0,1,0)  → quest +X maps to arm +Y ✓
    m_R = ClutchPoseMapper(R=R_z90)
    m_R.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m_R.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.0, 0.05, 0.0])
    print(f"T4 (R = Rz90, controller +X):  target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    # T5: rotation. Identity R, controller rotates 30° about Quest +Y.
    # Should produce target_quat = R_y30 (in arm base, since R=I).
    m = ClutchPoseMapper()
    m.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    ctrl_now_quat = rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0]))
    _, q = m.target(ctrl_engage_pos, ctrl_now_quat)
    print(f"T5 (controller rotates +30° Y, R=I): target_quat={q.round(4)}  "
          f"expected≈{rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0])).round(4)}  "
          f"[{'ok' if _close(q, rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0]))) else 'FAIL'}]")

    # T6: disengage → target returns None
    m.disengage()
    out = m.target(ctrl_engage_pos, ctrl_engage_quat)
    print(f"T6 (after disengage):          target={out}  [{'ok' if out is None else 'FAIL'}]")

    # T7: re-engage at a different EE pose — engage origin must update
    new_ee_pos = np.array([0.55, 0.10, 0.42])
    m.engage(ctrl_engage_pos, ctrl_engage_quat, new_ee_pos, ee_engage_quat)
    p, _ = m.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = new_ee_pos + np.array([0.05, 0, 0])
    print(f"T7 (re-engage at new EE):      target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    def _ang_deg(qa, qb):
        return float(np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(qa, quat_conj(qb))))))

    # T8: twist 120° in 1° increments with the arm held still; the target must
    # never lead the current EE orientation by more than rot_reach_limit.
    m8 = ClutchPoseMapper(rot_reach_limit=0.5, pos_reach_limit=0.25)
    m8.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    worst = 0.0
    q_ctrl = ctrl_engage_quat
    for i in range(1, 121):
        q_ctrl = rotvec_to_quat(np.array([0.0, np.radians(i), 0.0]))
        _, tq = m8.target(ctrl_engage_pos, q_ctrl, ee_engage_pos, ee_engage_quat)
        worst = max(worst, _ang_deg(tq, ee_engage_quat))
    ok = worst <= np.degrees(0.5) + 0.1
    print(f"T8 (rot reach limit, 120° push):     max target-vs-EE angle={worst:.1f}°  "
          f"(reach limit {np.degrees(0.5):.1f}°)  [{'ok' if ok else 'FAIL'}]")

    # T9: after the 120° push, reversing by just the reach limit angle must
    # bring the target back onto the EE orientation.
    back = 120.0 - np.degrees(0.5)
    _, tq = m8.target(ctrl_engage_pos,
                      rotvec_to_quat(np.array([0.0, np.radians(back), 0.0])),
                      ee_engage_pos, ee_engage_quat)
    resid = _ang_deg(tq, ee_engage_quat)
    print(f"T9 (reversal bites at reach limit):  residual after backing off {np.degrees(0.5):.1f}°: "
          f"{resid:.2f}°  [{'ok' if resid < 1.0 else 'FAIL'}]")

    # T10: a 1 m push clamps to pos_reach_limit; the overshoot is absorbed, so a
    # 5 cm reversal moves the target back 5 cm immediately.
    m10 = ClutchPoseMapper(rot_reach_limit=1.0, pos_reach_limit=0.25)
    m10.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m10.target(ctrl_engage_pos + np.array([1.0, 0, 0]), ctrl_engage_quat,
                      ee_engage_pos, ee_engage_quat)
    ok1 = _close(p, ee_engage_pos + np.array([0.25, 0, 0]))
    p, _ = m10.target(ctrl_engage_pos + np.array([0.95, 0, 0]), ctrl_engage_quat,
                      ee_engage_pos, ee_engage_quat)
    ok2 = _close(p, ee_engage_pos + np.array([0.20, 0, 0]))
    print(f"T10 (pos reach limit, mouse-style):  clamp@1m={'ok' if ok1 else 'FAIL'}  "
          f"reversal-bites={'ok' if ok2 else 'FAIL'}")

    # T11: within the reach limit, with the EE tracking, the incremental path
    # reproduces the absolute mapping. Exact only at scale_rotation=1.
    m_inc = ClutchPoseMapper(rot_reach_limit=3.0, pos_reach_limit=10.0)
    m_abs = ClutchPoseMapper()
    for mm in (m_inc, m_abs):
        mm.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    ee_q = ee_engage_quat
    worst = 0.0
    for i in range(1, 41):
        q_ctrl = rotvec_to_quat(np.radians(i) * np.array([0.5, 0.7, 0.2]))
        _, tq_i = m_inc.target(ctrl_engage_pos, q_ctrl, ee_engage_pos, ee_q)
        _, tq_a = m_abs.target(ctrl_engage_pos, q_ctrl)
        worst = max(worst, _ang_deg(tq_i, tq_a))
        ee_q = tq_i  # EE follows perfectly
    print(f"T11 (incremental == absolute within reach limit): max diff={worst:.4f}°  "
          f"[{'ok' if worst < 0.01 else 'FAIL'}]")


if __name__ == "__main__":
    main()
