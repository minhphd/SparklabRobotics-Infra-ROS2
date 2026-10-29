"""Command latch for one arm: DISARMED / ARMED / FAULT, held inside ``arm_server``.

Only ARMED forwards targets, rate-limited in time. Any other state answers with a
hold target fixed at the measured pose when the state was entered, so neither a
late command nor a reconnecting client can resume motion: leaving DISARMED/FAULT
takes an explicit arm request, then a command that matches the measured pose. No
hardware dependency; ``arm_server`` applies the targets it returns.

Hold means "keep commanding the pose measured at the transition", which stops
motion only while the low-level control loop is alive and its readings valid.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DISARMED, ARMED, FAULT = "DISARMED", "ARMED", "FAULT"


@dataclass
class GuardLimits:
    """Simulation defaults, not validated for hardware.

    max_joint_speed: arm joints, rad/s.
    max_gripper_speed: gripper command units (0..1) per second.
    max_dt: longest interval a single step is scaled by, s.
    command_timeout: an ARMED gap longer than this faults, s.
    arm_tol: per-joint |target - measured| allowed when arming, rad.
    gripper_arm_tol: the same for the gripper, command units.
    arm_request_timeout: a pending arm request expires after this, s; long enough
        for a replay or recorder started right after the request to send its first command.
    """

    max_joint_speed: float = 4.0
    max_gripper_speed: float = 4.0
    max_dt: float = 0.05
    command_timeout: float = 0.25
    arm_tol: float = 0.15
    gripper_arm_tol: float = 0.20
    arm_request_timeout: float = 15.0


class ArmGuard:
    """State machine for one arm.

    n_arm: arm joints; any further entries (the gripper) use the gripper limits.
    limits: speed, timeout and arming tolerances.
    clock: monotonic seconds, injectable for tests.
    """

    def __init__(self, n_arm: int, limits: GuardLimits | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.n_arm = n_arm
        self.limits = limits or GuardLimits()
        self.clock = clock
        self.state = DISARMED
        self.reason = "started; waiting for an arm request"
        self.hold_target: np.ndarray | None = None
        # Bumped on every hold transition, so the caller commands each new hold exactly once.
        self.hold_seq = 0
        self.last_cmd_t: float | None = None
        self.pending_since: float | None = None
        self.arm_result = "no arm request yet"

    # ---- helpers ------------------------------------------------------
    def _per_joint(self, arm_value: float, gripper_value: float, n: int) -> np.ndarray:
        out = np.full(n, float(gripper_value))
        out[: self.n_arm] = arm_value
        return out

    def _mismatch(self, target: np.ndarray, present: np.ndarray) -> str | None:
        tol = self._per_joint(self.limits.arm_tol, self.limits.gripper_arm_tol, len(target))
        err = np.abs(target - present)
        bad = np.flatnonzero(err > tol)
        if bad.size == 0:
            return None
        j = int(bad[np.argmax(err[bad])])
        name = f"joint{j + 1}" if j < self.n_arm else "gripper"
        return (f"{name} target differs from the measured pose by {err[j]:.3f} "
                f"(tolerance {tol[j]:.3f}); match the leader to the robot first")

    def _enter(self, state: str, reason: str, present: np.ndarray) -> np.ndarray:
        """Leave ARMED: fix the hold target at the measured pose."""
        self.state, self.reason = state, reason
        self.hold_target = np.asarray(present, float).copy()
        self.hold_seq += 1
        self.pending_since = None
        return self.hold_target

    @staticmethod
    def _valid(x: np.ndarray, n: int) -> bool:
        return x.shape == (n,) and bool(np.all(np.isfinite(x)))

    # ---- transitions ----------------------------------------------------
    def start(self, present: np.ndarray) -> np.ndarray:
        """Initial DISARMED hold at the startup pose."""
        return self._enter(DISARMED, "started; waiting for an arm request", present)

    def disarm(self, reason: str, present: np.ndarray) -> np.ndarray:
        if self.state == FAULT:
            return self.hold_target
        return self._enter(DISARMED, reason, present)

    def fault(self, reason: str, present: np.ndarray) -> np.ndarray:
        return self._enter(FAULT, reason, present)

    def request_arm(self, present: np.ndarray) -> str:
        """Ask to arm; the next incoming command must match the measured pose.

        Returns: the resulting arm_result message.
        """
        present = np.asarray(present, float)
        if not self._valid(present, len(present)):
            self.arm_result = "arm rejected: measured pose is not finite"
            return self.arm_result
        if self.state == ARMED:
            self.arm_result = "already ARMED"
            return self.arm_result
        self.pending_since = self.clock()
        self.arm_result = "pending: waiting for the next command to check the pose match"
        return self.arm_result

    def _try_arm(self, target: np.ndarray, present: np.ndarray) -> str:
        self.pending_since = None
        if not self._valid(target, len(present)):
            self.arm_result = "arm rejected: target is malformed or not finite"
            return self.arm_result
        why = self._mismatch(target, present)
        if why is not None:
            self.arm_result = f"arm rejected: {why}"
            return self.arm_result
        self.state, self.reason = ARMED, "armed by explicit request"
        self.last_cmd_t = self.clock()
        self.arm_result = "ARMED"
        return self.arm_result

    def check_timeout(self, present: np.ndarray) -> bool:
        """Fault an ARMED arm whose commands stopped. Returns True on that transition."""
        now = self.clock()
        if self.pending_since is not None and now - self.pending_since > self.limits.arm_request_timeout:
            self.pending_since = None
            self.arm_result = (f"arm request expired: no command arrived within "
                               f"{self.limits.arm_request_timeout:.1f} s")
        if self.state == ARMED and now - self.last_cmd_t > self.limits.command_timeout:
            self.fault(f"command timeout: no command for {now - self.last_cmd_t:.2f} s "
                       f"(limit {self.limits.command_timeout:.2f} s)", present)
            return True
        return False

    def filter(self, target: np.ndarray, present: np.ndarray,
               caps: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Decide what to command for one incoming target.

        target: requested absolute positions; arming checks this unclamped value.
        present: measured positions.
        caps: optional per-call step caps for the arm joints, applied with the time cap.
        Returns: (positions, execute). execute=False means hold_target stands and
            nothing new may be commanded.
        """
        target = np.asarray(target, float)
        present = np.asarray(present, float)
        n = len(present)
        if not self._valid(present, n):
            return self.fault("measured pose is not finite", np.nan_to_num(present)), False
        if not self._valid(target, n):
            return self.fault(f"invalid target (shape {target.shape}, finite="
                              f"{bool(np.all(np.isfinite(target)))})", present), False
        now = self.clock()
        if self.state == ARMED and now - self.last_cmd_t > self.limits.command_timeout:
            self.fault(f"command gap of {now - self.last_cmd_t:.2f} s "
                       f"(limit {self.limits.command_timeout:.2f} s)", present)
            return self.hold_target, False
        if self.state != ARMED:
            if self.pending_since is None:
                return self.hold_target, False
            self._try_arm(target, present)
            if self.state != ARMED:
                return self.hold_target, False
        dt = min(now - self.last_cmd_t, self.limits.max_dt)
        cap = self._per_joint(self.limits.max_joint_speed, self.limits.max_gripper_speed, n) * dt
        if caps is not None:
            cap[: self.n_arm] = np.minimum(cap[: self.n_arm], np.asarray(caps, float))
        self.last_cmd_t = now
        return present + np.clip(target - present, -cap, cap), True

    def status(self) -> dict:
        return {"state": self.state, "reason": self.reason, "arm_result": self.arm_result,
                "pending": self.pending_since is not None}
