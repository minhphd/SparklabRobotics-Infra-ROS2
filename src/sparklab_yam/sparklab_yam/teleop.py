"""Quest controllers -> YAM-Ultra joint targets: clutch pose mapping plus IK.

Per arm keeps a ClutchPoseMapper and a DecoupledIKSolver. Per tick
`_update_arm` runs: staleness gate -> EMA pose filter -> precision button ->
clutch edges -> IK step -> haptic.

step() returns {left,right}_joint_{1..6}.pos (rad) and {left,right}_gripper.pos
(0..1), the LeRobot action keys. Used by the teleop node and by the LeRobot
``bi_quest_teleop`` teleoperator.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32, String

from sparklab_quest import buttons as xr_buttons
from sparklab_quest.input import Controller, QuestInput
from sparklab_quest.pose_mapping import ClutchPoseMapper

from .ik.decoupled_ik import DecoupledIKSolver
from .model.kinematics import DEFAULT_Q_REST


# Default rotation taking Quest `local-floor` world axes into the arm base
DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

ARM_DOFS = 6
NQ = 8  # arm (6) + 2 gripper-finger sliders

# Gripper prismatic range from the URDF; both fingers share this qpos scale.
GRIPPER_QPOS_OPEN = 0.001
GRIPPER_QPOS_CLOSED = -0.045

# The bridge publishes the page's settings latched, so a teleop started after
# the page still gets them.
SETTINGS_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _headset_relative_calibration(
    base_r_calib: np.ndarray, head_quat_xyzw: np.ndarray | None
) -> np.ndarray:
    """Return Quest-world to arm-base rotation aligned to headset yaw."""
    if head_quat_xyzw is None:
        return base_r_calib.copy()
    qx, qy, qz, qw = np.asarray(head_quat_xyzw, dtype=float)
    headset_rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    forward = headset_rotation @ np.array([0.0, 0.0, -1.0])
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return base_r_calib.copy()
    forward /= norm
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    # Columns are headset-local X/Y/Z expressed in Quest world coordinates.
    headset_yaw = np.column_stack((right, up, -forward))
    return base_r_calib @ headset_yaw.T


@dataclass
class QuestTeleopConfig:
    """Config for QuestTeleop.

    `rest_qpos_{left,right}` (six joint angles, rad) seed each arm's qpos and
    bias the IK Tikhonov term, which is what breaks the elbow-flip ambiguity.
    """

    # Defaults to the vendored yam_ultra.xml; override for tightened ranges.
    arm_xml_path: str = ""
    # Row-major 3x3 taking Quest world vectors into the arm base frame.
    r_calib: list[list[float]] = field(default_factory=lambda: [row[:] for row in DEFAULT_R_CALIB])
    rest_qpos_left: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())
    rest_qpos_right: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())

    # IK / mapping knobs.
    lam: float = 0.05  # Position-solve base damping
    lam0: float = 0.15  # Adaptive-damping ramp amplitude near singular
    w0: float = 0.05  # Manipulability threshold where ramp starts
    mu: float = 0.02  # Tikhonov stiffness toward q_rest
    lam_rot: float = 0.05  # Orientation-solve base damping
    lam0_rot: float = 0.4  # Extra damping ramp amplitude near wrist gimbal
    w0_rot: float = 0.5  # Wrist-manipulability threshold where ramp starts

    rot_reach_limit: float = 0.6  # rad (~34°)
    pos_reach_limit: float = 0.25  # m
    rot_err_hold: float = 2.2
    scale_translation: float = 1.5  # controller→EE translation gain
    scale_rotation: float = 1.5  # controller→EE rotation gain (<1 = softer)
    precision_factor: float = 0.5

    # EMA on the controller pose. 1.0 = raw; at 200 Hz, 0.5 ≈ 7 ms time constant.
    pose_filter_alpha: float = 0.8
    # Rot cap must not exceed the driver's max_step (0.15): a larger value lets
    # the solver's qpos run away from the rate-clamped arm.
    max_dq_per_joint: list[float] | None = field(
        default_factory=lambda: [0.06, 0.06, 0.06, 0.15, 0.15, 0.15]
    )

    max_dq_per_joint_scalar_pos: float = 0.06
    max_dq_per_joint_scalar_rot: float = 0.15

    # Go-home ramp: lerp qpos[:6] to rest_qpos_{hand} over this window.
    rest_ramp_duration_s: float = 2.0


class ArmState(TypedDict):
    solver: DecoupledIKSolver
    mapper: ClutchPoseMapper
    qpos: np.ndarray
    last_grip: bool
    last_precision: bool
    trigger: float
    engaged: bool
    haptic: float
    needs_reanchor: bool
    pos_filt: np.ndarray | None
    quat_filt: np.ndarray | None
    last_rest_button: bool
    ramp_active: bool
    ramp_start_q: np.ndarray
    ramp_target_q: np.ndarray
    ramp_start_t: float


class QuestTeleop:
    """Quest controllers -> joint targets for one or both arms.

    Subscribes the Quest topics on `node`, publishes `/quest/<hand>/{haptic,engaged}`
    on every step() and applies the page's `/quest/settings`.

    config: a QuestTeleopConfig, mutated in place by the page's settings.
    node: the node to subscribe and publish on; the caller spins it.
    hands: which arms to drive.
    """

    def __init__(
        self,
        config: QuestTeleopConfig,
        node: Node,
        hands: tuple[str, ...] = ("left", "right"),
    ) -> None:
        self.config = config

        self._r_calib = np.asarray(config.r_calib, dtype=float)
        if self._r_calib.shape != (3, 3):
            raise ValueError(
                f"r_calib must be a 3x3 rotation matrix, got shape {self._r_calib.shape}"
            )
        self._mapping_r_calib = self._r_calib.copy()
        self._viewer_seen = False

        if not hands or any(hand not in ("left", "right") for hand in hands):
            raise ValueError(f"hands must contain left and/or right, got {hands!r}")
        self._hands = tuple(dict.fromkeys(hands))
        self._lock = threading.RLock()
        self._log = node.get_logger()
        self._gap_samples: list[float] = []
        self._gap_report_t: float = time.perf_counter()

        rest_qpos = {
            "left": np.asarray(config.rest_qpos_left, dtype=float),
            "right": np.asarray(config.rest_qpos_right, dtype=float),
        }
        for hand, q in rest_qpos.items():
            if q.shape != (ARM_DOFS,):
                raise ValueError(f"rest_qpos_{hand} must have {ARM_DOFS} values, got shape {q.shape}")

        self._arms: dict[str, ArmState] = {}
        for hand in self._hands:
            q_rest = rest_qpos[hand]
            qpos_init = np.zeros(NQ)
            qpos_init[:ARM_DOFS] = q_rest
            arm_solver = DecoupledIKSolver(
                arm_xml_path=config.arm_xml_path or None,
                lam_pos=config.lam,
                lam0=config.lam0,
                w0=config.w0,
                mu=config.mu,
                lam_rot=config.lam_rot,
                lam0_rot=config.lam0_rot,
                w0_rot=config.w0_rot,
                rot_err_hold=config.rot_err_hold,
                q_rest=q_rest.copy(),
                max_dq_per_joint=config.max_dq_per_joint,
            )
            self._arms[hand] = {
                "solver": arm_solver,
                "mapper": ClutchPoseMapper(
                    R=self._r_calib.copy(),
                    scale=config.scale_translation,
                    scale_rotation=config.scale_rotation,
                    rot_reach_limit=config.rot_reach_limit,
                    pos_reach_limit=config.pos_reach_limit,
                ),
                "qpos": qpos_init,
                "last_grip": False,
                "last_precision": False,  # tracks A/X button — re-anchor on edge
                "trigger": 0.0,
                "engaged": False,
                "haptic": 0.0,  # smoothed 0..1, published on /quest/<hand>/haptic
                "needs_reanchor": False,  # set during stale → re-anchor on recovery
                "pos_filt": None,  # EMA-smoothed controller position
                "quat_filt": None,  # EMA-smoothed controller orientation (wxyz)
                "last_rest_button": False,
                "ramp_active": False,
                "ramp_start_q": np.zeros(ARM_DOFS),
                "ramp_target_q": q_rest.copy(),
                "ramp_start_t": 0.0,
            }

        self._haptic_pubs = {h: node.create_publisher(Float32, f"/quest/{h}/haptic", 10)
                             for h in self._hands}
        self._engaged_pubs = {h: node.create_publisher(Bool, f"/quest/{h}/engaged", 10)
                              for h in self._hands}
        node.create_subscription(String, "/quest/settings", self._on_settings, SETTINGS_QOS)
        self.quest = QuestInput(node, self._hands)

    def xr_snapshot(self) -> dict | None:
        """Latest raw Quest poses plus engagement, for dataset debug logging.

        Returns None before the first controller pose; otherwise {age_s, viewer,
        controllers, engaged} with poses in Quest local-floor as received.
        """
        ctrls = {h: c for h in self._hands if (c := self.quest.controller(h)) is not None}
        if not ctrls:
            return None
        head = self.quest.head()
        with self._lock:
            engaged = {h: bool(self._arms[h]["engaged"]) for h in self._hands}
        return {
            "age_s": min(c.age_s for c in ctrls.values()),
            "viewer": None if head is None else {"position": head[0], "orientation": head[1]},
            "controllers": {h: {"position": c.pos, "orientation": c.quat_xyzw, "values": c.axes}
                            for h, c in ctrls.items()},
            "engaged": engaged,
        }

    def seed(self, obs: dict[str, float]) -> None:
        """Reset qpos / gripper / engagement to match a `.pos`-keyed dict.

        Also force-disengages, so a grip released while nothing was calling
        step() cannot leave a stale mapper anchor behind.

        obs: robot observation or commanded action. Prefer the command at
            handoff — it keeps a gripper squeezing an object closed rather than
            recording its blocked-open measurement. Missing keys are skipped.
        """
        with self._lock:
            for hand in self._hands:
                arm = self._arms[hand]
                for j in range(ARM_DOFS):
                    key = f"{hand}_joint_{j + 1}.pos"
                    if key in obs:
                        arm["qpos"][j] = float(obs[key])
                gkey = f"{hand}_gripper.pos"
                if gkey in obs:
                    g = float(np.clip(obs[gkey], 0.0, 1.0))
                    arm["trigger"] = g
                    grip_qpos = GRIPPER_QPOS_OPEN + g * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
                    arm["qpos"][6] = grip_qpos
                    arm["qpos"][7] = grip_qpos
                # Cancel any in-flight ramp: unguarded, it would overwrite the seed.
                # `last_rest_button` is left alone so a held thumbstick can't re-arm it.
                arm["ramp_active"] = False
                arm["mapper"].disengage()
                arm["engaged"] = False
                arm["last_grip"] = False
                arm["pos_filt"] = None
                arm["quat_filt"] = None
                arm["needs_reanchor"] = False

    def step(self) -> dict[str, float]:
        """Advance every arm one tick from the latest Quest state. Returns the action."""
        with self._lock:
            head = self.quest.head()
            ctrls = {hand: self.quest.controller(hand) for hand in self._hands}
            self._log_stream_health(ctrls)
            self._viewer_seen = head is not None
            self._mapping_r_calib = _headset_relative_calibration(
                self._r_calib, None if head is None else head[1])
            for hand, ctrl in ctrls.items():
                self._update_arm(hand, ctrl)
            for hand in self._hands:
                self._haptic_pubs[hand].publish(Float32(data=float(self._arms[hand]["haptic"])))
                self._engaged_pubs[hand].publish(Bool(data=bool(self._arms[hand]["engaged"])))
            return self._build_action()

    # ---------- internals ----------

    def _log_stream_health(self, ctrls: dict[str, Controller | None]) -> None:
        ages = [c.age_s for c in ctrls.values() if c is not None]
        if not ages:
            return
        self._gap_samples.append(max(ages))
        now = time.perf_counter()
        if now - self._gap_report_t > 10.0:
            g = np.asarray(self._gap_samples)
            self._log.info(
                f"xr stream health: age p50 {np.median(g) * 1e3:.0f}ms  "
                f"p99 {np.percentile(g, 99) * 1e3:.0f}ms  max {g.max() * 1e3:.0f}ms  "
                f"stale(>50ms) {(g > 0.05).mean() * 100:.1f}% of {len(g)} ticks")
            self._gap_samples.clear()
            self._gap_report_t = now

    def _build_action(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for hand in self._hands:
            arm = self._arms[hand]
            for j in range(ARM_DOFS):
                out[f"{hand}_joint_{j + 1}.pos"] = float(arm["qpos"][j])
            # LeRobot convention: gripper.pos in [0, 1], same polarity as trigger.
            out[f"{hand}_gripper.pos"] = float(np.clip(arm["trigger"], 0.0, 1.0))
        return out

    def _stow_arm(self, hand: str, arm: ArmState) -> None:
        """Thumbstick STOW: ramp this arm to rest_qpos without blocking the loop.

        _update_arm advances the ramp a step per tick, so the other hand stays
        live and a second stow press is seen immediately.
        """
        target = (
            self.config.rest_qpos_left if hand == "left" else self.config.rest_qpos_right
        )
        arm["ramp_start_q"] = arm["qpos"][:ARM_DOFS].copy()
        arm["ramp_target_q"] = np.asarray(target, dtype=float)
        arm["ramp_start_t"] = time.perf_counter()
        arm["ramp_active"] = True
        arm["haptic"] = 0.0
        self._log.info(
            f"{hand} STOW: ramping home over {self.config.rest_ramp_duration_s:.1f}s "
            f"(non-blocking, target={arm['ramp_target_q'].round(3).tolist()})")

    def _anchor_mapper(
        self, hand: str, arm: ArmState, pos: np.ndarray, quat_wxyz: np.ndarray, label: str
    ) -> None:
        """Capture the mapper's engage frame at the current robot+controller state.

        label: log tag for the calling path (ENGAGE, RE-ANCHOR, PRECISION, ...).
        """
        # Anchors at the last COMMANDED qpos, never the measurement: measured
        # lags command by the tracking error. See DESIGN.md#teleoperation.
        ee_pos, ee_quat = arm["solver"].fk(arm["qpos"])
        j4_pos = arm["solver"].j4_anchor_xpos()
        arm["mapper"].set_R(self._mapping_r_calib)
        arm["mapper"].engage(pos, quat_wxyz, ee_pos, ee_quat, pivot_armbase=j4_pos)
        arm["engaged"] = True
        # rel = Hᵀ (headset-yaw frame); identity here with viewer=True means no correction.
        rel = self._r_calib.T @ self._mapping_r_calib
        yaw_deg = float(np.degrees(np.arctan2(rel[2, 0], rel[2, 2])))
        self._log.info(
            f"{hand} clutch {label}  ee_pos={ee_pos.round(3)}  j4_pos={j4_pos.round(3)}  "
            f"headset_yaw={yaw_deg:.0f}°  viewer={self._viewer_seen}")

    def _advance_ramp(
        self,
        hand: str,
        arm: ArmState,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        pose_is_fresh: bool,
    ) -> bool:
        """Advance a stow trajectory even when the XR stream is stale."""
        if not arm["ramp_active"]:
            return False
        duration = max(1e-3, float(self.config.rest_ramp_duration_s))
        elapsed = time.perf_counter() - arm["ramp_start_t"]
        t = min(1.0, elapsed / duration)
        arm["qpos"][:ARM_DOFS] = arm["ramp_start_q"] + t * (
            arm["ramp_target_q"] - arm["ramp_start_q"]
        )
        if t >= 1.0:
            arm["ramp_active"] = False
            if arm["engaged"] and pose_is_fresh:
                self._anchor_mapper(hand, arm, pos, quat_wxyz, "RAMP-DONE")
            elif arm["engaged"]:
                arm["needs_reanchor"] = True
        arm["haptic"] *= 0.6
        return True

    def _update_arm(self, hand: str, ctrl: Controller | None) -> None:
        if ctrl is None:
            return
        arm = self._arms[hand]

        pos_raw = ctrl.pos
        ox, oy, oz, ow = ctrl.quat_xyzw
        quat_raw = np.array([ow, ox, oy, oz])

        # Stow is the thumbstick, not B/Y: B/Y arms/disarms the teleop node and
        # toggles episodes when recording, so one press would do both jobs.
        rest_btn = xr_buttons.pressed(ctrl.buttons, xr_buttons.THUMBSTICK)
        if rest_btn and not arm["last_rest_button"] and not arm["ramp_active"]:
            arm["last_rest_button"] = rest_btn
            self._stow_arm(hand, arm)
            return
        arm["last_rest_button"] = rest_btn

        pose_is_fresh = ctrl.age_s <= xr_buttons.XR_FRAME_STALE_TIMEOUT_S
        if self._advance_ramp(hand, arm, pos_raw, quat_raw, pose_is_fresh):
            return

        # Stale pose: skip engage/disengage/IK and flag a re-anchor, so the
        # engage-delta restarts at zero and there is no catch-up motion.
        if not pose_is_fresh:
            if arm["engaged"] and not arm["needs_reanchor"]:
                arm["needs_reanchor"] = True
                self._log.warning(f"{hand} controller pose stale ({ctrl.age_s:.2f}s) — pausing")
            arm["haptic"] *= 0.6  # decay so vibration doesn't linger
            return

        # nlerp with a hemisphere check — without it the lerp can drift to the
        # antipodal point and flip. First tick copies raw instead of lerping.
        alpha = float(self.config.pose_filter_alpha)
        if arm["pos_filt"] is None or arm["quat_filt"] is None:
            arm["pos_filt"] = pos_raw.copy()
            arm["quat_filt"] = quat_raw.copy()
        else:
            arm["pos_filt"] = (1.0 - alpha) * arm["pos_filt"] + alpha * pos_raw
            q_in = quat_raw if np.dot(arm["quat_filt"], quat_raw) >= 0.0 else -quat_raw
            qf = (1.0 - alpha) * arm["quat_filt"] + alpha * q_in
            arm["quat_filt"] = qf / np.linalg.norm(qf)
        pos = arm["pos_filt"]
        quat_wxyz = arm["quat_filt"]

        grip = xr_buttons.pressed(ctrl.buttons, xr_buttons.GRIP)
        trigger = xr_buttons.value(ctrl.axes, xr_buttons.TRIGGER)
        precision = xr_buttons.pressed(ctrl.buttons, xr_buttons.A_X)
        # Precision scales the mapper gains down; re-anchor on either edge so the
        # accumulated delta is not reinterpreted under the new scale.
        if precision != arm["last_precision"] and arm["engaged"]:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, "PRECISION" if precision else "FULL-SCALE"
            )
        arm["last_precision"] = precision
        scale_factor = self.config.precision_factor if precision else 1.0
        arm["mapper"].scale = self.config.scale_translation * scale_factor
        arm["mapper"].scale_rotation = self.config.scale_rotation * scale_factor
        arm["mapper"].rot_reach_limit = self.config.rot_reach_limit
        arm["mapper"].pos_reach_limit = self.config.pos_reach_limit

        # Re-anchor after a stale window if the grip is still held; a release
        # during the stall falls through to the normal disengage edge below.
        if arm["needs_reanchor"] and arm["engaged"] and grip:
            self._anchor_mapper(hand, arm, pos, quat_wxyz, "RE-ANCHOR")
        arm["needs_reanchor"] = False

        # Edge-detect clutch.
        if grip and not arm["last_grip"]:
            self._anchor_mapper(hand, arm, pos, quat_wxyz, "ENGAGE")
        elif not grip and arm["last_grip"]:
            arm["mapper"].disengage()
            arm["engaged"] = False
            # Dropped on release so the next engage has no lerp-lag from old state.
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            self._log.info(f"{hand} clutch RELEASE")
        arm["last_grip"] = grip

        # Engaged only: otherwise stray trigger pressure would move the gripper
        # outside an active intervention.
        if arm["engaged"]:
            arm["trigger"] = trigger
            grip_qpos = GRIPPER_QPOS_OPEN + trigger * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
            arm["qpos"][6] = grip_qpos
            arm["qpos"][7] = grip_qpos

        # FK at the last commanded qpos; feeds the mapper's reach limits.
        ee_pos_now, ee_quat_now = arm["solver"].fk(arm["qpos"])
        out = arm["mapper"].target(pos, quat_wxyz, ee_pos_now, ee_quat_now)
        if out is not None:
            tgt_pos, tgt_quat = out
            arm["qpos"][:ARM_DOFS] = arm["solver"].solve(tgt_pos, tgt_quat, arm["qpos"])

            # Max of four 0..1 cues: limit pressure, reach error, singularity and
            # gimbal proximity — near gimbal the wrist goes sluggish, not limited.
            solver = arm["solver"]
            i_limit = min(1.0, max(0.0, (solver.last_limit_pressure - 0.05) / 0.25))
            i_reach = min(1.0, max(0.0, (solver.last_pos_err_norm - 0.03) / 0.10))
            i_singular = max(0.0, (solver.last_singularity_proximity - 0.95) / 0.2)
            i_gimbal = min(1.0, max(0.0, (solver.last_wrist_gimbal_proximity - 0.5) / 0.5))
            raw_intensity = max(i_limit, i_reach, i_singular, i_gimbal)
            arm["haptic"] = 0.6 * arm["haptic"] + 0.4 * raw_intensity
        else:
            arm["haptic"] *= 0.6  # decay so vibration doesn't linger after release

    # ---------- Live config (from the headset page) ----------

    # Page-tunable knobs and their hard safety bounds; out-of-range values are
    # clamped with a warning so a runaway slider can't reach unsafe gains.
    _LIVE_CONFIG_BOUNDS: dict[str, tuple[float, float]] = {
        "scale_translation": (0.1, 10.0),
        "scale_rotation": (0.1, 10.0),
        "pose_filter_alpha": (0.05, 1.0),
        "precision_factor": (0.05, 1.0),
        # Per-group Δq cap (rad/tick): joints 1-3 and 4-6. The side-effect below
        # rebuilds the 6-vector and pushes it into both solvers.
        "max_dq_per_joint_scalar_pos": (0.005, 0.50),
        "max_dq_per_joint_scalar_rot": (0.005, 0.50),
    }

    def _on_settings(self, msg: String) -> None:
        self._apply_config_update(json.loads(msg.data))

    def _apply_config_update(self, cfg: dict) -> None:
        """Apply the page's settings snapshot.

        Mutates `self.config` in place under the lock; `_update_arm` re-reads it
        every tick, so the new value lands on the next solve.

        cfg: {field: value} for keys in _LIVE_CONFIG_BOUNDS; others are ignored.
        """
        applied: dict[str, float] = {}
        with self._lock:
            for key, (lo, hi) in self._LIVE_CONFIG_BOUNDS.items():
                if key not in cfg:
                    continue
                try:
                    v = float(cfg[key])
                except (TypeError, ValueError):
                    self._log.warning(f"settings: {key}={cfg[key]!r} not a number; ignoring")
                    continue
                if not (lo <= v <= hi):
                    self._log.warning(
                        f"settings: {key}={v:.3f} out of range [{lo:.2f}, {hi:.2f}]; clamping")
                    v = max(lo, min(hi, v))
                setattr(self.config, key, v)
                applied[key] = v
            # Once after the loop, so a payload touching both groups writes once.
            if (
                "max_dq_per_joint_scalar_pos" in applied
                or "max_dq_per_joint_scalar_rot" in applied
            ):
                pos = float(self.config.max_dq_per_joint_scalar_pos)
                rot = float(self.config.max_dq_per_joint_scalar_rot)
                arr = [pos] * 3 + [rot] * 3
                self.config.max_dq_per_joint = arr
                for hand in self._hands:
                    self._arms[hand]["solver"].max_dq_per_joint = np.asarray(
                        arr, dtype=float
                    ).copy()
        if applied:
            self._log.info(
                "settings applied: " + ", ".join(f"{k}={v:.3f}" for k, v in applied.items()))
