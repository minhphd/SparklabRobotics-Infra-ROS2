"""One YAM Active Leader (GELLO) as YAM joint targets.

Reads the leader through its own driver: no ``calibrate()``, ``configure()``,
``calibrate_trigger()`` or EEPROM write; torque is on only inside ``home()``,
while ``mirror()`` has it follow the arm, and while ``hold`` locks it.
Driver degrees become YAM radians through ``gello_to_yam.yaml``; the lever's raw
ticks become a gripper target 0..1.

Every ``read()`` is a fresh bus read. A failed, late, malformed or implausible
reading raises ``GelloInputError`` and returns nothing, so nothing stale is ever
returned.
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sparklab_yam.constants import ARM_JOINTS
from sparklab_yam.model.kinematics import ARM_XML

PORT_GLOBS = ("/dev/tty.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*")
NAMES = [f"joint_{j}" for j in range(1, ARM_JOINTS + 1)]
HOME_SPEED = 300  # raw Goal_Velocity, steps/s (~26 deg/s); 0 would be the servo's maximum
HOME_TORQUE = 500  # raw Torque_Limit of 1000, the leader driver's own approach limit
HOME_TOL_DEG = 2.0  # the servos settle ~1 deg short of the goal
HOLD_AFTER_S = 1.0  # every joint within HOLD_STILL_DEG this long -> hold
HOLD_STILL_DEG = 0.5
HOLD_RELEASE_DEG = 2.0  # any joint pushed this far from the hold -> release
MIRROR_TORQUE = 300  # raw Torque_Limit while mirroring, below homing's: a hand overrides it
MIRROR_STEP_DEG = 0.5  # rewrite the mirror goal only past this, so the arm's jitter is not echoed


class GelloInputError(RuntimeError):
    """The leader reading cannot be trusted; no target was produced."""


def find_leader_port() -> str:
    """The servo adapter's serial port, when exactly one candidate exists."""
    ports = sorted(p for g in PORT_GLOBS for p in glob.glob(g))
    if len(ports) != 1:
        raise RuntimeError(f"Expected exactly one servo adapter port, found {ports or 'none'}; "
                           "set the port explicitly.")
    return ports[0]


class GelloReader:
    """One GELLO leader.

    calibration_path: leader calibration in LeRobot's format; must match its EEPROM.
    mapping_path: GELLO -> YAM sign/zero/gripper mapping.
    port: servo adapter serial port; auto-detected when empty.
    read_retries: immediate retries inside one read before it fails (the driver's num_retry).
    max_read_latency_s: a read slower than this fails.
    max_input_speed_deg_s: a joint reading changing faster than this fails.
    range_margin_deg: allowed excess beyond the EEPROM calibration range.
    hold: lock the arm joints where they are once the leader is still; a push releases them.
    """

    def __init__(self, calibration_path: str | Path, mapping_path: str | Path, port: str = "",
                 read_retries: int = 1, max_read_latency_s: float = 0.1,
                 max_input_speed_deg_s: float = 720.0, range_margin_deg: float = 10.0,
                 hold: bool = False) -> None:
        self.calibration_path = Path(calibration_path)
        self.port = port
        self.read_retries = read_retries
        self.max_read_latency_s = max_read_latency_s
        self.max_input_speed_deg_s = max_input_speed_deg_s
        self.range_margin_deg = range_margin_deg
        self.hold = hold
        mapping = yaml.safe_load(Path(mapping_path).read_text())
        self._joint_map = mapping["joints"]
        missing = [k for k, v in self._joint_map.items() if v.get("sign") is None]
        if missing:
            raise ValueError(f"{mapping_path}: sign not set for {missing}")
        self._raw_a = float(mapping["gripper"]["raw_a"])
        self._raw_b = float(mapping["gripper"]["raw_b"])
        self._limits = mujoco.MjModel.from_xml_path(str(ARM_XML)).jnt_range[:ARM_JOINTS].copy()
        self._leader = None
        self._half_range_deg: dict[str, float] = {}
        self._raw_max = 0
        self._prev: tuple[float, dict[str, float]] | None = None
        self._last: tuple[dict[str, float], float] | None = None
        self._mirroring = False
        self._mirror_goal: tuple[dict[str, float], int] | None = None
        self._still: tuple[float, dict[str, float]] | None = None
        self._held: dict[str, float] | None = None

    def connect(self) -> str:
        """Open the bus read-only and check the calibration file against the EEPROM.

        Returns: the port opened.
        """
        # Imported after mujoco on purpose: on macOS, torch's bundled libomp loaded
        # first aborts the process (OMP Error #15) once the conda libomp follows.
        from lerobot_teleoperator_yamactiveleader import (
            YamActiveLeaderTeleoperator,
            YamActiveLeaderTeleoperatorConfig,
        )

        port = self.port or find_leader_port()
        leader = YamActiveLeaderTeleoperator(
            YamActiveLeaderTeleoperatorConfig(port=port), active_trigger=False
        )
        leader._load_calibration(self.calibration_path)
        leader.bus.connect()
        # In memory only; write_calibration() would rewrite EEPROM.
        leader.bus.calibration = leader.calibration
        if not leader.bus.is_calibrated:
            leader.bus.disconnect(disable_torque=False)
            raise RuntimeError(f"{self.calibration_path} does not match the leader's EEPROM.")
        bus = leader.bus
        for name, m in bus.motors.items():
            res = bus.model_resolution_table[m.model]
            cal = leader.calibration[name]
            self._half_range_deg[name] = (cal.range_max - cal.range_min) / 2 * 360 / (res - 1)
        self._raw_max = bus.model_resolution_table[bus.motors["gripper"].model] - 1
        self._prev = None
        self._leader = leader
        return port

    def home(self, q: np.ndarray, gripper: float, timeout_s: float = 5.0) -> float:
        """Drive the leader to the pose that maps to q and the lever to gripper, then release.

        q: YAM joint positions (6,) in rad; the robot's measured pose.
        gripper: lever target, 0 closed .. 1 open; the robot's measured gripper.
        Returns: the largest remaining error in degrees when torque was released.
        """
        bus = self._leader.bus
        motors = NAMES + ["gripper"]
        goal, raw_goal = self._goal(q, gripper)
        bus.sync_write("Goal_Velocity", {n: HOME_SPEED for n in motors}, normalize=False)
        bus.sync_write("Torque_Limit", {n: HOME_TORQUE for n in motors}, normalize=False)
        try:
            # These servos drop a goal written while torque is off; hold the present
            # pose through the enable so a firmware that keeps it cannot jump.
            deg, raw = self._read()
            bus.sync_write("Goal_Position", deg)
            bus.sync_write("Goal_Position", {"gripper": int(raw)}, normalize=False)
            bus.enable_torque(motors)
            bus.sync_write("Goal_Position", goal)
            bus.sync_write("Goal_Position", {"gripper": raw_goal}, normalize=False)
            deadline = time.monotonic() + timeout_s
            while True:
                deg, raw = self._read()
                err = max(max(abs(deg[n] - goal[n]) for n in NAMES),
                          abs(raw - raw_goal) * 360 / (self._raw_max + 1))
                if err < HOME_TOL_DEG or time.monotonic() > deadline:
                    return err
                time.sleep(0.05)
        finally:
            bus.disable_torque(motors)

    def _goal(self, q: np.ndarray, gripper: float) -> tuple[dict[str, float], int]:
        """Leader joint degrees and lever raw ticks that read back as q and gripper."""
        goal = {n: float(self._joint_map[n]["zero_deg"] + self._joint_map[n]["sign"] * np.degrees(q[j]))
                for j, n in enumerate(NAMES)}
        raw = int(round(self._raw_a + np.clip(gripper, 0.0, 1.0) * (self._raw_b - self._raw_a)))
        return goal, raw

    def mirror(self, q: np.ndarray, gripper: float) -> None:
        """Servo the leader, at MIRROR_TORQUE, to the pose that reads back as q, the lever to gripper.

        The first call enables torque, holding the last reading through the enable as home()
        does; later calls rewrite the goal only once it has moved MIRROR_STEP_DEG.
        q: YAM joint positions (6,) in rad; gripper: 0 closed .. 1 open.
        """
        bus = self._leader.bus
        motors = NAMES + ["gripper"]
        goal, raw_goal = self._goal(q, gripper)
        try:
            if not self._mirroring:
                deg, raw = self._last
                bus.sync_write("Goal_Velocity", {n: HOME_SPEED for n in motors}, normalize=False)
                bus.sync_write("Torque_Limit", {n: MIRROR_TORQUE for n in motors}, normalize=False)
                bus.sync_write("Goal_Position", deg)
                bus.sync_write("Goal_Position", {"gripper": int(raw)}, normalize=False)
                bus.enable_torque(motors)
                self._mirroring, self._mirror_goal, self._held, self._still = True, None, None, None
            last = self._mirror_goal
            if (last is None or max(abs(goal[n] - last[0][n]) for n in NAMES) > MIRROR_STEP_DEG
                    or abs(raw_goal - last[1]) * 360 / (self._raw_max + 1) > MIRROR_STEP_DEG):
                bus.sync_write("Goal_Position", goal)
                bus.sync_write("Goal_Position", {"gripper": raw_goal}, normalize=False)
                self._mirror_goal = (goal, raw_goal)
        except (ConnectionError, OSError) as e:
            raise GelloInputError(f"GELLO mirror write failed: {e}") from e

    def let_down(self) -> None:
        """Fold a mirroring leader to YAM zero, where the drivers park the arms, then let it go limp.

        The lever stays where it is. Does nothing to a leader that is already limp.
        """
        if not self._mirroring:
            return
        self.home(np.zeros(ARM_JOINTS), (self._last[1] - self._raw_a) / (self._raw_b - self._raw_a))
        self._mirroring = False

    def release(self) -> None:
        """Let a mirroring leader go limp; does nothing otherwise."""
        if not self._mirroring:
            return
        try:
            self._leader.bus.disable_torque(NAMES + ["gripper"])
        except (ConnectionError, OSError) as e:
            raise GelloInputError(f"GELLO release failed: {e}") from e
        self._mirroring = False

    def read(self) -> tuple[np.ndarray, float]:
        """One validated reading as YAM targets.

        Returns: joint positions (6,) in rad, clipped to the model's limits, and
            the gripper target 0 closed .. 1 open.
        """
        deg, raw = self._read()
        if self.hold and not self._mirroring:
            self._update_hold(deg)
        q = np.empty(ARM_JOINTS)
        for j in range(ARM_JOINTS):
            m = self._joint_map[NAMES[j]]
            q[j] = m["sign"] * np.radians(deg[NAMES[j]] - m["zero_deg"])
        q = np.clip(q, self._limits[:, 0], self._limits[:, 1])
        g = (raw - self._raw_a) / (self._raw_b - self._raw_a)
        # + 0.0 turns the -0.0 that raw == raw_a yields into 0.0.
        return q, float(np.clip(g, 0.0, 1.0)) + 0.0

    def _update_hold(self, deg: dict[str, float]) -> None:
        bus = self._leader.bus
        try:
            if self._held is not None:
                if max(abs(deg[n] - self._held[n]) for n in NAMES) > HOLD_RELEASE_DEG:
                    bus.disable_torque(NAMES)
                    self._held, self._still = None, None
                return
            now = time.monotonic()
            if self._still is None or max(abs(deg[n] - self._still[1][n]) for n in NAMES) > HOLD_STILL_DEG:
                self._still = (now, dict(deg))
            elif now - self._still[0] > HOLD_AFTER_S:
                bus.sync_write("Torque_Limit", {n: HOME_TORQUE for n in NAMES}, normalize=False)
                # Present pose on both sides of the enable, as in home().
                bus.sync_write("Goal_Position", deg)
                bus.enable_torque(NAMES)
                bus.sync_write("Goal_Position", deg)
                self._held = dict(deg)
        except (ConnectionError, OSError) as e:
            raise GelloInputError(f"GELLO hold write failed: {e}") from e

    def _read(self) -> tuple[dict[str, float], float]:
        """One validated reading: joint degrees and the lever's raw ticks."""
        bus = self._leader.bus
        t0 = time.perf_counter()
        try:
            deg = bus.sync_read("Present_Position", NAMES, num_retry=self.read_retries)
            raw = bus.sync_read("Present_Position", "gripper", normalize=False,
                                num_retry=self.read_retries).get("gripper")
        except (ConnectionError, OSError) as e:
            raise GelloInputError(
                f"GELLO read failed after {self.read_retries + 1} attempts: {e}") from e
        now = time.perf_counter()
        if now - t0 > self.max_read_latency_s:
            raise GelloInputError(f"GELLO read took {now - t0:.3f} s "
                                  f"(limit {self.max_read_latency_s:.3f} s)")
        self._validate(deg, raw, now)
        self._last = (deg, float(raw))
        return deg, float(raw)

    def _validate(self, deg: dict, raw, now: float) -> None:
        missing = [n for n in NAMES if n not in deg]
        if missing or raw is None:
            raise GelloInputError(f"GELLO reading is missing {missing or ['gripper']}")
        for n in NAMES:
            v = deg[n]
            if not np.isfinite(v):
                raise GelloInputError(f"GELLO {n} reading is not finite: {v}")
            limit = self._half_range_deg[n] + self.range_margin_deg
            if abs(v) > limit:
                raise GelloInputError(f"GELLO {n} reading {v:.1f} deg is outside the calibrated "
                                      f"range ±{self._half_range_deg[n]:.1f} deg "
                                      f"(+{self.range_margin_deg:.0f} margin)")
        if not np.isfinite(raw) or not 0 <= raw <= self._raw_max:
            raise GelloInputError(f"GELLO lever raw reading {raw} is outside 0..{self._raw_max}")
        if self._prev is not None:
            # Floor dt so two back-to-back reads cannot turn encoder noise into a "jump".
            dt = max(now - self._prev[0], 1 / 240)
            for n in NAMES:
                speed = abs(deg[n] - self._prev[1][n]) / dt
                if speed > self.max_input_speed_deg_s:
                    raise GelloInputError(
                        f"GELLO {n} jumped {deg[n] - self._prev[1][n]:+.1f} deg in {dt * 1e3:.0f} ms "
                        f"({speed:.0f} deg/s, limit {self.max_input_speed_deg_s:.0f})")
        self._prev = (now, {n: float(deg[n]) for n in NAMES})

    def disconnect(self) -> None:
        if self._leader is not None:
            try:
                if self._held is not None or self._mirroring:
                    self._leader.bus.disable_torque(NAMES + ["gripper"] if self._mirroring else NAMES)
            finally:
                self._held, self._mirroring = None, False
                self._leader.bus.disconnect(disable_torque=False)
                self._leader = None
