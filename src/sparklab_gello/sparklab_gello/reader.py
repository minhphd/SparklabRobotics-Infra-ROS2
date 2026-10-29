"""One YAM Active Leader (GELLO), read-only, as YAM joint targets.

Reads the leader through its own driver: no ``calibrate()``, ``configure()``,
``calibrate_trigger()`` or torque write. Driver degrees become YAM radians
through ``gello_to_yam.yaml``; the lever's raw ticks become a gripper target 0..1.

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
    """

    def __init__(self, calibration_path: str | Path, mapping_path: str | Path, port: str = "",
                 read_retries: int = 1, max_read_latency_s: float = 0.1,
                 max_input_speed_deg_s: float = 720.0, range_margin_deg: float = 10.0) -> None:
        self.calibration_path = Path(calibration_path)
        self.port = port
        self.read_retries = read_retries
        self.max_read_latency_s = max_read_latency_s
        self.max_input_speed_deg_s = max_input_speed_deg_s
        self.range_margin_deg = range_margin_deg
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

    def read(self) -> tuple[np.ndarray, float]:
        """One validated reading as YAM targets.

        Returns: joint positions (6,) in rad, clipped to the model's limits, and
            the gripper target 0 closed .. 1 open.
        """
        deg, raw = self._read()
        q = np.empty(ARM_JOINTS)
        for j in range(ARM_JOINTS):
            m = self._joint_map[NAMES[j]]
            q[j] = m["sign"] * np.radians(deg[NAMES[j]] - m["zero_deg"])
        q = np.clip(q, self._limits[:, 0], self._limits[:, 1])
        g = (raw - self._raw_a) / (self._raw_b - self._raw_a)
        # + 0.0 turns the -0.0 that raw == raw_a yields into 0.0.
        return q, float(np.clip(g, 0.0, 1.0)) + 0.0

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
            self._leader.bus.disconnect(disable_torque=False)
            self._leader = None
