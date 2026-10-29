import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from sparklab_gello.reader import GelloInputError, GelloReader

CONFIG = Path(__file__).resolve().parents[1] / "config"
JOINTS = [f"joint_{j}" for j in range(1, 7)]


class FakeBus:
    """Scripted stand-in for the leader's FeetechMotorsBus; no serial port."""

    is_connected = True

    def __init__(self) -> None:
        self.deg = {n: 0.0 for n in JOINTS}
        self.raw = 2045
        self.error: Exception | None = None
        self.delay_s = 0.0
        self.retries_seen: list[int] = []

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        self.retries_seen.append(num_retry)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if not normalize:
            return {"gripper": self.raw}
        return {n: self.deg[n] for n in motors if n in self.deg}


def connected_reader(bus: FakeBus) -> GelloReader:
    r = GelloReader(CONFIG / "yam_active_leader_eeprom.json", CONFIG / "gello_to_yam.yaml")
    r._leader = SimpleNamespace(bus=bus)
    r._half_range_deg = {n: 90.0 for n in JOINTS}
    r._raw_max = 4095
    return r


class GelloInputTest(unittest.TestCase):
    """Every reading is fresh and validated; a bad one raises and produces no target."""

    def setUp(self) -> None:
        self.bus = FakeBus()
        self.t = connected_reader(self.bus)

    def test_valid_reading_is_six_joints_and_a_gripper(self) -> None:
        q, g = self.t.read()
        self.assertEqual(q.shape, (6,))
        self.assertEqual(g, 0.0)

    def test_mapping_is_sign_times_radians_of_offset(self) -> None:
        mapping = yaml.safe_load((CONFIG / "gello_to_yam.yaml").read_text())["joints"]
        self.bus.deg["joint_1"] = 10.0
        self.bus.deg["joint_2"] = -30.0
        self.bus.deg["joint_4"] = -20.0
        q, _ = self.t.read()
        for j, n in enumerate(JOINTS):
            m = mapping[n]
            expected = m["sign"] * np.radians(self.bus.deg[n] - m["zero_deg"])
            self.assertAlmostEqual(q[j], np.clip(expected, *self.t._limits[j]))

    def test_gripper_maps_raw_a_to_0_raw_b_to_1_and_clips(self) -> None:
        grip = yaml.safe_load((CONFIG / "gello_to_yam.yaml").read_text())["gripper"]
        for raw, expected in ((grip["raw_a"], 0.0), (grip["raw_b"], 1.0),
                              ((grip["raw_a"] + grip["raw_b"]) / 2, 0.5),
                              (grip["raw_a"] + 100, 0.0), (grip["raw_b"] - 100, 1.0)):
            with self.subTest(raw=raw):
                self.bus.raw = raw
                self.assertAlmostEqual(self.t.read()[1], expected)

    def test_repeated_identical_readings_are_not_an_error(self) -> None:
        for _ in range(200):
            self.t.read()

    def test_read_failure_raises(self) -> None:
        self.t.read()
        for err in (ConnectionError("no status packet"), OSError("device disconnected")):
            with self.subTest(err=type(err).__name__):
                self.bus.error = err
                with self.assertRaises(GelloInputError):
                    self.t.read()
        self.assertEqual(set(self.bus.retries_seen), {self.t.read_retries})

    def test_non_finite_missing_and_out_of_range_readings_fail(self) -> None:
        cases = {"nan": ("joint_2", float("nan")), "inf": ("joint_3", float("inf")),
                 "range": ("joint_4", 150.0)}
        for name, (joint, value) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.bus.deg[joint] = value
                with self.assertRaises(GelloInputError):
                    self.t.read()
        self.setUp()
        del self.bus.deg["joint_5"]
        with self.assertRaises(GelloInputError):
            self.t.read()
        self.setUp()
        self.bus.raw = 5000
        with self.assertRaises(GelloInputError):
            self.t.read()

    def test_small_range_excess_within_margin_is_accepted(self) -> None:
        self.bus.deg["joint_4"] = 95.2
        self.t.read()

    def test_jump_fails(self) -> None:
        self.t.read()
        self.bus.deg["joint_1"] = 60.0
        with self.assertRaisesRegex(GelloInputError, "jumped"):
            self.t.read()

    def test_slow_motion_after_a_pause_is_not_a_jump(self) -> None:
        self.t.read()
        time.sleep(0.2)
        self.bus.deg["joint_1"] = 30.0
        self.t.read()

    def test_late_read_fails(self) -> None:
        self.bus.delay_s = 0.06
        with self.assertRaisesRegex(GelloInputError, "took"):
            self.t.read()


class HomingBus(FakeBus):
    """Records writes; like the leader's servos, a goal counts only while torque is on."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple] = []
        self.torque = False

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0):
        self.log.append((data_name, dict(values), normalize))
        if data_name == "Goal_Position" and self.torque:
            if "gripper" in values:
                self.raw = values["gripper"]
            else:
                self.deg.update(values)

    def enable_torque(self, motors=None, num_retry=0):
        self.log.append(("enable", list(motors)))
        self.torque = True

    def disable_torque(self, motors=None, num_retry=0):
        self.log.append(("disable", list(motors)))
        self.torque = False


class GelloHomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = HomingBus()
        self.t = connected_reader(self.bus)
        self.t.max_input_speed_deg_s = 1e9  # the fake reaches its goal instantly

    def test_home_holds_present_pose_through_enable_then_sends_goal_and_releases(self) -> None:
        self.bus.deg["joint_1"] = -40.0
        self.bus.raw = 1500
        err = self.t.home(0.0)
        names = [e[0] for e in self.bus.log]
        self.assertEqual(names, ["Goal_Velocity", "Torque_Limit", "Goal_Position", "Goal_Position",
                                 "enable", "Goal_Position", "Goal_Position", "disable"])
        self.assertTrue(all(v > 0 for v in self.bus.log[0][1].values()))
        self.assertIn("gripper", self.bus.log[0][1])
        self.assertEqual(self.bus.log[2][1]["joint_1"], -40.0)
        self.assertEqual(self.bus.log[3][1], {"gripper": 1500})
        self.assertIn("gripper", self.bus.log[-1][1])
        self.assertFalse(self.bus.torque)
        self.assertLess(err, 1.0)

    def test_home_goal_maps_to_yam_zero_and_the_given_gripper(self) -> None:
        self.bus.raw = 1500
        self.t.home(0.0)
        q, g = self.t.read()
        np.testing.assert_allclose(q, np.zeros(6), atol=1e-12)
        self.assertEqual(g, 0.0)
        self.t.home(1.0)
        self.assertAlmostEqual(self.t.read()[1], 1.0)

    def test_home_releases_torque_on_failure_and_on_timeout(self) -> None:
        self.bus.error = OSError("bus gone")
        with self.assertRaises(GelloInputError):
            self.t.home(0.0)
        self.assertFalse(self.bus.torque)
        self.bus.error = None
        self.bus.deg = {n: 0.0 for n in JOINTS}
        self.bus.sync_write = lambda *a, **kw: None  # a joint that never moves
        self.assertGreater(self.t.home(0.0, timeout_s=0.1), 2.0)
        self.assertFalse(self.bus.torque)


if __name__ == "__main__":
    unittest.main()
