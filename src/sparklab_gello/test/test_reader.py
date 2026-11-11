import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from sparklab_gello import reader as reader_mod
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


class HoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = HomingBus()
        self.bus.disconnect = lambda disable_torque=True: None
        self.t = connected_reader(self.bus)
        self.t.hold = True
        patcher = mock.patch.object(reader_mod, "HOLD_AFTER_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def events(self) -> list[str]:
        return [e[0] for e in self.bus.log]

    def test_off_by_default_never_writes(self) -> None:
        self.t.hold = False
        for _ in range(5):
            self.t.read()
        self.assertEqual(self.bus.log, [])

    def test_still_leader_is_held_at_its_present_pose(self) -> None:
        self.bus.deg["joint_3"] = 12.0
        self.t.read()
        self.assertEqual(self.bus.log, [])
        self.t.read()
        self.assertEqual(self.events(), ["Torque_Limit", "Goal_Position", "enable", "Goal_Position"])
        self.assertEqual(self.bus.log[-1][1]["joint_3"], 12.0)
        self.assertTrue(self.bus.torque)

    def test_moving_leader_is_not_held(self) -> None:
        for k in range(5):
            self.bus.deg["joint_1"] = float(k)
            self.t.read()
        self.assertFalse(self.bus.torque)

    def test_small_push_keeps_the_hold_a_larger_one_releases(self) -> None:
        self.t.read()
        self.t.read()
        self.bus.deg["joint_5"] += 1.0
        self.t.read()
        self.assertTrue(self.bus.torque)
        self.bus.deg["joint_5"] += 1.5
        self.t.read()
        self.assertFalse(self.bus.torque)
        self.assertEqual(self.events()[-1], "disable")

    def test_disconnect_while_held_releases(self) -> None:
        self.t.read()
        self.t.read()
        self.t.disconnect()
        self.assertFalse(self.bus.torque)


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


Z = np.zeros(6)


class GelloHomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = HomingBus()
        self.t = connected_reader(self.bus)
        self.t.max_input_speed_deg_s = 1e9  # the fake reaches its goal instantly

    def test_home_holds_present_pose_through_enable_then_sends_goal_and_releases(self) -> None:
        self.bus.deg["joint_1"] = -40.0
        self.bus.raw = 1500
        err = self.t.home(Z, 0.0)
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

    def test_home_goal_maps_to_the_given_pose_and_gripper(self) -> None:
        q = np.array([0.3, 0.2, 0.5, -0.25, 0.1, -0.4])
        self.t.home(q, 0.5)
        got, g = self.t.read()
        np.testing.assert_allclose(got, q, atol=1e-9)
        self.assertAlmostEqual(g, 0.5, delta=0.01)

    def test_home_goal_maps_to_yam_zero_and_the_given_gripper(self) -> None:
        self.bus.raw = 1500
        self.t.home(Z, 0.0)
        q, g = self.t.read()
        np.testing.assert_allclose(q, np.zeros(6), atol=1e-12)
        self.assertEqual(g, 0.0)
        self.t.home(Z, 1.0)
        self.assertAlmostEqual(self.t.read()[1], 1.0)

    def test_home_releases_torque_on_failure_and_on_timeout(self) -> None:
        self.bus.error = OSError("bus gone")
        with self.assertRaises(GelloInputError):
            self.t.home(Z, 0.0)
        self.assertFalse(self.bus.torque)
        self.bus.error = None
        self.bus.deg = {n: 0.0 for n in JOINTS}
        self.bus.sync_write = lambda *a, **kw: None  # a joint that never moves
        self.assertGreater(self.t.home(Z, 0.0, timeout_s=0.1), 2.0)
        self.assertFalse(self.bus.torque)


class MirrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = HomingBus()
        self.bus.disconnect = lambda disable_torque=True: None
        self.t = connected_reader(self.bus)
        self.t.max_input_speed_deg_s = 1e9  # the fake reaches its goal instantly

    def test_first_mirror_holds_the_present_pose_through_enable_then_goes_to_the_arms_pose(self) -> None:
        self.bus.deg["joint_2"] = 15.0
        self.t.read()
        q = np.array([0.3, 0.2, 0.5, -0.25, 0.1, -0.4])
        self.t.mirror(q, 0.5)
        self.assertEqual([e[0] for e in self.bus.log], ["Goal_Velocity", "Torque_Limit", "Goal_Position",
                                                        "Goal_Position", "enable", "Goal_Position", "Goal_Position"])
        self.assertEqual(set(self.bus.log[1][1].values()), {reader_mod.MIRROR_TORQUE})
        self.assertEqual(self.bus.log[2][1]["joint_2"], 15.0)
        got, g = self.t.read()
        np.testing.assert_allclose(got, q, atol=1e-9)
        self.assertAlmostEqual(g, 0.5, delta=0.01)

    def test_goal_is_rewritten_only_past_the_step(self) -> None:
        self.t.read()
        self.t.mirror(Z, 0.0)
        n = len(self.bus.log)
        self.t.mirror(Z + np.radians(0.3), 0.0)
        self.assertEqual(len(self.bus.log), n)
        self.t.mirror(Z + np.radians(1.0), 0.0)
        self.assertEqual([e[0] for e in self.bus.log[n:]], ["Goal_Position", "Goal_Position"])

    def test_hold_stays_out_while_mirroring_and_release_lets_go_once(self) -> None:
        self.t.hold = True
        self.t.read()
        self.t.mirror(Z, 0.0)
        n = len(self.bus.log)
        with mock.patch.object(reader_mod, "HOLD_AFTER_S", 0.0):
            self.t.read()
            self.t.read()
        self.assertEqual(len(self.bus.log), n)
        self.t.release()
        self.assertEqual(self.bus.log[-1], ("disable", JOINTS + ["gripper"]))
        self.assertFalse(self.bus.torque)
        self.t.release()
        self.assertEqual(len(self.bus.log), n + 1)

    def test_let_down_folds_the_leader_to_zero_keeps_the_lever_and_lets_go(self) -> None:
        self.bus.raw = 1500
        self.t.read()
        self.t.let_down()
        self.assertEqual(self.bus.log, [])
        self.t.mirror(Z + 0.3, 0.5)
        lever = self.bus.raw
        self.t.read()
        self.t.let_down()
        np.testing.assert_allclose(self.t.read()[0], Z, atol=1e-9)
        self.assertEqual(self.bus.raw, lever)
        self.assertFalse(self.bus.torque)
        n = len(self.bus.log)
        self.t.let_down()
        self.assertEqual(len(self.bus.log), n)

    def test_disconnect_while_mirroring_lets_go(self) -> None:
        self.t.read()
        self.t.mirror(Z, 0.0)
        self.t.disconnect()
        self.assertFalse(self.bus.torque)


if __name__ == "__main__":
    unittest.main()
