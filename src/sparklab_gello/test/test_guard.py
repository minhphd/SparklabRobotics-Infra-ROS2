import unittest

import numpy as np

from sparklab_gello.guard import ARMED, DISARMED, FAULT, ArmGuard, GuardLimits


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class ArmGuardTest(unittest.TestCase):
    """State transitions, hold targets and time-based limits of the arm command latch."""

    def setUp(self) -> None:
        self.clock = Clock()
        self.g = ArmGuard(6, GuardLimits(), clock=self.clock)
        self.home = np.zeros(7)
        self.g.start(self.home)

    def arm(self, target=None) -> None:
        self.g.request_arm(self.home)
        self.g.filter(self.home if target is None else target, self.home)

    def test_starts_disarmed_and_blocks_commands(self) -> None:
        self.assertEqual(self.g.state, DISARMED)
        cmd, execute = self.g.filter(np.full(7, 0.5), self.home)
        self.assertFalse(execute)
        np.testing.assert_array_equal(cmd, self.home)

    def test_arm_needs_a_request_and_a_matching_command(self) -> None:
        self.g.filter(self.home, self.home)
        self.assertEqual(self.g.state, DISARMED)
        self.g.request_arm(self.home)
        far = self.home.copy()
        far[2] = 0.5
        cmd, execute = self.g.filter(far, self.home)
        self.assertFalse(execute)
        self.assertEqual(self.g.state, DISARMED)
        self.assertIn("joint3", self.g.arm_result)
        self.assertFalse(self.g.status()["pending"])
        self.arm()
        self.assertEqual(self.g.state, ARMED)

    def test_rate_limit_is_time_based_and_capped_after_a_delay(self) -> None:
        self.arm()
        target = np.full(7, 1.0)
        self.clock.t += 0.01
        cmd, execute = self.g.filter(target, self.home)
        self.assertTrue(execute)
        np.testing.assert_allclose(cmd[:6], 0.04)
        np.testing.assert_allclose(cmd[6], 0.04)
        self.clock.t += 0.2   # a slow loop, still under the command timeout
        cmd, _ = self.g.filter(target, self.home)
        np.testing.assert_allclose(cmd[:6], 4.0 * GuardLimits().max_dt)

    def test_per_call_caps_still_apply(self) -> None:
        self.arm()
        self.clock.t += 0.05
        cmd, _ = self.g.filter(np.full(7, 1.0), self.home, caps=np.full(6, 0.01))
        np.testing.assert_allclose(cmd[:6], 0.01)

    def test_command_gap_faults_and_late_commands_never_resume(self) -> None:
        self.arm()
        moved = self.home + 0.03
        self.clock.t += 0.3
        cmd, execute = self.g.filter(self.home, moved)
        self.assertFalse(execute)
        self.assertEqual(self.g.state, FAULT)
        np.testing.assert_array_equal(self.g.hold_target, moved)
        for _ in range(5):
            self.clock.t += 0.01
            cmd, execute = self.g.filter(np.full(7, 0.5), moved)
            self.assertFalse(execute)
            np.testing.assert_array_equal(cmd, moved)

    def test_watchdog_faults_at_the_measured_pose_not_the_last_target(self) -> None:
        self.arm()
        self.clock.t += 0.01
        self.g.filter(np.full(7, 1.0), self.home)
        drifted = self.home + 0.02
        self.clock.t += 0.3
        self.assertTrue(self.g.check_timeout(drifted))
        self.assertEqual(self.g.state, FAULT)
        np.testing.assert_array_equal(self.g.hold_target, drifted)

    def test_rearm_after_fault_needs_request_and_match(self) -> None:
        self.arm()
        self.clock.t += 0.3
        self.g.check_timeout(self.home)
        self.g.filter(self.home, self.home)
        self.assertEqual(self.g.state, FAULT)
        self.arm()
        self.assertEqual(self.g.state, ARMED)

    def test_invalid_inputs_fault(self) -> None:
        self.arm()
        for bad in (np.array([np.nan] * 7), np.array([np.inf] * 7), np.zeros(5)):
            with self.subTest(bad=bad):
                self.g.state = ARMED
                self.g.last_cmd_t = self.clock.t
                _, execute = self.g.filter(bad, self.home)
                self.assertFalse(execute)
                self.assertEqual(self.g.state, FAULT)

    def test_disarm_keeps_a_fault_reason(self) -> None:
        self.g.fault("test fault", self.home)
        self.g.disarm("client disconnect", self.home)
        self.assertEqual(self.g.state, FAULT)
        self.assertEqual(self.g.reason, "test fault")

    def test_pending_request_expires(self) -> None:
        self.g.request_arm(self.home)
        self.clock.t += GuardLimits().arm_request_timeout + 1
        self.g.check_timeout(self.home)
        self.assertFalse(self.g.status()["pending"])
        self.assertIn("expired", self.g.arm_result)

    def test_each_hold_transition_is_numbered(self) -> None:
        seq = self.g.hold_seq
        self.arm()
        self.g.disarm("episode end", self.home)
        self.assertEqual(self.g.hold_seq, seq + 1)


if __name__ == "__main__":
    unittest.main()
