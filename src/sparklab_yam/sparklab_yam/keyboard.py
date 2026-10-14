"""Drive the arms from the keyboard, one joint at a time — turtlesim's teleop_key for the YAM.

    ros2 run sparklab_yam keyboard

Each press moves one joint of the selected arm by ``step`` rad and publishes
the new target on /<arm>/arm/command; the driver clamps it as usual. Don't run
it alongside the Quest teleop or a policy.
"""

from __future__ import annotations

import signal
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from .constants import ARM_JOINTS, HANDS
from .model.kinematics import build_model_with_tool0_site

HELP = """
Keyboard control for the YAM arms
---------------------------------
   joint   1  2  3  4  5  6
     +     q  w  e  r  t  y
     -     a  s  d  f  g  h

   z / x   open / close the gripper
   space   switch arm (left <-> right)
   Ctrl-C  quit (the arm holds where it is)
"""
JOG = {k: (j, +1) for j, k in enumerate("qwerty")} | {k: (j, -1) for j, k in enumerate("asdfgh")}
# After a pause, a target this far from the measured pose means something else moved the arm.
# Within a burst the target is kept, so a held key doesn't accumulate the arm's sag.
RESYNC_RAD, BURST_S = 0.03, 0.5


class Keyboard(Node):
    def __init__(self) -> None:
        super().__init__("yam_keyboard")
        self.step = self.declare_parameter("step", 0.02).value
        self.arm = self.declare_parameter("arm", "left").value
        self.limits = build_model_with_tool0_site()[0].jnt_range[:ARM_JOINTS].copy()
        self.measured: dict[str, np.ndarray] = {}
        self.target: dict[str, np.ndarray] = {}
        self._last_press = 0.0
        self._arm_pubs = {h: self.create_publisher(Float64MultiArray, f"/{h}/arm/command", 10) for h in HANDS}
        self._gripper_pubs = {h: self.create_publisher(Float64, f"/{h}/gripper/command", 10) for h in HANDS}
        for h in HANDS:
            self.create_subscription(JointState, f"/{h}/joint_states",
                                     lambda m, h=h: self.measured.__setitem__(h, np.array(m.position)), 10)

    def press(self, key: str) -> str:
        """Act on one key press.

        Returns: a line describing what happened.
        """
        if key == " ":
            self.arm = HANDS[1 - HANDS.index(self.arm)]
            return f"now driving the {self.arm} arm"
        if key not in JOG and key not in ("z", "x"):
            return "unknown key — see the table above"
        measured = self.measured.get(self.arm)
        if measured is None:
            return f"no /{self.arm}/joint_states — is the driver running?"
        if key in ("z", "x"):
            self._gripper_pubs[self.arm].publish(Float64(data=0.0 if key == "z" else 1.0))
            return f"{self.arm} gripper {'open' if key == 'z' else 'close'}"
        target = self.target.get(self.arm)
        paused = time.monotonic() - self._last_press > BURST_S
        if target is None or paused and np.max(np.abs(target - measured[:ARM_JOINTS])) > RESYNC_RAD:
            target = measured[:ARM_JOINTS].copy()
        self._last_press = time.monotonic()
        j, sign = JOG[key]
        target[j] = np.clip(target[j] + sign * self.step, *self.limits[j])
        self.target[self.arm] = target
        self._arm_pubs[self.arm].publish(Float64MultiArray(data=target.tolist()))
        return f"{self.arm} joint {j + 1}: {target[j]:+.2f} rad"


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = Keyboard()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.1) for _ in iter(stop.is_set, True)])
    spinner.start()
    settings = termios.tcgetattr(sys.stdin)
    print(HELP)
    print(f"driving the {node.arm} arm, {node.step} rad per press")
    try:
        tty.setcbreak(sys.stdin)  # one key at a time, no Enter; Ctrl-C still works
        while key := sys.stdin.read(1):
            if key == "\x1b":  # arrow and function keys: drop the rest of the sequence
                sys.stdin.read(2)
                continue
            print(node.press(key.lower()))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        stop.set()
        spinner.join()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
