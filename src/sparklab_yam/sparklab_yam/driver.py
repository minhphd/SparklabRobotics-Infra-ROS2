"""ROS 2 driver for one YAM-Ultra arm + LINEAR_4310 gripper. Owns CAN and torque.

    ros2 run sparklab_yam driver --ros-args -r __ns:=/left -p channel:=can_left
    ros2 run sparklab_yam driver --ros-args -r __ns:=/left -p sim:=true -p viewer:=true

Topics are relative to the namespace (one arm per namespace):

    joint_states     out  sensor_msgs/JointState  joint_1..6 + gripper: pos, vel, effort
    ee_pose          out  geometry_msgs/PoseStamped  tool0 in <ns>_base, FK of joint_states
    /tf              out  the same pose as <ns>_base -> <ns>_tool0
    arm/command      in   std_msgs/Float64MultiArray  6 absolute joint targets (rad)
    gripper/command  in   std_msgs/Float64  gripper target, i2rt's normalised 0 closed .. 1 open
    park             srv  std_srvs/Trigger  ramp everything to zero; blocks for the ramp
    zero_gravity     srv  std_srvs/SetBool  true: arm joints float under gravity compensation,
                                            damped, gripper holding, commands ignored;
                                            false: hold where it is

Every arm command is clamped to ``max_step`` from the measured pose. A dead
control loop stops joint_states, which is how clients detect it. With
``rest_on_start`` the arm ramps to the rest pose (DEFAULT_Q_REST) on start,
ignoring commands until it is there. On SIGINT or SIGTERM the arm parks before
torque is cut, unless ``park_on_exit`` is false.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster

from .constants import ARM_JOINTS, DEFAULT_MAX_RELATIVE_TARGET
from .ik.solver import IKSolver
from .model.kinematics import DEFAULT_Q_REST
from .run import run
from .sim_coupling import apply_joint_couplings

JOINT_NAMES = [f"joint_{j}" for j in range(1, ARM_JOINTS + 1)] + ["gripper"]
RAMP_DT = 0.01
RAMP_TOL = 0.03
"""rad from its target at which a park or startup ramp counts the arm as there."""


class YamDriver(Node):
    def __init__(self) -> None:
        super().__init__("yam_driver")
        channel = self.declare_parameter("channel", "can_left").value
        sim = self._sim = self.declare_parameter("sim", False).value
        rate_hz = self.declare_parameter("rate_hz", 100.0).value
        self._max_step = np.asarray(self.declare_parameter(
            "max_step", list(DEFAULT_MAX_RELATIVE_TARGET)).value, dtype=float)
        self._park_s = self.declare_parameter("park_duration_s", 5.0).value
        self._park_on_exit = self.declare_parameter("park_on_exit", True).value
        rest_on_start = self.declare_parameter("rest_on_start", False).value
        # Recovery re-asserts the pre-error target at full gains from inside
        # i2rt, which max_step cannot see or bound: it trades a stop for a snap.
        auto_recovery = self.declare_parameter("enable_auto_recovery", False).value
        # The gripper's mass in i2rt's gravity model, kg; 0 keeps the stock LINEAR_4310's 0.553.
        ee_mass = self.declare_parameter("ee_mass", 0.0).value
        # Scale on the modelled gravity torque per arm joint, while holding position
        # (i2rt's YAM-Ultra default) and while floating in zero gravity.
        gravity = self.declare_parameter("gravity_comp_factor", [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]).value
        self._float_gravity = np.asarray(self.declare_parameter(
            "zero_gravity_comp_factor", [1.0] * ARM_JOINTS).value, dtype=float)
        viewer = self.declare_parameter("viewer", False).value

        self.get_logger().info(
            f"connecting {channel}{' (SIM)' if sim else ''} — the gripper self-homes, keep clear "
            f"(ee_mass {ee_mass or 'stock'}, gravity_comp_factor {list(gravity)}, "
            f"zero_gravity_comp_factor {self._float_gravity.tolist()})")
        # Upstream i2rt has no enable_auto_recovery; pass it only when asked for.
        self._robot = get_yam_robot(
            channel=channel, arm_type=ArmType.YAM_ULTRA,
            gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=False, sim=sim,
            gravity_comp_factor=np.asarray(gravity, dtype=float), ee_mass=ee_mass or None,
            **({"enable_auto_recovery": True} if auto_recovery else {}))
        self._lock = threading.Lock()
        # Held for a whole ramp, so a park asked for mid-ramp waits instead of fighting it.
        self._ramp_lock = threading.Lock()
        self._ramping = False
        self._zero_gravity = False
        self._held_gains = None  # (kp, kd, gravity factor) to restore after zero gravity
        self._dead_logged = False
        self._last_clamp_warn = 0.0
        self._cmd = np.asarray(self._robot.get_joint_pos(), dtype=float)
        self.get_logger().info(
            f"pose at startup: {np.round(self._cmd[:ARM_JOINTS], 3).tolist()} (park target is zero)")

        self._fk = IKSolver()
        arm = self.get_namespace().strip("/") or "arm"
        self._frame_id, self._tool_frame = f"{arm}_base", f"{arm}_tool0"
        self._tf = TransformBroadcaster(self)
        group = ReentrantCallbackGroup()
        self._state_pub = self.create_publisher(JointState, "joint_states", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "ee_pose", 10)
        self.create_subscription(Float64MultiArray, "arm/command", self._on_arm, 10,
                                 callback_group=group)
        self.create_subscription(Float64, "gripper/command", self._on_gripper, 10,
                                 callback_group=group)
        self.create_service(Trigger, "park", self._on_park, callback_group=group)
        self.create_service(SetBool, "zero_gravity", self._on_zero_gravity, callback_group=group)
        self.create_timer(1.0 / rate_hz, self._publish_state, callback_group=group)

        self._viewer = None
        if viewer:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self._robot._model, self._robot._data)
            self.create_timer(1.0 / 60.0, self._sync_viewer, callback_group=group)
        if rest_on_start:
            threading.Thread(target=self._ramp_to_rest, daemon=True).start()

    def _alive(self) -> bool:
        """False once i2rt's control loop has died; SimRobots have no chain."""
        chain = getattr(self._robot, "motor_chain", None)
        return chain is None or getattr(chain, "running", True)

    def _command(self, q: np.ndarray) -> None:
        """Command the robot; in sim, also move joints the SimRobot leaves to its equalities."""
        self._robot.command_joint_pos(q)
        if self._sim:
            # Same lock the viewer syncs under, so it never draws a half-updated pose.
            with self._robot._lock:
                apply_joint_couplings(self._robot._model, self._robot._data,
                                      min(len(q), self._robot._model.nq))

    # ---- state ------------------------------------------------------------
    def _publish_state(self) -> None:
        # A dead chain keeps returning the last pose, so publishing it would
        # make the arm look healthy while every command goes nowhere.
        if not self._alive():
            if not self._dead_logged:
                self._dead_logged = True
                self.get_logger().error(
                    "i2rt control loop is dead — the motors stopped answering on CAN. "
                    "Check for 'loss communication' above and the E-stop, then restart.")
            return
        obs = self._robot.get_observations()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = np.concatenate([obs["joint_pos"], obs["gripper_pos"]]).tolist()
        msg.velocity = np.concatenate([obs["joint_vel"], obs["gripper_vel"]]).tolist()
        msg.effort = np.concatenate([obs["joint_eff"], obs["gripper_eff"]]).tolist()
        self._state_pub.publish(msg)

        pos, quat_wxyz = self._fk.fk(obs["joint_pos"])
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self._frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos.tolist()
        (pose.pose.orientation.w, pose.pose.orientation.x,
         pose.pose.orientation.y, pose.pose.orientation.z) = quat_wxyz.tolist()
        self._pose_pub.publish(pose)

        tf = TransformStamped(header=pose.header, child_frame_id=self._tool_frame)
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos.tolist()
        tf.transform.rotation = pose.pose.orientation
        self._tf.sendTransform(tf)

    def _sync_viewer(self) -> None:
        with self._robot._lock:
            self._viewer.sync()

    # ---- commands ---------------------------------------------------------
    def _on_arm(self, msg: Float64MultiArray) -> None:
        target = np.asarray(msg.data, dtype=float)
        if target.shape != (ARM_JOINTS,):
            self.get_logger().error(f"arm/command needs {ARM_JOINTS} values, got {len(target)}")
            return
        with self._lock:
            if self._ramping or self._zero_gravity or not self._alive():
                return
            present = np.asarray(self._robot.get_joint_pos(), dtype=float)[:ARM_JOINTS]
            step = target - present
            overshoot = np.abs(step) - self._max_step
            self._cmd[:ARM_JOINTS] = present + np.clip(step, -self._max_step, self._max_step)
            self._command(self._cmd.copy())
        if np.any(overshoot > 0) and time.monotonic() - self._last_clamp_warn > 1.0:
            self._last_clamp_warn = time.monotonic()
            j = int(np.argmax(overshoot))
            self.get_logger().warning(
                f"joint_{j + 1} command exceeded max_step {self._max_step[j]:.3f} rad by "
                f"{overshoot[j]:.3f} — clamped (suppressed for 1 s)")

    def _on_gripper(self, msg: Float64) -> None:
        with self._lock:
            if self._ramping or self._zero_gravity or not self._alive():
                return
            self._cmd[ARM_JOINTS] = float(np.clip(msg.data, 0.0, 1.0))
            self._command(self._cmd.copy())

    # ---- parking ----------------------------------------------------------
    def _ramp(self, target: np.ndarray) -> bool:
        """Ease every joint, gripper included, to target over park_duration_s, ignoring commands.

        Each command leads the measured pose by at most max_step, as client commands do, so
        an arm that stalls is never commanded far ahead of itself. See DESIGN.md#the-arm-drivers.
        Returns: whether the arm joints got within RAMP_TOL of target by twice park_duration_s.
        """
        lead = np.append(self._max_step, np.inf)
        with self._ramp_lock:
            with self._lock:
                if self._zero_gravity:
                    self._set_floating(False)
            self._ramping = True
            try:
                start = np.asarray(self._robot.get_joint_pos(), dtype=float)
                t0 = time.monotonic()
                while True:
                    a = min(1.0, (time.monotonic() - t0) / self._park_s)
                    a = a * a * (3.0 - 2.0 * a)
                    present = np.asarray(self._robot.get_joint_pos(), dtype=float)
                    command = present + np.clip((1.0 - a) * start + a * target - present, -lead, lead)
                    self._command(command)
                    arrived = a == 1.0 and np.max(np.abs(present - target)[:ARM_JOINTS]) < RAMP_TOL
                    if arrived or time.monotonic() - t0 > 2 * self._park_s:
                        break
                    time.sleep(RAMP_DT)
                with self._lock:
                    self._cmd = command
            finally:
                self._ramping = False
        if not arrived:
            self.get_logger().warning(
                f"ramp stopped short of {np.round(target[:ARM_JOINTS], 3).tolist()} at "
                f"{np.round(present[:ARM_JOINTS], 3).tolist()} — blocked, or too weak to get there")
        return arrived

    def _park(self) -> bool:
        """Ramp every joint, gripper included, to zero — the folded park pose."""
        if not self._alive():
            self.get_logger().warning(
                "control loop is dead — cannot park; the arm drops when torque is cut")
            return False
        self.get_logger().info(f"parking: ramping to zero over {self._park_s:.1f}s")
        self._ramp(np.zeros_like(self._cmd))
        return True

    def _ramp_to_rest(self) -> None:
        """Ramp the arm to the rest pose, the gripper staying where it homed."""
        if not self._alive():
            return
        target = np.asarray(self._robot.get_joint_pos(), dtype=float)
        target[:ARM_JOINTS] = DEFAULT_Q_REST
        self.get_logger().info(f"ramping to the rest pose {DEFAULT_Q_REST.tolist()} over "
                               f"{self._park_s:.1f}s; commands are ignored until it is there")
        if self._ramp(target):
            self.get_logger().info("at the rest pose")

    def _set_floating(self, on: bool) -> None:
        """Arm joints to zero stiffness under the floating gravity factor, or back.

        Damping and the gripper's gains stay, so a released arm can't run away and
        the gripper keeps its grip. Call with the lock held.
        """
        if on and not self._zero_gravity:
            info = self._robot.get_robot_info()
            self._held_gains = (info["kp"].copy(), info["kd"].copy(),
                                np.array(self._robot.gravity_comp_factor, dtype=float))
            kp = self._held_gains[0].copy()
            kp[:ARM_JOINTS] = 0.0
            self._robot.gravity_comp_factor = np.append(self._float_gravity,
                                                        self._held_gains[2][ARM_JOINTS:])
            self._robot.update_kp_kd(kp, self._held_gains[1])
        elif not on and self._zero_gravity:
            kp, kd, self._robot.gravity_comp_factor = self._held_gains
            self._robot.update_kp_kd(kp, kd)
        # update_kp_kd only stores gains; commanding the present pose applies them.
        self._cmd = np.asarray(self._robot.get_joint_pos(), dtype=float)
        self._command(self._cmd.copy())
        self._zero_gravity = on

    def _on_zero_gravity(self, request: SetBool.Request,
                         response: SetBool.Response) -> SetBool.Response:
        # SimRobots have no gains to drop.
        if request.data and not hasattr(self._robot, "update_kp_kd"):
            response.message = "zero gravity refused: this robot has no zero-gravity mode"
            self.get_logger().warning(response.message)
            return response
        with self._lock:
            self._set_floating(request.data)
        response.success = True
        response.message = ("zero gravity ON: move the arm by hand" if request.data
                            else "zero gravity OFF: holding position")
        self.get_logger().info(response.message)
        return response

    def _on_park(self, _request, response: Trigger.Response) -> Trigger.Response:
        response.success = self._park()
        response.message = "parked" if response.success else "control loop dead"
        return response

    def close(self) -> None:
        # i2rt's close() cuts torque wherever the arm stands.
        if self._park_on_exit:
            self._park()
        if self._viewer is not None:
            self._viewer.close()
        self._robot.close()


def main() -> None:
    run(YamDriver)


if __name__ == "__main__":
    main()
