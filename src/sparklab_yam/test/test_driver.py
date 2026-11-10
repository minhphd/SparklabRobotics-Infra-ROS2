"""The driver against an i2rt SimRobot, over real topics and services."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage

import sparklab_yam.driver
from sparklab_yam.constants import DEFAULT_MAX_RELATIVE_TARGET
from sparklab_yam.driver import JOINT_NAMES, YamDriver
from sparklab_yam.model.kinematics import DEFAULT_Q_REST


KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0])
KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])
GRAVITY = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0, 1.0])


@pytest.fixture
def gains(monkeypatch):
    """Give the SimRobot i2rt's gain API, recording (kp, kd, gravity factor) per change."""
    calls, real = [], sparklab_yam.driver.get_yam_robot

    def robot(**kwargs):
        r = real(**kwargs)
        calls.append(kwargs)
        r.get_robot_info = lambda: {"kp": KP, "kd": KD}
        r.gravity_comp_factor = GRAVITY.copy()
        r.update_kp_kd = lambda kp, kd: calls.append((kp.copy(), kd.copy(), r.gravity_comp_factor.copy()))
        return r
    monkeypatch.setattr(sparklab_yam.driver, "get_yam_robot", robot)
    return calls


@pytest.fixture
def rig(request):
    rest = "true" if getattr(request, "param", False) else "false"
    rclpy.init(args=["--ros-args", "-p", "sim:=true", "-p", "park_duration_s:=0.5", "-p", f"rest_on_start:={rest}"])
    driver, probe = YamDriver(), Node("probe")
    executor = MultiThreadedExecutor()
    executor.add_node(driver)
    executor.add_node(probe)
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    thread.start()
    states: list[JointState] = []
    probe.create_subscription(JointState, "joint_states", states.append, 10)
    yield probe, states
    stop.set()
    thread.join()
    executor.shutdown()
    driver.close()
    rclpy.try_shutdown()


def _latest(states, after: float = 0.3) -> JointState:
    time.sleep(after)
    return states[-1]


def test_publishes_joint_states_at_rate(rig):
    _, states = rig
    time.sleep(1.0)
    assert list(states[-1].name) == JOINT_NAMES
    assert len(states[-1].effort) == 7
    assert 80 <= len(states) <= 120


def test_tf_carries_ee_pose_as_base_to_tool0(rig):
    probe, _ = rig
    tfs, poses = [], []
    probe.create_subscription(TFMessage, "/tf", lambda m: tfs.extend(m.transforms), 10)
    probe.create_subscription(PoseStamped, "ee_pose", poses.append, 10)
    time.sleep(0.5)
    tf = tfs[-1]
    assert (tf.header.frame_id, tf.child_frame_id) == ("arm_base", "arm_tool0")
    pose = next(p for p in poses if p.header.stamp == tf.header.stamp)
    t, p = tf.transform.translation, pose.pose.position
    assert [t.x, t.y, t.z] == pytest.approx([p.x, p.y, p.z])
    assert tf.transform.rotation == pose.pose.orientation


def test_arm_command_is_clamped_to_max_step_from_present(rig):
    probe, states = rig
    pub = probe.create_publisher(Float64MultiArray, "arm/command", 10)
    time.sleep(0.3)
    pub.publish(Float64MultiArray(data=[1.0] * 6))
    q = _latest(states).position
    assert list(q[:6]) == pytest.approx(DEFAULT_MAX_RELATIVE_TARGET)


def test_gripper_command_holds_the_arm(rig):
    probe, states = rig
    arm = probe.create_publisher(Float64MultiArray, "arm/command", 10)
    gripper = probe.create_publisher(Float64, "gripper/command", 10)
    time.sleep(0.3)
    arm.publish(Float64MultiArray(data=[0.1] * 6))
    time.sleep(0.1)
    gripper.publish(Float64(data=0.6))
    q = _latest(states).position
    assert list(q) == pytest.approx([0.1] * 6 + [0.6])


def test_park_ramps_everything_to_zero(rig):
    probe, states = rig
    pub = probe.create_publisher(Float64MultiArray, "arm/command", 10)
    client = probe.create_client(Trigger, "park")
    time.sleep(0.3)
    pub.publish(Float64MultiArray(data=[0.1] * 6))
    time.sleep(0.1)
    future = client.call_async(Trigger.Request())
    deadline = time.monotonic() + 5.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert future.result().success
    assert list(_latest(states).position) == pytest.approx([0.0] * 7)


@pytest.mark.parametrize("rig", [True], indirect=True)
def test_start_ramps_to_the_rest_pose_and_a_park_waits_for_it(rig):
    probe, states = rig
    time.sleep(0.2)
    assert _call(probe, Trigger, "park", Trigger.Request()).success
    assert list(_latest(states).position) == pytest.approx([0.0] * 7)
    rest = next(s for s in states if np.allclose(s.position[:6], DEFAULT_Q_REST, atol=0.02))
    assert rest.position[6] == pytest.approx(states[0].position[6])


def test_a_ramp_never_leads_a_stalled_arm_by_more_than_max_step():
    sent, warnings = [], []
    stalled = SimpleNamespace(get_joint_pos=lambda: np.zeros(7), command_joint_pos=lambda q: sent.append(np.array(q)))
    driver = SimpleNamespace(
        _max_step=np.asarray(DEFAULT_MAX_RELATIVE_TARGET), _ramp_lock=threading.Lock(), _lock=threading.Lock(),
        _zero_gravity=False, _ramping=False, _robot=stalled, _command=stalled.command_joint_pos, _park_s=0.2,
        _cmd=np.zeros(7),
        get_logger=lambda: SimpleNamespace(warning=warnings.append))
    assert not YamDriver._ramp(driver, np.array([0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0]))
    lead = np.max(np.abs(np.array(sent)[:, :6]), axis=0)
    assert np.all(lead <= np.asarray(DEFAULT_MAX_RELATIVE_TARGET) + 1e-12)
    assert lead[2] == pytest.approx(DEFAULT_MAX_RELATIVE_TARGET[2])
    assert warnings and not driver._ramping


class _LiveArray:
    """Like i2rt's MotorChainRobot, get_joint_pos hands out its live array, replaced on update."""

    def __init__(self):
        self.pos = np.zeros(7)
        self.sent = []

    def get_joint_pos(self):
        return self.pos

    def command_joint_pos(self, q):
        self.sent.append(np.array(q))
        self.pos = np.array(q)


def test_the_start_ramp_begins_where_the_arm_is_when_the_robot_hands_out_its_live_array(monkeypatch):
    rest = np.array([0.0, 0.5, 1.0, -1.0, 0.0, 0.0])
    monkeypatch.setattr(sparklab_yam.driver, "DEFAULT_Q_REST", rest)
    robot = _LiveArray()
    first = robot.pos
    logged = []
    driver = SimpleNamespace(
        _max_step=np.asarray(DEFAULT_MAX_RELATIVE_TARGET), _ramp_lock=threading.Lock(), _lock=threading.Lock(),
        _zero_gravity=False, _ramping=False, _robot=robot, _command=robot.command_joint_pos, _park_s=0.2,
        _cmd=np.zeros(7), _alive=lambda: True,
        get_logger=lambda: SimpleNamespace(info=logged.append, warning=logged.append))
    driver._ramp = lambda target: YamDriver._ramp(driver, target)
    YamDriver._ramp_to_rest(driver)
    # SimRobot copies, so only this stand-in shows the rig's failure: the start taken as the target.
    assert np.all(first == 0.0)
    assert np.max(np.abs(robot.sent[0][:6])) < 0.01
    np.testing.assert_allclose(robot.sent[-1][:6], rest)


def _call(probe, srv_type, name, request):
    future = probe.create_client(srv_type, name).call_async(request)
    deadline = time.monotonic() + 5.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    return future.result()


def test_zero_gravity_needs_the_i2rt_mode(rig):
    probe, _ = rig
    time.sleep(0.3)
    assert not _call(probe, SetBool, "zero_gravity", SetBool.Request(data=True)).success


def test_zero_gravity_floats_the_arm_then_holds(gains, rig):
    probe, states = rig
    pub = probe.create_publisher(Float64MultiArray, "arm/command", 10)
    time.sleep(0.3)
    assert gains[0]["ee_mass"] is None  # 0.0 by default: i2rt's stock gripper

    assert _call(probe, SetBool, "zero_gravity", SetBool.Request(data=True)).success
    kp, kd, gravity = gains[-1]
    assert kp.tolist() == [0.0] * 6 + [20.0]  # arm limp, gripper holding
    assert kd.tolist() == KD.tolist()  # damping kept
    assert gravity.tolist() == [1.0] * 7
    pub.publish(Float64MultiArray(data=[0.1] * 6))
    assert list(_latest(states).position[:6]) == pytest.approx([0.0] * 6)

    assert _call(probe, SetBool, "zero_gravity", SetBool.Request(data=False)).success
    kp, kd, gravity = gains[-1]
    assert kp.tolist() == KP.tolist() and kd.tolist() == KD.tolist()
    pub.publish(Float64MultiArray(data=[0.1] * 6))
    assert list(_latest(states).position[:6]) == pytest.approx([0.1] * 6)


def test_park_while_floating_restores_gains_first(gains, rig):
    probe, states = rig
    time.sleep(0.3)
    assert _call(probe, SetBool, "zero_gravity", SetBool.Request(data=True)).success
    assert _call(probe, Trigger, "park", Trigger.Request()).success
    assert gains[-1][0].tolist() == KP.tolist()
    assert list(_latest(states).position) == pytest.approx([0.0] * 7)
