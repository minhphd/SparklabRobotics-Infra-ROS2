"""Run ABC's released policy on the rig: a websocket client of ABC's ``serve_policy.py``.

    ros2 run abc_bridge bridge --ros-args -p prompt:="throw plastic bottles in bin"
    ros2 run abc_bridge bridge --ros-args -p prompt:="..." -p dry_run:=false   # real arms
    ros2 run abc_bridge bridge --ros-args -p prompt:="..." -p record_dir:=$HOME/recordings

Dry run (the default) does everything but publishes to /abc/preview/<arm>/... instead
of the arm and gripper command topics, and never parks. The episode ends on Ctrl-C.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import msgpack
import numpy as np
import rclpy
import websockets.sync.client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import (Float32MultiArray, Float64, Float64MultiArray, MultiArrayDimension,
                          MultiArrayLayout, String)
from std_srvs.srv import Trigger

from sparklab_yam.constants import ARM_JOINTS, HANDS

# ABC's deploy command: --execute-chunk-dim=16 --rtc-prefix-length=4 --rtc-inference-lead-steps=7
CHUNK, PREFIX, LEAD = 16, 4, 7
DT = 0.03333
# (chunk_len, action_dim) of ABC's released checkpoints; the server's noise input.
NOISE_SHAPE = (30, 14)
# _DEFAULT_INIT_Q in ABC's deploy/robot/config.py, per arm, gripper last.
START_POSE = np.array([0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 1.0])
RAMP_S = 5.0
STALE_S = 0.5
# ABC camera name -> our camera id. ABC trained on 640x480 (4:3) everywhere.
CAMERAS = {"top": "top", "left": "left_wrist", "right": "right_wrist"}
IMAGE_HW = (480, 640)


def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str,
                b"shape": obj.shape}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


def _to_abc_image(msg: Image) -> np.ndarray:
    """Centre-crop to 4:3, resize to 640x480. Returns (3, H, W) uint8 RGB."""
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    h, w = img.shape[:2]
    if w * 3 > h * 4:
        x0 = (w - h * 4 // 3) // 2
        img = img[:, x0:x0 + h * 4 // 3]
    if img.shape[:2] != IMAGE_HW:
        img = cv2.resize(img, IMAGE_HW[::-1], interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def _array(a: np.ndarray) -> Float32MultiArray:
    a = np.asarray(a, dtype=np.float32)
    dims = [MultiArrayDimension(label=f"dim{i}", size=n, stride=int(np.prod(a.shape[i:])))
            for i, n in enumerate(a.shape)]
    return Float32MultiArray(layout=MultiArrayLayout(dim=dims), data=a.ravel().tolist())


class Bridge(Node):
    """ROS side: joint states and images in, 14-D actions out.

    host, port: ABC's policy server.
    prompt: task string; must match one the checkpoint trained on.
    dry_run: publish under /abc/preview instead of the command topics.
    record_dir: if set, the episode is recorded to <record_dir>/<time>_<prompt>/ as MCAP.
    noise_seed: >= 0 sends fresh N(0, 1) flow-matching noise with each request, from this seed;
        -1 sends none, so the server uses zeros.
    start_pose: per-arm ramp target, 6 joints then gripper; defaults to ABC's START_POSE.
    """

    def __init__(self):
        super().__init__("abc_bridge")
        self.host = self.declare_parameter("host", "localhost").value
        self.port = self.declare_parameter("port", 8000).value
        self.prompt = self.declare_parameter("prompt", "").value
        self.dry_run = self.declare_parameter("dry_run", True).value
        self.record_dir = self.declare_parameter("record_dir", "").value
        noise_seed = self.declare_parameter("noise_seed", -1).value
        self._rng = np.random.default_rng(noise_seed) if noise_seed >= 0 else None
        self.start_pose = np.array(self.declare_parameter("start_pose", START_POSE.tolist()).value)
        if self.start_pose.shape != START_POSE.shape:
            raise ValueError(f"start_pose needs {START_POSE.size} values per arm, got {self.start_pose.size}")
        if not self.prompt:
            raise ValueError("set -p prompt:=... to a task name the checkpoint trained on")
        self._state: dict[str, JointState] = {}
        self._images: dict[str, Image] = {}
        ns = self.ns = "/abc/preview" if self.dry_run else ""
        self._arm_pubs, self._gripper_pubs, self._park = {}, {}, {}
        for h in HANDS:
            self.create_subscription(JointState, f"/{h}/joint_states",
                                     lambda m, h=h: self._state.__setitem__(h, m), 10)
            self._arm_pubs[h] = self.create_publisher(Float64MultiArray, f"{ns}/{h}/arm/command", 10)
            self._gripper_pubs[h] = self.create_publisher(Float64, f"{ns}/{h}/gripper/command", 10)
            self._park[h] = self.create_client(Trigger, f"/{h}/park")
        for c in CAMERAS.values():
            self.create_subscription(Image, f"/cameras/{c}/color/image_raw",
                                     lambda m, c=c: self._images.__setitem__(c, m), 1)
        self._obs_image_pubs = {k: self.create_publisher(Image, f"/abc/obs/images/{k}", 1)
                                for k in CAMERAS}
        self._obs_pubs = {k: self.create_publisher(Float32MultiArray, f"/abc/obs/{k}", 10)
                          for k in ("state", "action_prefix", *(("noise",) if self._rng else ()))}
        self._prompt_pub = self.create_publisher(String, "/abc/obs/prompt", 10)
        self._actions_pub = self.create_publisher(Float32MultiArray, "/abc/actions", 10)

    def record_request(self, obs: dict) -> None:
        """Publish a request exactly as sent: /abc/obs/{state,action_prefix,noise,prompt,images/<cam>}.

        Images keep the camera node's publish stamp, so publish time minus stamp is their age.
        """
        for k, img in obs["images"].items():
            _, h, w = img.shape
            msg = Image(height=h, width=w, encoding="rgb8", step=w * 3,
                        data=img.transpose(1, 2, 0).tobytes())
            msg.header.stamp, msg.header.frame_id = self._obs_stamps[k], CAMERAS[k]
            self._obs_image_pubs[k].publish(msg)
        for k, pub in self._obs_pubs.items():
            pub.publish(_array(obs[k]))
        self._prompt_pub.publish(String(data=obs["prompt"]))

    def record_reply(self, actions: np.ndarray) -> None:
        """Publish the server's full (30, 14) reply on /abc/actions."""
        self._actions_pub.publish(_array(actions))

    def start_recorder(self) -> subprocess.Popen:
        """Start `ros2 bag record` on the request, reply, joint and command topics.

        Blocks until it subscribes to every /abc topic, so the first request and reply are in the bag.
        """
        name = time.strftime("%Y%m%d_%H%M%S_") + re.sub(r"\W+", "_", self.prompt).strip("_")
        out = os.path.join(os.path.expanduser(self.record_dir), name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        topics = [*(f"/abc/obs/images/{k}" for k in CAMERAS), "/abc/actions",
                  *(f"/abc/obs/{k}" for k in (*self._obs_pubs, "prompt"))]
        for h in HANDS:
            topics += [f"/{h}/joint_states", f"{self.ns}/{h}/arm/command", f"{self.ns}/{h}/gripper/command"]
        # Own session: the terminal's Ctrl-C must not stop it before the park is recorded.
        # stdin closed: ros2 bag's keyboard controls would otherwise take the Enter prompt's input.
        proc = subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", out, *topics],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 15.0
        while (any(self.count_subscribers(t) == 0 for t in topics if t.startswith("/abc/"))
               and time.monotonic() < deadline):
            time.sleep(0.1)
        self.get_logger().info(f"recording to {out}")
        return proc

    def _age(self, msg) -> float:
        return (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9

    def stale(self, images: bool = True) -> list[str]:
        """Topics with no message, or none within STALE_S."""
        out = [f"/{h}/joint_states" for h in HANDS
               if h not in self._state or self._age(self._state[h]) > STALE_S]
        if images:
            out += [f"/cameras/{c}/color/image_raw" for c in CAMERAS.values()
                    if c not in self._images or self._age(self._images[c]) > STALE_S]
        return out

    def state(self) -> np.ndarray:
        """[left j1..j6, left gripper, right j1..j6, right gripper], float32 (14,)."""
        if missing := self.stale(images=False):
            raise RuntimeError(f"stale: {', '.join(missing)}")
        return np.concatenate([self._state[h].position[:ARM_JOINTS + 1] for h in HANDS]).astype(np.float32)

    def observation(self) -> dict:
        if missing := self.stale():
            raise RuntimeError(f"stale: {', '.join(missing)}")
        msgs = {k: self._images[c] for k, c in CAMERAS.items()}
        self._obs_stamps = {k: m.header.stamp for k, m in msgs.items()}
        obs = {"state": self.state(), "prompt": self.prompt,
               "images": {k: _to_abc_image(m) for k, m in msgs.items()}}
        if self._rng is not None:
            obs["noise"] = self._rng.standard_normal(NOISE_SHAPE, dtype=np.float32)
        return obs

    def publish(self, action: np.ndarray) -> None:
        for i, h in enumerate(HANDS):
            a = action[i * (ARM_JOINTS + 1):(i + 1) * (ARM_JOINTS + 1)]
            self._arm_pubs[h].publish(Float64MultiArray(data=a[:ARM_JOINTS].tolist()))
            self._gripper_pubs[h].publish(Float64(data=float(a[ARM_JOINTS])))

    def park(self) -> None:
        futures = {h: self._park[h].call_async(Trigger.Request()) for h in HANDS}
        deadline = time.monotonic() + 30.0
        while not all(f.done() for f in futures.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        for h, f in futures.items():
            if not f.done() or not f.result().success:
                self.get_logger().warning(f"{h} arm did not park")


class Policy:
    """ABC's websocket protocol: msgpack-numpy request in, {"actions": (30, 14)} out."""

    def __init__(self, host: str, port: int):
        self._ws = websockets.sync.client.connect(f"ws://{host}:{port}", compression=None,
                                                  max_size=None, ping_interval=300, ping_timeout=600)
        self.metadata = msgpack.unpackb(self._ws.recv(), object_hook=_unpack)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def infer(self, obs: dict) -> np.ndarray:
        self._ws.send(msgpack.packb(obs, default=_pack))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error:\n{reply}")
        return np.asarray(msgpack.unpackb(reply, object_hook=_unpack)["actions"])

    def infer_async(self, obs: dict) -> Future:
        """Returns: a future of (actions, seconds the round trip took)."""
        def timed():
            t = time.monotonic()
            return self.infer(obs), time.monotonic() - t
        return self._executor.submit(timed)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._ws.close()


def first_action_check(state: np.ndarray, actions: np.ndarray) -> None:
    """Print each joint's first-action delta, as ABC's policy_safety.py does, and wait for Enter."""
    labels = [f"{h[0].upper()}_{j}" for h in HANDS for j in [*range(1, ARM_JOINTS + 1), "grip"]]
    delta = actions[0] - state
    print(f"\n{'joint':<8}{'current':>10}{'action[0]':>11}{'delta':>9}")
    for name, s, a, d in zip(labels, state, actions[0], delta):
        print(f"{name:<8}{s:>+10.4f}{a:>+11.4f}{d:>+9.4f}")
    worst = np.abs(delta).max()
    print(f"max |delta| {worst:.4f} rad ({np.degrees(worst):.1f} deg)")
    if worst > 0.5:
        print(f"*** WARNING: large first-step delta ({worst:.3f} rad) ***")
    input("Press ENTER to execute, or Ctrl-C to abort > ")


def ramp_to_start(bridge: Bridge) -> None:
    q0 = bridge.state()
    q1 = np.tile(bridge.start_pose, 2)
    n = int(RAMP_S / DT)
    t = time.monotonic()
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        bridge.publish(q0 + s * (q1 - q0))
        t += DT
        time.sleep(max(0.0, t - time.monotonic()))


def run_episode(bridge: Bridge, policy: Policy) -> None:
    """ABC's RTC loop (rtc_gym_env.py, policy_rollout.py): execute 16 of each 30-step chunk
    at 30 Hz, sending the next request LEAD steps before the chunk ends. Runs until Ctrl-C.
    """
    log = bridge.get_logger()
    obs = bridge.observation()
    obs.update(action_prefix=np.tile(obs["state"], (PREFIX, 1)), prefix_length=PREFIX)
    bridge.record_request(obs)
    reply = policy.infer(obs)
    bridge.record_reply(reply)
    actions = reply[PREFIX:PREFIX + CHUNK]
    first_action_check(obs["state"], actions)

    n_chunks = n_late = 0
    t = time.monotonic()
    while True:
        for step, command in enumerate(actions):
            if step == CHUNK - LEAD:
                obs = bridge.observation()
                obs.update(action_prefix=actions[-PREFIX:], prefix_length=PREFIX)
                bridge.record_request(obs)
                pending = policy.infer_async(obs)
            bridge.state()  # raises on a stale joint state
            bridge.publish(command)
            t += DT
            time.sleep(max(0.0, t - time.monotonic()))
        n_chunks += 1
        late = not pending.done()
        if late:
            n_late += 1
            log.warning(f"chunk {n_chunks}: inference still pending; holding the last command")
        reply, seconds = pending.result(timeout=5.0)
        bridge.record_reply(reply)
        nxt = reply[PREFIX:PREFIX + CHUNK]
        if late:
            t = time.monotonic()
        log.info(f"chunk {n_chunks}: infer {seconds * 1e3:.0f} ms, "
                 f"boundary |dq| {np.abs(nxt[0] - actions[-1]).max():.3f} rad, late {n_late}")
        actions = nxt


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # A background job inherits SIGINT ignored; restore it with SIGTERM.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    bridge = Bridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.1)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    log = bridge.get_logger()
    policy = recorder = None
    moved = False
    try:
        log.info(f"{'DRY RUN, publishing to /abc/preview' if bridge.dry_run else 'LIVE'}; "
                 f"waiting for ws://{bridge.host}:{bridge.port}")
        policy = Policy(bridge.host, bridge.port)
        deadline = time.monotonic() + 10.0
        while (missing := bridge.stale()) and time.monotonic() < deadline:
            time.sleep(0.05)
        if missing:
            raise RuntimeError(f"nothing fresh on {', '.join(missing)}; start the rig first")
        log.info(f"ramping to ABC's start pose over {RAMP_S:.0f} s")
        moved = True
        ramp_to_start(bridge)
        if bridge.record_dir:
            recorder = bridge.start_recorder()
        run_episode(bridge, policy)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if moved and not bridge.dry_run:
            log.info("parking both arms")
            bridge.park()
        if recorder is not None:
            recorder.send_signal(signal.SIGINT)
            try:
                recorder.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                recorder.kill()
            log.info("recording closed")
        if policy is not None:
            policy.close()
        stop.set()
        spinner.join()
        executor.shutdown()
        bridge.destroy_node()
        rclpy.try_shutdown()
