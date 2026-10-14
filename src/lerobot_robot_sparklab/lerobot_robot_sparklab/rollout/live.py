"""``lerobot-rollout`` with a live ROS 2 control channel.

Same CLI as ``lerobot-rollout``, every flag forwarded untouched. While the loop
runs you can retarget the policy, halt it and send the arms home without paying
the ~90 s model load again::

    ros2 run lerobot_robot_sparklab rollout \\
        --strategy.type=base --policy.path=... --robot.type=yam_ultra_sim \\
        --task="put the marker into the cardboard box" --fps 20

    ros2 topic pub --once /rollout/task std_msgs/msg/String "{data: 'pick up the scissors'}"
    ros2 service call /rollout/reset std_srvs/srv/Trigger

Channel: subscribes /rollout/task; Trigger services /rollout/{reset,park,home,
pause,resume,quit}; publishes /rollout/status (JSON, 2 Hz), and every
--publish-every ticks /rollout/state and /rollout/images/<cam>/compressed —
what the policy was shown. ROS callbacks never touch the robot or the policy:
they queue commands the control loop drains between ticks, since neither the
robot transport nor the inference engine is thread-safe. See DESIGN.md for the
retarget, ramp and hold semantics.
"""

from __future__ import annotations

import array
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .. import ros

logger = logging.getLogger(__name__)

# Reset target. Zeros is the YAM's straight-up rest pose and matches what the
# smoke tests use; override with --home if your bench differs.
DEFAULT_HOME = 0.0


# ---- control state --------------------------------------------------------
@dataclass
class Control:
    """Shared state between the ROS callbacks and the control loop.

    Commands are queued rather than applied in place, so every mutation happens
    between reading an observation and sending an action — applying a task
    change mid-``send_next_action`` would mix two prompts inside one chunk.
    """

    task: str = ""
    paused: bool = False
    # Parked and waiting. Distinct from `paused` ("stopped mid-task, the task
    # still stands"): without it the next tick climbs back out of home.
    held: bool = False
    home_pos: float = DEFAULT_HOME
    reset_tolerance: float = 0.05      # rad, per joint
    # 0 falls back to sending the final target and letting the follower's Δq
    # clamp rate-limit it — faster, but abrupt enough to read as a jolt.
    home_duration_s: float = 5.0
    # Budget for the arm to *settle* onto the target after the ramp, not for
    # the ramp itself.
    reset_timeout_s: float = 8.0

    _pending: queue.Queue = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # counters, for /rollout/status
    ticks: int = 0
    resets: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    last_event: str = "started"

    @property
    def stepping(self) -> bool:
        """Whether the loop should be driving the policy this tick."""
        return not (self.paused or self.held)

    def submit(self, cmd: str, arg: str | None = None) -> dict:
        """Queue a command from a ROS callback. Returns immediately."""
        cmd = (cmd or "").strip().lower()
        if cmd not in _COMMANDS:
            return {"ok": False, "error": f"unknown command {cmd!r}; "
                                          f"try {sorted(_COMMANDS)}"}
        if cmd == "task" and not (arg or "").strip():
            return {"ok": False, "error": "task needs a non-empty argument"}
        self._pending.put((cmd, arg))
        return {"ok": True, "queued": cmd}

    def drain(self) -> list[tuple[str, str | None]]:
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "task": self.task,
                "paused": self.paused,
                "held": self.held,
                "ticks": self.ticks,
                "resets": self.resets,
                "uptime_s": round(time.perf_counter() - self.started_at, 1),
                "last_event": self.last_event,
            }

    def note(self, event: str) -> None:
        with self._lock:
            self.last_event = event


_COMMANDS = {"task", "reset", "park", "pause", "resume", "home", "quit"}


# ---- control channel ------------------------------------------------------
class RolloutNode(Node):
    """The ROS side of the channel. Callbacks only queue; the loop calls publish()."""

    def __init__(self, control: Control, publish_every: int) -> None:
        super().__init__("rollout")
        self.control = control
        # The JPEG encode sits on the loop's critical path, and an agent
        # reasoning at ~1 Hz needs far fewer frames.
        self.publish_every = publish_every
        self._seq = 0
        self.create_subscription(String, "/rollout/task",
                                 lambda m: control.submit("task", m.data), 10)
        for cmd in sorted(_COMMANDS - {"task"}):
            self.create_service(Trigger, f"/rollout/{cmd}",
                                lambda _req, res, c=cmd: self._trigger(c, res))
        self._status_pub = self.create_publisher(String, "/rollout/status", 10)
        self._state_pub = self.create_publisher(JointState, "/rollout/state", 10)
        self._image_pubs: dict = {}
        self._cameras: tuple[str, ...] = ()  # replaced, never mutated: read from the timer
        self.create_timer(0.5, self._publish_status)

    def _trigger(self, cmd: str, response: Trigger.Response) -> Trigger.Response:
        out = self.control.submit(cmd)
        response.success = out["ok"]
        response.message = out.get("queued") or out.get("error", "")
        return response

    def _publish_status(self) -> None:
        status = {**self.control.snapshot(), "cameras": list(self._cameras)}
        self._status_pub.publish(String(data=json.dumps(status)))

    def publish(self, obs: dict) -> None:
        """Publish the tick's observation for watchers.

        Failures are swallowed: a watcher missing a frame must never interrupt
        the robot.
        """
        self._seq += 1
        if self._seq % self.publish_every:
            return
        try:
            stamp = self.get_clock().now().to_msg()
            state = JointState()
            state.header.stamp = stamp
            for k, v in obs.items():
                if getattr(v, "ndim", 0) == 3:
                    if k not in self._image_pubs:
                        self._image_pubs[k] = self.create_publisher(
                            CompressedImage, f"/rollout/images/{k}/compressed", 1)
                        self._cameras = tuple(sorted(self._image_pubs))
                    # LeRobot hands out RGB; cv2 encodes BGR.
                    ok, buf = cv2.imencode(".jpg", np.asarray(v)[:, :, ::-1],
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        # array.array, not bytes: rclpy copies bytes element by element.
                        msg = CompressedImage(format="jpeg", data=array.array("B", buf.tobytes()))
                        msg.header.stamp = stamp
                        self._image_pubs[k].publish(msg)
                elif isinstance(v, (int, float)):
                    state.name.append(k)
                    state.position.append(float(v))
            self._state_pub.publish(state)
        except Exception:
            logger.debug("publish() failed", exc_info=True)


# ---- strategy -------------------------------------------------------------
def build_strategy_class():
    """Import LeRobot lazily and return the interactive strategy class.

    Deferred because importing ``lerobot.rollout`` pulls in torch and the plugin
    registry, which must not happen before ``main`` rewrites argv.
    """
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.robot_utils import precise_sleep

    class InteractiveStrategy(BaseStrategy):
        """``BaseStrategy`` whose loop checks a control queue every tick.

        Reimplemented rather than wrapped because ``pause``/``resume`` are no-op
        hooks on the ABC, so pausing has to happen in the caller. Everything
        else is reused from the parent unchanged.
        """

        def __init__(self, config, control: Control, channel: RolloutNode):
            super().__init__(config)
            self.control = control
            self.channel = channel

        # -- helpers ----------------------------------------------------
        def retarget(self, task: str) -> None:
            """Point the policy at a new instruction, starting from a clean chunk.

            Also releases a hold — submitting a task is the operator saying go.
            """
            engine = self._engine
            engine._task = task
            # Without this the queued chunk from the previous prompt keeps
            # playing; see the module docstring.
            engine.reset()
            if self._interpolator is not None:
                self._interpolator.reset()
            self._cached_obs_processed = None
            with self.control._lock:
                self.control.task = task
                was_held = self.control.held
                self.control.held = False
            logger.info("task -> %r%s", task, " (released hold)" if was_held else "")

        def hold(self, event: str) -> None:
            """Park the loop: stop stepping the policy until a task arrives."""
            with self.control._lock:
                self.control.held = True
            self.control.note(event)
            logger.info("HOLDING at home — send a task to start moving again")

        def home_action(self, robot) -> dict:
            return {k: self.control.home_pos for k in robot.action_features}

        def go_home(self, ctx) -> bool:
            """Interpolate to the home pose over ``home_duration_s``.

            Commands the trajectory rather than letting the Δq clamp rate-limit
            a single target, so speed is a property of this move and not a side
            effect of a safety bound. Eased with smoothstep, so velocity starts
            and ends at zero.

            Returns: True if home was reached within tolerance.
            """
            robot = ctx.hardware.robot_wrapper
            cfg = ctx.runtime.cfg
            target = self.home_action(robot)
            dt = 1.0 / max(cfg.fps, 1.0)
            duration = max(self.control.home_duration_s, 0.0)

            def joint_err(obs) -> float:
                return max((abs(float(obs[k]) - target[k])
                            for k in target if k in obs), default=0.0)

            obs = robot.get_observation()
            if joint_err(obs) <= self.control.reset_tolerance:
                logger.info("already home")
                return True

            # From where the arm actually is. An unreported joint is pinned to
            # its target rather than swept from a guessed start.
            start = {k: (float(obs[k]) if k in obs else target[k]) for k in target}

            t_start = time.perf_counter()
            while (elapsed := time.perf_counter() - t_start) < duration:
                if ctx.runtime.shutdown_event.is_set():
                    return False
                tick = time.perf_counter()
                a = elapsed / duration
                a = a * a * (3.0 - 2.0 * a)              # smoothstep
                robot.send_action({k: start[k] + (target[k] - start[k]) * a
                                   for k in target})
                if (rest := dt - (time.perf_counter() - tick)) > 0:
                    precise_sleep(rest)

            # The arm lags the waypoint, so hold the target until it arrives.
            # Sent at least once: the ramp exits a hair short of the target.
            deadline = time.perf_counter() + self.control.reset_timeout_s
            while True:
                if ctx.runtime.shutdown_event.is_set():
                    return False
                tick = time.perf_counter()
                robot.send_action(target)
                err = joint_err(robot.get_observation())
                if err <= self.control.reset_tolerance or time.perf_counter() > deadline:
                    break
                if (rest := dt - (time.perf_counter() - tick)) > 0:
                    precise_sleep(rest)

            if err <= self.control.reset_tolerance:
                logger.info("home reached in %.1fs (max joint error %.4f rad)",
                            time.perf_counter() - t_start, err)
                return True
            logger.warning("home NOT reached — last error %.4f rad. The arm is "
                           "wherever the ramp got to; it was not teleported.", err)
            return False

        def reset_scene(self, ctx) -> None:
            """Ask the sim to put the props back, if the robot is a sim robot.

            Best-effort: hardware has nothing to reset and an older
            policy_server has no /reset route. Neither aborts the reset.
            """
            robot = getattr(ctx.hardware.robot_wrapper, "inner", None)
            request = getattr(robot, "_request", None)
            if request is None:
                return
            try:
                out = request("/reset", {})
                logger.info("scene reset: %s", out.get("reset", "(no prop list)"))
            except Exception as e:
                logger.info("scene reset skipped (%s: %s)", type(e).__name__, e)

        def do_reset(self, ctx) -> None:
            """Home the arms, put the props back, clear the policy's queue."""
            logger.info("--- reset ---")
            self.go_home(ctx)
            self.reset_scene(ctx)
            self._engine.reset()
            if self._interpolator is not None:
                self._interpolator.reset()
            self._cached_obs_processed = None
            self._warmup_flushed = False
            with self.control._lock:
                self.control.resets += 1
                self.control.held = True
            logger.info("--- reset complete: parked, holding for a new task ---")

        def apply_commands(self, ctx) -> bool:
            """Drain the control queue. Returns False if asked to quit."""
            for cmd, arg in self.control.drain():
                if cmd == "task":
                    self.retarget(arg.strip())
                    self.control.note(f"task={arg.strip()!r}")
                elif cmd == "park":
                    # Server-side ramp, so an interactive park and an
                    # end-of-run park are the same code path.
                    robot = getattr(ctx.hardware.robot_wrapper, "inner", None)
                    request = getattr(robot, "_request", None)
                    if request is None:
                        logger.info("park: robot has no sim transport; "
                                    "using the local ramp instead")
                        self.go_home(ctx)
                    else:
                        try:
                            info = request("/park",
                                           {"duration_s": self.control.home_duration_s},
                                           timeout=30.0)
                            logger.info("parked (residual %.4f rad)",
                                        info.get("max_residual_rad", float("nan")))
                        except Exception as e:
                            logger.warning("park failed (%s: %s)",
                                           type(e).__name__, e)
                    # park/reset/home all put the arm somewhere safe, so all
                    # three hold it there — otherwise the next tick undoes it.
                    self.hold("parked")
                elif cmd == "reset":
                    self.do_reset(ctx)          # sets the hold itself
                    self.control.note("reset")
                elif cmd == "home":
                    self.go_home(ctx)
                    self.hold("home")
                elif cmd == "pause":
                    with self.control._lock:
                        self.control.paused = True
                    logger.info("PAUSED -- policy is not being stepped")
                    self.control.note("paused")
                elif cmd == "resume":
                    # Resuming mid-chunk would replay stale actions computed
                    # from a pre-pause observation, so start clean.
                    self._engine.reset()
                    if self._interpolator is not None:
                        self._interpolator.reset()
                    self._cached_obs_processed = None
                    with self.control._lock:
                        self.control.paused = False
                        # Explicit operator "go", so it lifts a hold too: continue
                        # the current task after a reset without retyping it.
                        self.control.held = False
                    logger.info("RESUMED")
                    self.control.note("resumed")
                elif cmd == "quit":
                    logger.info("quit requested over control channel")
                    ctx.runtime.shutdown_event.set()
                    return False
            return True

        # -- loop -------------------------------------------------------
        def run(self, ctx) -> None:
            engine = self._engine
            cfg = ctx.runtime.cfg
            robot = ctx.hardware.robot_wrapper
            interpolator = self._interpolator
            control = self.control

            control_interval = interpolator.get_control_interval(cfg.fps)
            start_time = time.perf_counter()
            engine.resume()

            with control._lock:
                control.task = engine._task
            logger.info("interactive loop started -- task %r", engine._task)

            while not ctx.runtime.shutdown_event.is_set():
                loop_start = time.perf_counter()

                if not self.apply_commands(ctx):
                    break

                if cfg.duration > 0 and (loop_start - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                # Keep pulling observations while stopped, so the feed stays live
                # while you reposition something or decide what to send next.
                obs = robot.get_observation()
                # Publish before the check: a stopped loop is exactly when a
                # watcher most needs to see the scene.
                self.channel.publish(obs)

                if not control.stepping:
                    if (rest := control_interval - (time.perf_counter() - loop_start)) > 0:
                        precise_sleep(rest)
                    continue

                obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                with control._lock:
                    control.ticks += 1

                dt = time.perf_counter() - loop_start
                if (rest := control_interval - dt) > 0:
                    precise_sleep(rest)
                else:
                    logger.warning(
                        "loop running slower (%.1f Hz) than target %.0f Hz",
                        1 / dt, cfg.fps)

    return InteractiveStrategy


# ---- entry point ----------------------------------------------------------
def _extract_own_flags(argv: list[str]) -> tuple[dict, list[str]]:
    """Pull this module's own flags out of argv before draccus sees it.

    draccus errors on anything it does not recognise, so rather than fork its
    config dataclass the extra flags are removed and the rest passed verbatim.
    """
    own = {"--home": ("home", float),
           "--home-duration": ("home_duration", float),
           "--publish-every": ("publish_every", int)}
    out, rest, i = {}, [], 0
    while i < len(argv):
        tok = argv[i]
        key, _, inline = tok.partition("=")
        if key in own:
            name, cast = own[key]
            if inline:
                out[name] = cast(inline)
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{key} needs a value")
                out[name] = cast(argv[i + 1])
                i += 2
            continue
        rest.append(tok)
        i += 1
    return out, rest


def main() -> int:
    own, rest = _extract_own_flags(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]

    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import (
        init_visualization,
        shutdown_visualization,
    )
    from lerobot.utils.import_utils import register_third_party_plugins

    # Registers yam_ultra_sim / yam_ultra_bimanual. Must run before the config
    # is parsed, or --robot.type is rejected as an unknown choice.
    register_third_party_plugins()
    InteractiveStrategy = build_strategy_class()

    def _run(cfg):
        init_logging()
        if cfg.display_data:
            init_visualization(cfg.display_mode, session_name="rollout",
                               ip=cfg.display_ip, port=cfg.display_port)

        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        ctx = build_rollout_context(cfg, signal_handler.shutdown_event)

        control = Control(
            home_pos=own.get("home", DEFAULT_HOME),
            home_duration_s=own.get("home_duration", Control.home_duration_s),
        )
        ros.init()
        channel = RolloutNode(control, max(1, int(own.get("publish_every", 1))))
        close_channel = ros.spin_in_background(channel)

        strategy = InteractiveStrategy(cfg.strategy, control, channel)
        try:
            strategy.setup(ctx)
            strategy.run(ctx)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            strategy.teardown(ctx)
            close_channel()
            if cfg.display_data:
                shutdown_visualization(cfg.display_mode)
        logger.info("Rollout finished")

    # parser.wrap() reads the RAW annotation, which PEP 563 makes a string
    # here, so bind the real class rather than writing out `cfg: RolloutConfig`.
    _run.__annotations__["cfg"] = RolloutConfig
    parser.wrap()(_run)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
