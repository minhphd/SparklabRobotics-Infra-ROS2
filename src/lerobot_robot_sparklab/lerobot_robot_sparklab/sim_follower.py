"""The Isaac twin as a LeRobot Robot — ``--robot.type=yam_ultra_sim``.

Talks to ``sparklab_sim.policy_server`` over HTTP and presents the exact
observation/action schema the hardware follower does, so a rollout runs
against the simulator with no policy changes::

    ./scripts/isaac_python.sh -m sparklab_sim.policy_server
    lerobot-rollout --robot.type=yam_ultra_sim ... --policy.path=...

Kinematic only: joint targets are written straight through, nothing resists
them and the gripper closes on nothing. Good for checking that a checkpoint
produces sane, in-range actions from real camera geometry; useless for judging
whether a grasp succeeds. See DESIGN.md.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from lerobot.cameras import CameraConfig  # noqa: F401  (schema parity)
from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig

from sparklab_yam.constants import ARM_JOINTS, DEFAULT_MAX_RELATIVE_TARGET, HANDS

logger = logging.getLogger(__name__)

# Fixed by the recorded dataset's meta/info.json; the server crops to match.
IMAGE_SHAPE = (360, 640, 3)
CAMERA_NAMES = ("top", "left_wrist", "right_wrist")


@RobotConfig.register_subclass("yam_ultra_sim")
@dataclass
class YamUltraSimConfig(RobotConfig):
    """Config for the Isaac-backed bimanual YAM-Ultra."""

    host: str = "127.0.0.1"
    port: int = 8081

    # Generous: failing a long rollout over one slow first frame is worse
    # than waiting for it.
    timeout_s: float = 20.0

    # Isaac takes ~15 s to boot cold, so don't give up on it immediately.
    connect_timeout_s: float = 90.0

    # The server is a persistent workspace, so the pose a run ends in is the
    # pose the next one starts from. A crash is covered by its --auto-park-s.
    park_on_disconnect: bool = True
    park_seconds: float = 2.5

    # ---- action shaping, shared with the hardware follower -----------------
    max_relative_target: float | list[float] | None = field(
        default_factory=lambda: list(DEFAULT_MAX_RELATIVE_TARGET)
    )


class YamUltraSim(Robot):
    """Two simulated YAM-Ultra arms + three rendered cameras, as one Robot."""

    config_class = YamUltraSimConfig
    name = "yam_ultra_sim"

    def __init__(self, config: YamUltraSimConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._url = f"http://{config.host}:{config.port}"
        self._last_action: dict[str, float] | None = None
        # Returned by the last /step, waiting for get_observation().
        self._pending_obs: dict | None = None
        # send_action clamps against this, mirroring the YAM driver clamping
        # against the arm's measured present pose.
        self._last_state: dict[str, float] | None = None
        self._last_clamp_warn: float = 0.0

        # Per-tick Δq cap, resolved once from max_relative_target.
        mrt = config.max_relative_target
        self._caps: np.ndarray | None = (
            None if mrt is None
            else np.broadcast_to(np.asarray(mrt, dtype=float),
                                 (ARM_JOINTS,)).copy())

    # ---- feature schema -----------------------------------------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft: dict[str, type] = {}
        for h in HANDS:
            for j in range(1, ARM_JOINTS + 1):
                ft[f"{h}_joint_{j}.pos"] = float
            ft[f"{h}_gripper.pos"] = float
        return ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **{n: IMAGE_SHAPE for n in CAMERA_NAMES}}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ---- transport ----------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        """One request against the sim server.

        path: server route, e.g. "/step".
        payload: JSON body; None issues a GET.
        timeout: overrides the config default, which /park needs since it
            blocks for the length of its ramp.
        Returns: the decoded JSON body.
        """
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        with urllib.request.urlopen(
                req, timeout=self.config.timeout_s if timeout is None else timeout) as r:
            body = json.loads(r.read())
        if "error" in body:
            raise RuntimeError(f"sim server: {body['error']}")
        return body

    @staticmethod
    def _decode(images: dict[str, str]) -> dict[str, np.ndarray]:
        import cv2

        out = {}
        for name, b64 in images.items():
            buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not decode image {name!r}")
            frame = bgr[..., ::-1]           # BGR -> RGB
            if frame.shape != IMAGE_SHAPE:
                # Loud, not coerced: resizing would hand the policy a subtly
                # wrong field of view.
                raise RuntimeError(
                    f"camera {name!r} returned {frame.shape}, expected "
                    f"{IMAGE_SHAPE} — check CAMERAS in policy_server.py")
            out[name] = np.ascontiguousarray(frame)
        return out

    # ---- lifecycle ----------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Nothing to calibrate: the twin is defined by the URDF."""

    def configure(self) -> None:
        """No gains to set — joint targets are written kinematically."""

    def connect(self, calibrate: bool = True) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                health = self._request("/health")
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = e
                time.sleep(1.0)
        else:
            raise ConnectionError(
                f"no sim server on {self._url} after "
                f"{self.config.connect_timeout_s:.0f}s ({last}). Start it with:\n"
                f"    ./scripts/isaac_python.sh -m sparklab_sim.policy_server")

        served = set(health.get("cameras", {}))
        if served != set(CAMERA_NAMES):
            raise RuntimeError(
                f"sim server serves cameras {sorted(served)}, this robot "
                f"expects {sorted(CAMERA_NAMES)} — the observation schema is "
                f"fixed by the recorded dataset")
        self._connected = True
        logger.info("%s connected to %s (cameras: %s)", self, self._url,
                    ", ".join(sorted(served)))

    def disconnect(self) -> None:
        # Deliberately does not stop the server — but because it survives, the
        # pose left behind is inherited by the next run, so hand it back tidy.
        if self.config.park_on_disconnect and self._connected:
            try:
                # The server ramps it: it owns the workspace, and a park must
                # work the same way when another client asks for one.
                info = self._request(
                    "/park", {"duration_s": self.config.park_seconds},
                    timeout=self.config.timeout_s + self.config.park_seconds + 5.0)
                logger.info("%s parked arms (residual %.4f rad)", self,
                            info.get("max_residual_rad", float("nan")))
            except Exception as e:
                # Never let tidying up turn a finished rollout into a failure.
                logger.warning("%s could not park on disconnect (%s: %s); the "
                               "server's idle auto-park will catch it",
                               self, type(e).__name__, e)

        self._connected = False
        self._pending_obs = None
        self._last_state = None
        logger.info("%s disconnected (sim server left running).", self)

    # ---- control ------------------------------------------------------------
    def _require_live(self, where: str) -> None:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected ({where})")

    def _unpack(self, body: dict) -> dict:
        self._last_state = dict(body["state"])
        obs: dict[str, object] = dict(body["state"])
        obs.update(self._decode(body["images"]))
        return obs

    def get_observation(self) -> dict:
        """The observation from the most recent step, or a fresh render.

        ``/step`` already renders after applying the action, so consuming that
        result here makes a control tick one render instead of two. The cache
        is consumed, not kept, so two reads without an intervening action never
        return the same stale frame.
        """
        self._require_live("get_observation")
        if self._pending_obs is not None:
            body, self._pending_obs = self._pending_obs, None
            return self._unpack(body)
        return self._unpack(self._request("/obs"))

    # The 12 rate-limited arm-joint keys in clamp-vector order. Grippers are
    # deliberately absent — they are not rate-limited, see send_action.
    _ARM_KEYS = tuple(f"{h}_joint_{j}.pos"
                      for h in HANDS
                      for j in range(1, ARM_JOINTS + 1))

    def _dq_caps(self) -> np.ndarray | None:
        """Per-joint Δq allowance (rad) for one tick, or None if uncapped."""
        return self._caps

    def send_action(self, action: dict) -> dict:
        """Apply an action and return what was actually sent, Δq-clamped.

        One round trip does act-then-render; the frames land in the next
        get_observation(), the same ordering the hardware follower has.

        action: `.pos` keys for both hands' joints and grippers.
        Returns: the clamped values actually sent.
        """
        self._require_live("send_action")
        caps = self._dq_caps()

        # Against the present pose, not the last command, as the driver
        # does — otherwise error accumulates whenever the arm misses a target.
        present = self._last_state or {}
        last = self._last_action or {}

        # Hold the last command for omitted keys; zero would snap the arm home.
        target = np.fromiter(
            (float(action.get(k, last.get(k, 0.0))) for k in self._ARM_KEYS),
            dtype=float, count=len(self._ARM_KEYS))

        if caps is None:
            clamped = target
            worst = None
        else:
            # The same six caps apply to each hand, so tile rather than re-derive.
            cap = np.tile(caps, len(HANDS))
            p = np.fromiter(
                (float(present.get(k, t)) for k, t in zip(self._ARM_KEYS, target)),
                dtype=float, count=len(self._ARM_KEYS))
            step = target - p
            clamped = p + np.clip(step, -cap, cap)

            over = np.abs(step) - cap
            i = int(np.argmax(over))
            worst = ((float(over[i]), self._ARM_KEYS[i][:-len(".pos")])
                     if over[i] > 0 else None)

        sent: dict[str, float] = dict(zip(self._ARM_KEYS, clamped.tolist()))

        # Not rate-limited, matching the driver, which only clamps the arm
        # joints: a 0..1 command onto a small prismatic pair throws nothing.
        for h in HANDS:
            gk = f"{h}_gripper.pos"
            sent[gk] = float(action.get(gk, last.get(gk, 0.0)))

        if worst is not None:
            # Rate-limited like the hardware follower's: sustained clamping is
            # normal, and one line per tick would bury every other message.
            now = time.monotonic()
            if now - self._last_clamp_warn > 1.0:
                self._last_clamp_warn = now
                logger.warning(
                    "sim action exceeded this tick's cap on %s by %.3f rad — "
                    "clamped. Hardware would clamp here too (further messages "
                    "suppressed for 1s)", worst[1], worst[0])

        self._pending_obs = self._request("/step", {"action": sent})
        if self._pending_obs is not None:
            self._last_state = dict(self._pending_obs["state"])
        self._last_action = dict(sent)
        return sent
