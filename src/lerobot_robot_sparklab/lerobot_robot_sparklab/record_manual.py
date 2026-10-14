"""Manual-clutch dataset recording — the spacebar replaces the episode/reset timers.

Takes the same ``--robot.* / --teleop.* / --dataset.*`` flags as ``lerobot-record``
but ignores ``episode_time_s`` / ``reset_time_s``: space or B/Y ends the episode
(saved immediately, then the arms park to the all-zeros pose) and another press
starts the next one; teleop stays live between episodes. ``r`` discards the
episode in progress (also parks), ``q``/Esc exits (a half-recorded episode is
discarded, not saved).

Teleop and robot run at ``control_hz`` (default 120); dataset frames are taken
every ``control_hz / fps``-th tick. Needs the rig up
(``ros2 launch sparklab_yam yam.launch.py``, without ``teleop:=true``)::

    ros2 run lerobot_robot_sparklab record_manual \\
        --robot.type=yam_ultra_bimanual --teleop.type=bi_quest_teleop \\
        --dataset.repo_id=minhphd/task --dataset.single_task="..." \\
        --dataset.num_episodes=50
"""

import logging
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np

from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from sparklab_quest.buttons import B_Y

logger = logging.getLogger(__name__)

HELP_LINE = "space or B/Y = start/end episode | r = discard episode | q/Esc = quit"


@dataclass
class ManualRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig
    # Teleop/robot control rate. The teleop's filters and Δq caps are per-tick
    # constants tuned near 200 Hz; recording ticks run at dataset.fps.
    control_hz: float = 120.0
    # Debug: record raw XR poses (both controllers + headset) and buttons.
    log_xr: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        ratio = self.control_hz / self.dataset.fps
        if abs(ratio - round(ratio)) > 1e-6 or ratio < 1:
            raise ValueError(
                f"control_hz ({self.control_hz}) must be an integer multiple "
                f"of dataset.fps ({self.dataset.fps})"
            )


@dataclass
class _Session:
    started_t: float
    saved: int = 0
    discarded: int = 0
    last_ep_s: float = 0.0
    total_recorded_s: float = 0.0
    loop_hz: float = 0.0


def _print_stats(dataset: LeRobotDataset, sess: _Session, fps: int) -> None:
    total_s = dataset.num_frames / fps
    mean_s = total_s / dataset.num_episodes if dataset.num_episodes else 0.0
    elapsed = time.perf_counter() - sess.started_t
    sys.stdout.write(
        "\r\033[K"
        + "\n".join(
            [
                "─" * 62,
                f"▶ RECORDING episode {dataset.num_episodes}   (space = end episode)",
                f"  saved: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
                f" ({total_s / 60:.1f} min at {fps} fps)",
                f"  this session: {sess.saved} saved, {sess.discarded} discarded,"
                f" last ep {sess.last_ep_s:.1f}s",
                f"  mean episode: {mean_s:.1f}s   session: {elapsed / 60:.1f} min"
                f"   loop: {sess.loop_hz:.1f} Hz",
                f"  {HELP_LINE}",
                "─" * 62,
            ]
        )
        + "\n"
    )
    sys.stdout.flush()


_XR_BUTTONS_N = 16  # Touch Plus reports 12-13; padded so the shape is fixed


def _xr_features() -> dict:
    """Dataset features for XR debug logging. Poses are Quest `local-floor`."""
    pose = {
        "dtype": "float32",
        "shape": (8,),
        "names": ["px", "py", "pz", "qx", "qy", "qz", "qw", "valid"],
    }
    buttons = {
        "dtype": "float32",
        "shape": (_XR_BUTTONS_N,),
        "names": [f"b{i}_value" for i in range(_XR_BUTTONS_N)],
    }
    return {
        "observation.xr.left_pose": dict(pose),
        "observation.xr.right_pose": dict(pose),
        "observation.xr.headset_pose": dict(pose),
        "observation.xr.left_buttons": dict(buttons),
        "observation.xr.right_buttons": dict(buttons),
        "observation.xr.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["left_engaged", "right_engaged", "frame_age_s"],
        },
    }


def _pose_vec(entry: dict | None) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    if entry is not None:
        out[0:3] = np.asarray(entry["position"], dtype=np.float32)
        out[3:7] = np.asarray(entry["orientation"], dtype=np.float32)  # xyzw
        out[7] = 1.0
    return out


def _buttons_vec(entry: dict | None) -> np.ndarray:
    out = np.zeros(_XR_BUTTONS_N, dtype=np.float32)
    if entry is not None:
        values = list(entry.get("values") or [])[:_XR_BUTTONS_N]
        out[:len(values)] = values
    return out


def _xr_frame_values(teleop) -> dict[str, np.ndarray]:
    snap = getattr(teleop, "xr_snapshot", lambda: None)() or {}
    ctrls = snap.get("controllers") or {}
    engaged = snap.get("engaged") or {}
    return {
        "observation.xr.left_pose": _pose_vec(ctrls.get("left")),
        "observation.xr.right_pose": _pose_vec(ctrls.get("right")),
        "observation.xr.headset_pose": _pose_vec(snap.get("viewer")),
        "observation.xr.left_buttons": _buttons_vec(ctrls.get("left")),
        "observation.xr.right_buttons": _buttons_vec(ctrls.get("right")),
        "observation.xr.state": np.array(
            [
                float(engaged.get("left", False)),
                float(engaged.get("right", False)),
                float(snap.get("age_s", -1.0)),
            ],
            dtype=np.float32,
        ),
    }


def _park_and_reseed(robot, teleop) -> None:
    """Ramp the arms to the all-zeros park pose, then re-sync teleop to it.

    The reseed force-disengages the clutch, so without it the next tick would
    snap the arms back to the pre-park pose. Blocks for the ramp.
    """
    if hasattr(robot, "park"):
        robot.park()
    if hasattr(teleop, "seed_qpos_from_obs"):
        teleop.seed_qpos_from_obs(robot.get_observation())


def _status_line(state: str, frames: int, seconds: float) -> str:
    if state == "recording":
        return f"● REC  {seconds:6.1f}s  {frames} frames   (space = end episode)"
    return "○ RESET — reposition, then space = start next episode"


@parser.wrap()
def record_manual(cfg: ManualRecordConfig) -> LeRobotDataset:
    init_logging()

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    if cfg.log_xr:
        dataset_features = {**dataset_features, **_xr_features()}

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    if cfg.resume:
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        )
    else:
        cfg.dataset.stamp_repo_id()
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        )

    # A resumed dataset's schema wins: no XR columns there means none can be added.
    log_xr = cfg.log_xr and "observation.xr.left_pose" in dataset.features
    if cfg.log_xr and not log_xr:
        logging.warning("resumed dataset has no XR features — XR logging disabled")
    # xr.* is filled from the teleop, not the robot obs; build_dataset_frame
    # must not try to source its names ("px", ...) from obs_processed.
    frame_features = {
        k: v for k, v in dataset.features.items() if not k.startswith("observation.xr.")
    }

    # Key presses land on the listener thread; the loop consumes the flags at
    # tick boundaries so an episode never splits mid-frame.
    events = {"toggle": False, "discard": False, "stop": False}

    def on_key(name: str) -> None:
        key = name.lower()
        if key == "space":
            events["toggle"] = True
        elif key == "r":
            events["discard"] = True
        elif key in ("esc", "q"):
            events["stop"] = True

    fps = cfg.dataset.fps
    interval = 1.0 / cfg.control_hz
    record_every = round(cfg.control_hz / fps)
    status_every = max(1, round(cfg.control_hz / 10))
    state = "reset"
    listener = None

    try:
        robot.connect()
        teleop.connect()
        # Required before teleop drives the robot (DESIGN.md#teleoperation);
        # without it the first action snaps the arms to the teleop's rest pose.
        if hasattr(teleop, "seed_qpos_from_obs"):
            teleop.seed_qpos_from_obs(robot.get_observation())
        listener = create_key_listener(on_key, controls_help=HELP_LINE)
        if listener is None:
            raise RuntimeError("No keyboard backend available — manual recording needs one.")

        sess = _Session(started_t=time.perf_counter())
        ep_frames = 0
        ep_start_t = 0.0
        last_handoff = False
        tick = 0
        obs = robot.get_observation()

        print(f"\nSTANDBY — teleop live, nothing recording yet.  {HELP_LINE}\n")

        with VideoEncodingManager(dataset):
            while not events["stop"] and dataset.num_episodes < cfg.dataset.num_episodes:
                t0 = time.perf_counter()

                # Full observation (motors + cameras) only on record ticks;
                # control ticks reuse the last one — the default processors
                # are identity and never look at it.
                record_tick = state == "recording" and tick % record_every == 0
                if record_tick:
                    obs = robot.get_observation()
                    obs_processed = robot_observation_processor(obs)

                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                sent = robot.send_action(robot_action_processor((act_processed, obs)))

                # Quest B/Y as a second episode toggle, read straight off the
                # Quest topics. Stow is the thumbstick, so no conflict.
                quest = getattr(teleop, "quest", None)
                handoff = quest is not None and quest.pressed(B_Y)
                if handoff and not last_handoff:
                    events["toggle"] = True
                last_handoff = handoff

                if record_tick:
                    # The Δq-clamped action the arms were actually told to take,
                    # so recorded action and state can never diverge unbounded.
                    frame = {
                        **build_dataset_frame(frame_features, obs_processed, prefix=OBS_STR),
                        **build_dataset_frame(frame_features, sent, prefix=ACTION),
                        "task": cfg.dataset.single_task,
                    }
                    if log_xr:
                        frame.update(_xr_frame_values(teleop))
                    dataset.add_frame(frame)
                    ep_frames += 1

                if events["discard"]:
                    events["discard"] = False
                    if state == "recording":
                        dataset.clear_episode_buffer()
                        sess.discarded += 1
                        state = "reset"
                        print(f"\n✗ discarded episode after {ep_frames} frames — parking\n")
                        _park_and_reseed(robot, teleop)

                if events["toggle"]:
                    events["toggle"] = False
                    if state == "reset":
                        state = "recording"
                        ep_frames = 0
                        tick = -1  # first recording tick lands on a record tick
                        ep_start_t = time.perf_counter()
                        _print_stats(dataset, sess, fps)
                    else:
                        state = "reset"
                        sess.last_ep_s = time.perf_counter() - ep_start_t
                        sess.total_recorded_s += sess.last_ep_s
                        dataset.save_episode()
                        sess.saved += 1
                        print(
                            f"\n✔ saved episode {dataset.num_episodes - 1}: "
                            f"{ep_frames} frames, {sess.last_ep_s:.1f}s — parking\n"
                        )
                        _park_and_reseed(robot, teleop)

                dt = time.perf_counter() - t0
                if state == "recording" and dt > 1.0 / fps:
                    logger.warning(
                        "control tick took %.0f ms — below the %d fps record rate", dt * 1e3, fps
                    )
                precise_sleep(max(interval - dt, 0.0))
                full = time.perf_counter() - t0
                if full > 0:
                    sess.loop_hz = 0.9 * sess.loop_hz + 0.1 / full if sess.loop_hz else 1.0 / full

                tick += 1
                if tick % status_every == 0:
                    seconds = time.perf_counter() - ep_start_t if state == "recording" else 0.0
                    sys.stdout.write("\r\033[K" + _status_line(state, ep_frames, seconds))
                    sys.stdout.flush()

            if state == "recording":
                dataset.clear_episode_buffer()
                print(f"\n✗ quit mid-episode — {ep_frames} frames discarded (not saved)")
                state = "reset"
    finally:
        # A partial buffer would poison finalize(); drop it on any exit path.
        if state == "recording":
            dataset.clear_episode_buffer()
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        # A zero-episode dataset is an unloadable shell (no tasks.parquet) that
        # 404s in every tool later; delete it rather than leave the corpse.
        if not cfg.resume and dataset.num_episodes == 0:
            shutil.rmtree(dataset.root, ignore_errors=True)
            print(f"\nno episodes saved — removed empty dataset at {dataset.root}")
        else:
            print(
                f"\ndone: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
                f" at {dataset.root}"
            )
        if cfg.dataset.push_to_hub:
            if dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")
    return dataset


def main() -> None:
    register_third_party_plugins()
    record_manual()


if __name__ == "__main__":
    main()
