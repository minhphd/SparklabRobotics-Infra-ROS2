# lerobot_robot_sparklab

The bridge between this robot and [LeRobot](https://github.com/huggingface/lerobot).
Once the workspace is built and sourced, every `lerobot-*` command understands:

```
--robot.type=yam_ultra_bimanual   the real arms (talks to the drivers over ROS)
--robot.type=yam_ultra_sim        the Isaac Sim twin (talks to it over HTTP)
--teleop.type=bi_quest_teleop     the Quest headset
```

```
lerobot_robot_sparklab/follower.py       the real robot, as a LeRobot Robot
lerobot_robot_sparklab/teleoperator.py   the Quest, as a LeRobot Teleoperator
lerobot_robot_sparklab/sim_follower.py   the Isaac twin, as a LeRobot Robot
lerobot_robot_sparklab/record_manual.py  record a dataset, episodes started/ended by hand
lerobot_robot_sparklab/rollout/          run a policy you can retarget while it runs
lerobot_robot_sparklab/harness/          web page + AI agent that picks the policy's task
lerobot_robot_sparklab/ros.py            small rclpy helpers
```

Everything here needs the robot started first: `ros2 launch sparklab_yam yam.launch.py`.

## The robot (`yam_ultra_bimanual`)

It opens no hardware itself. It reads `/<arm>/joint_states` and
`/cameras/<id>/color/image_raw`, and sends `/<arm>/arm/command` and
`/<arm>/gripper/command`.

- **Observation:** `left_joint_1.pos` … `left_joint_6.pos`, `left_gripper.pos`,
  the same for `right_`, and one RGB image per camera.
- **Action:** the same `.pos` keys. Each step is clamped the same way the driver
  clamps, and `send_action` returns what was actually sent.
- **Safety:** if joint states or images are older than 0.5 s it raises an error
  instead of acting on stale data. On disconnect it parks both arms.

Options: `max_relative_target` (the per-step clamp), `park_on_disconnect`,
`camera_ids`, `connect_timeout_s`.

## The teleoperator (`bi_quest_teleop`)

A thin wrapper around `sparklab_yam`'s `QuestTeleop`; its settings are flags
(e.g. `--teleop.scale_translation=1.2`). Call `seed_qpos_from_obs(obs)` after
anything else has moved the arms.

## Recording (`record_manual`)

Started by `./scripts/record_manual.sh`. Takes the same flags as `lerobot-record`,
but you start and stop episodes yourself:

| key | does |
|---|---|
| Space, or B/Y on a controller | start / end an episode (saved, then the arms park) |
| `r` | throw away the current episode (and park) |
| `q` / Esc | quit (an unfinished episode is thrown away) |

Teleop keeps working between episodes. The recorded action is the clamped one
that was actually sent. By default it also records the raw headset and
controller poses (`observation.xr.*`). `--resume=true` adds to an existing
dataset.

## More

- [`rollout/`](lerobot_robot_sparklab/rollout/README.md) — retarget a running policy
- [`harness/`](lerobot_robot_sparklab/harness/README.md) — the AI agent web page
