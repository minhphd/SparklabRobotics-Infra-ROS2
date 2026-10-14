#!/usr/bin/env bash
# Fixed-duration teleop dataset recording via lerobot-record. Needs the rig up,
# without teleop:=true:
#     ros2 launch sparklab_yam yam.launch.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

lerobot-record \
    --robot.type=yam_ultra_bimanual \
    --teleop.type=bi_quest_teleop \
    --dataset.repo_id=minhphd/test \
    --dataset.single_task="put stuff in a box" \
    --dataset.num_episodes=1 --dataset.fps=30 \
    --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false
