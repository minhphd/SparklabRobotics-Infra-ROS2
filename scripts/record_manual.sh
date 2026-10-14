#!/usr/bin/env bash
# Teleop dataset recording with manual episode boundaries: space or Quest B/Y
# starts/ends each episode (arms park to zeros on end), r discards it, q quits.
#
#   ros2 launch sparklab_yam yam.launch.py          # once, without teleop:=true
#   ./scripts/record_manual.sh [extra flags for record_manual]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec ros2 run lerobot_robot_sparklab record_manual \
    --robot.type=yam_ultra_bimanual \
    --teleop.type=bi_quest_teleop \
    --dataset.repo_id=minhphd/clean \
    --dataset.single_task="clean up the tables" \
    --dataset.num_episodes=100 --dataset.fps=30 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    "$@"
