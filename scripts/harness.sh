#!/usr/bin/env bash
# Web harness: a high-level agent in front of the low-level VLA.
#
#     ./scripts/harness.sh                      # scripted agent, no API key
#     ./scripts/harness.sh --agent gemini       # needs HARNESS_MODEL + GOOGLE_API_KEY
#     ./scripts/harness.sh --agent none         # UI + REST only, nothing decides
#
# Drives a live rollout over /rollout/*; that process owns the arms and the policy:
#
#     ros2 launch sparklab_yam yam.launch.py   # terminal 1
#     ./scripts/rollout.sh --mode=live         # terminal 2
#     ./scripts/harness.sh                     # terminal 3, then http://127.0.0.1:8099/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec ros2 run lerobot_robot_sparklab harness --port "${HARNESS_PORT:-8099}" "$@"
