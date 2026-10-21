#!/usr/bin/env bash
# Run Python inside Isaac Sim 5.1, with this repo's src/ and Isaac's bundled ROS 2 Humble.
#
#   ./scripts/isaac_python.sh -m sparklab_sim.twin              # the digital twin
#   ./scripts/isaac_python.sh -m sparklab_sim.convert --twin    # robot USD, once
#
# Isaac is the pip install in the conda env ISAAC_ENV (default: behavior, which
# also holds OmniGibson). Its rclpy talks DDS to sparklab-ros on the same domain.

set -euo pipefail

ISAAC_ENV="${ISAAC_ENV:-$HOME/.conda/envs/behavior}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2="$ISAAC_ENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble"

[[ -x "$ISAAC_ENV/bin/python" && -d "$ROS2" ]] || {
    echo "ERROR: no Isaac Sim with its ROS 2 bridge in $ISAAC_ENV — set ISAAC_ENV." >&2
    exit 1
}

# A sourced ROS workspace or conda env would put a second, mismatched ROS on these paths.
export LD_LIBRARY_PATH="$ROS2/lib"
export PYTHONPATH="$REPO/src:$ROS2/rclpy"
export ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OMNI_KIT_ACCEPT_EULA=YES
exec "$ISAAC_ENV/bin/python" "$@"
