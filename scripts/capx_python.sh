#!/usr/bin/env bash
# Run something with cap-x's GPU stack and this workspace's ROS 2 in one Python.
#
# cap-x's venv (CUDA torch, cuRobo, GraspGenX) and sparklab-ros (rclpy) are both
# CPython 3.12: this env's interpreter runs isolated, cap-x's packages first.
#
#   ./scripts/capx_python.sh -m capx_sparklab.robot.build    # module
#   ./scripts/capx_python.sh -m pytest src/capx_sparklab     # tests
#   ./scripts/capx_python.sh                                 # REPL
#
# Needs sparklab-ros active and the workspace built. The cap-x checkout (with its
# venv in .venv) defaults to ../cap-x next to this repo; override with CAPX=/path.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPX="${CAPX:-$(cd "$REPO/.." && pwd)/cap-x}"
CAPX_SITE="$CAPX/.venv/lib/python3.12/site-packages"

[[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python3.12" ]] || {
    echo "ERROR: activate the sparklab-ros env first." >&2
    exit 1
}
[[ -d "$CAPX_SITE" ]] || {
    echo "ERROR: no cap-x venv at $CAPX_SITE — set CAPX=/path/to/cap-x." >&2
    exit 1
}

# setup.bash reads unset variables.
set +u
# shellcheck disable=SC1091
source "$REPO/install/setup.bash"
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CAPX CAPX_SITE
# The Vertex service-account file cap-x uses for gemini: models, unless one is set already.
if [[ -z "${GEMINI_KEY_FILE:-}" && -f "$HOME/.config/capx/gemini_key.json" ]]; then
    export GEMINI_KEY_FILE="$HOME/.config/capx/gemini_key.json"
fi
export ROS_PYTHONPATH="$PYTHONPATH:$CONDA_PREFIX/lib/python3.12/site-packages"
exec "$CONDA_PREFIX/bin/python3.12" -I -S "$REPO/scripts/capx_boot.py" "$@"
