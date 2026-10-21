"""cap-x's ReAct agent and dashboard driving the YAM through the TAMP tools.

    ./scripts/capx_python.sh -m vlm_tamp_sparklab.agent --model gemini:gemini-3.8-flash
    ROS_DOMAIN_ID=42 ./scripts/capx_python.sh -m vlm_tamp_sparklab.agent ...   # in the twin

Defaults to this package's config and all three cameras, so the model can check
grasps through the wrist cameras. The dashboard serves on :8090.
"""

from __future__ import annotations

import tyro

from capx.agent import react_loop

from . import CONFIG


def main() -> None:
    cameras = ("top", "left_wrist", "right_wrist")
    react_loop.main(tyro.cli(react_loop.Args, default=react_loop.Args(config_path=str(CONFIG), cameras=cameras)))


if __name__ == "__main__":
    main()
