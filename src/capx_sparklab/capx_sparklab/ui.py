"""cap-x's browser UI on the YAM rig: capx/agent/flask_ui.py with this rig's defaults.

    ./scripts/capx_python.sh -m capx_sparklab.ui --model gemini:gemini-3.1-pro-preview

Defaults to this package's config and the top camera. cap-x's page shows LIBERO's
"agentview" camera, which the rig does not have; here it shows the first of --cameras.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import tyro

from capx.agent import flask_ui

from . import CONFIG


def panel_encoder(camera: str) -> Callable[[dict], str | None]:
    """flask_ui.encode_frame, for one of the rig's cameras."""
    return functools.partial(flask_ui.encode_frame, camera=camera)


def main() -> None:
    args = tyro.cli(flask_ui.UIArgs,
                    default=flask_ui.UIArgs(config_path=str(CONFIG), cameras=("top",), max_turns=40))
    flask_ui.encode_frame = panel_encoder(args.cameras[0])
    flask_ui.main(args)


if __name__ == "__main__":
    main()
