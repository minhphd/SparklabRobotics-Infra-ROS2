"""cap-x's ReAct agent and dashboard on the YAM rig: capx/agent/react_loop.py with this rig's defaults.

    ./scripts/capx_python.sh -m capx_sparklab.agent --model gemini:gemini-3.8-flash

Defaults to this package's config and the top camera. The dashboard serves on
:8090 and the robot's 3D scene on :8080. The dashboard's point cloud, on :8081, also
shows every perceived object's points and box, and the last grasp candidates.
"""

from __future__ import annotations

from unittest import mock

import tyro

from capx.agent import dashboard, react_loop

from . import CONFIG


class Dashboard(dashboard.Dashboard):
    """cap-x's dashboard, with the env's scene drawing into its point cloud too."""

    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(env, *args, **kwargs)
        server = self._cloud._server
        # cap-x reads server.port, which viser 1.0 lacks, so the page kept 8081 when viser moved off it.
        self._page = dashboard._PAGE.replace("CLOUD_PORT", str(server.get_port()))
        env.low_level_env.scene.viewer = server


def main() -> None:
    args = tyro.cli(react_loop.Args, default=react_loop.Args(config_path=str(CONFIG), cameras=("top",)))
    with mock.patch.object(dashboard, "Dashboard", Dashboard):
        react_loop.main(args)


if __name__ == "__main__":
    main()
