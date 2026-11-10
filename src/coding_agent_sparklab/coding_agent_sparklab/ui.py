"""The coding-agent page: the rig's three cameras on the left, a chat with Claude Code or Antigravity on the right.

    ros2 run coding_agent_sparklab ui                          # http://localhost:8090
    ros2 run coding_agent_sparklab ui --port 8091 --workdir ~/robot_scripts
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from .agents import Antigravity, Chat, ClaudeCode
from .robot import ARMS, CAMERAS

PORT = 8090
JPEG_QUALITY = 80
PARK_TIMEOUT = 20.0
"""s to wait on both drivers' park, which takes park_duration_s and at most twice that."""


class Rig(Node):
    """The newest colour frame of each camera, as JPEG on request, and the arms' park services."""

    def __init__(self) -> None:
        super().__init__("coding_agent_ui")
        self.frames: dict[str, Image] = {}
        for cam in CAMERAS:
            self.create_subscription(Image, f"/cameras/{cam}/color/image_raw",
                                     lambda m, c=cam: self.frames.__setitem__(c, m), 1)
        self._park = {arm: self.create_client(Trigger, f"/{arm}/park") for arm in ARMS}

    def park(self) -> list[str]:
        """Park both arms at once, folding them to all zeros, and wait for both.

        Returns: the arms whose driver did not confirm the park.
        """
        futures = {arm: c.call_async(Trigger.Request()) for arm, c in self._park.items() if c.service_is_ready()}
        deadline = time.monotonic() + PARK_TIMEOUT
        while not all(f.done() for f in futures.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        return [arm for arm in ARMS if arm not in futures or not futures[arm].done()
                or not futures[arm].result().success]

    def jpeg(self, cam: str) -> bytes | None:
        msg = self.frames.get(cam)
        if msg is None:
            return None
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def serve(rig: Rig, chat: Chat, agents: dict, workdir: Path) -> FastAPI:
    """The page and its JSON API over one chat and the agents that write into it."""
    app = FastAPI()
    page = (Path(__file__).parent / "page.html").read_text()
    mode = [next(iter(agents))]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/state")
    def state(since: int = -1) -> JSONResponse:
        with chat.lock:
            body = {"mode": mode[0], "modes": {k: a.label for k, a in agents.items()}, "model": chat.model,
                    "busy": chat.busy, "version": chat.version, "workdir": str(workdir)}
            if since != chat.version:
                body["items"] = chat.items
            return JSONResponse(body)

    @app.post("/api/send")
    def send(body: dict) -> JSONResponse:
        text = str(body.get("text", "")).strip()
        with chat.lock:
            if not text or chat.busy:
                return JSONResponse({"ok": False}, status_code=409)
            chat.busy = True
        agents[mode[0]].send(text)
        return JSONResponse({"ok": True})

    @app.post("/api/mode")
    def select(body: dict) -> JSONResponse:
        if chat.busy or body.get("mode") not in agents:
            return JSONResponse({"ok": False}, status_code=409)
        if body["mode"] != mode[0]:
            mode[0] = body["mode"]
            chat.add("note", text=f"Now talking to {agents[mode[0]].label}")
        return JSONResponse({"ok": True})

    @app.post("/api/stop")
    def stop() -> JSONResponse:
        """Kill both agents and everything they started, then park both arms; sending waits for the park."""
        for agent in agents.values():
            agent.stop()
        with chat.lock:
            chat.busy = True
        chat.add("note", text="Stopped. Parking both arms")
        failed = rig.park()
        chat.idle(f"The {' and '.join(failed)} arm did not park: is its driver running?" if failed
                  else "Both arms parked")
        return JSONResponse({"ok": not failed})

    @app.get("/api/camera/{cam}.jpg")
    def camera(cam: str) -> Response:
        blob = rig.jpeg(cam)
        return Response(blob, media_type="image/jpeg") if blob else Response(status_code=404)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="The coding-agent page.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--workdir", type=Path, help="where the agent keeps its scripts; "
                        "default a new folder under ~/.local/share/coding_agent_sparklab")
    args = parser.parse_args(remove_ros_args()[1:])
    workdir = args.workdir or Path.home() / ".local/share/coding_agent_sparklab" / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)

    # uvicorn owns Ctrl-C, so the agents and their scripts are stopped below.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    rig = Rig()
    executor = SingleThreadedExecutor()
    executor.add_node(rig)
    stop = threading.Event()
    spinner = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    spinner.start()
    chat = Chat()
    agents = {"claude": ClaudeCode(chat, workdir), "antigravity": Antigravity(chat, workdir)}
    print(f"http://localhost:{args.port}  (agent working directory {workdir})")
    try:
        # Localhost only: whoever reaches the page runs code on the rig as you.
        uvicorn.run(serve(rig, chat, agents, workdir), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        for agent in agents.values():
            agent.stop()
        stop.set()
        spinner.join()
        executor.shutdown()
        rig.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
