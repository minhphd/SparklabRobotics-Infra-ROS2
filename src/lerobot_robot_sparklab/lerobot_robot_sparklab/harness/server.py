"""Web harness: put a high-level agent in front of the low-level VLA.

    ros2 run lerobot_robot_sparklab harness --agent scripted
    ros2 run lerobot_robot_sparklab harness --agent gemini   # HARNESS_MODEL + GOOGLE_API_KEY

then open http://127.0.0.1:8099/ .

This does not own the robot. It serves the UI, runs the agent loop, and drives a
separate ``rollout.live`` process — which owns the arms and the policy — over
its ROS channel (/rollout/*). The agent loop only ever sets a task; the human
keeps pause and reset. See DESIGN.md for the process split.

The built-in loop is a convenience, not the interface — anything that speaks
HTTP can be the planner::

    curl -s :8099/api/state                     # frames available, task, ticks
    curl -s :8099/api/frame/top    -o top.jpg   # what the robot sees
    curl -s :8099/api/task -d '{"task":"pick up the scissors"}'

With --agent none the harness is just that facade, deciding nothing.
"""

# NO `from __future__ import annotations` here: FastAPI resolves route
# annotations in the module namespace, and under PEP 563 every POST 422s.

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .. import ros

logger = logging.getLogger(__name__)

WEB = Path(__file__).parent / "web"


class RolloutLink(Node):
    """A live rollout's /rollout/* channel, cached for the HTTP handlers."""

    def __init__(self) -> None:
        super().__init__("harness")
        self._status: dict | None = None
        self._status_t = 0.0
        self._state: dict[str, float] = {}
        self._state_t = 0.0
        self._frames: dict[str, bytes | None] = {}
        self.create_subscription(String, "/rollout/status", self._on_status, 10)
        self.create_subscription(JointState, "/rollout/state", self._on_state, 10)
        self._task_pub = self.create_publisher(String, "/rollout/task", 10)
        self._triggers = {c: self.create_client(Trigger, f"/rollout/{c}")
                          for c in ("pause", "resume", "reset", "park", "home")}

    def _on_status(self, msg: String) -> None:
        self._status = json.loads(msg.data)
        self._status_t = time.monotonic()
        # The rollout names its cameras once it has published a frame of each.
        for cam in self._status.get("cameras", []):
            if cam not in self._frames:
                self._frames[cam] = None
                self.create_subscription(
                    CompressedImage, f"/rollout/images/{cam}/compressed",
                    lambda m, c=cam: self._frames.__setitem__(c, bytes(m.data)), 1)

    def _on_state(self, msg: JointState) -> None:
        self._state = dict(zip(msg.name, msg.position))
        self._state_t = time.time()

    def status(self) -> dict:
        if self._status is None or time.monotonic() - self._status_t > 2.0:
            return {"error": "no /rollout/status — is the rollout running?"}
        return self._status

    def observation(self) -> dict:
        return {
            "cameras": sorted(c for c, f in self._frames.items() if f is not None),
            "state": dict(self._state),
            "captured_at": self._state_t,
            "age_s": round(time.time() - self._state_t, 3) if self._state_t else None,
        }

    def frame(self, cam: str) -> bytes | None:
        return self._frames.get(cam)

    def command(self, cmd: str, arg: str | None = None) -> dict:
        """Send one command; raises ConnectionError if no rollout is listening."""
        if cmd == "task":
            if not self._task_pub.get_subscription_count():
                raise ConnectionError("nothing subscribes /rollout/task")
            self._task_pub.publish(String(data=arg or ""))
            return {"ok": True, "queued": "task"}
        client = self._triggers[cmd]
        if not client.service_is_ready():
            raise ConnectionError(f"/rollout/{cmd} is not available")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 20.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise ConnectionError(f"/rollout/{cmd} did not answer")
        res = future.result()
        return {"ok": res.success, "queued" if res.success else "error": res.message}


class Harness:
    """Agent loop + a cache of the rollout's state. All HTTP handlers read this."""

    def __init__(self, link: RolloutLink, agent, goal: str = "", interval_s: float = 4.0):
        self.link = link
        self.agent = agent
        self.goal = goal
        self.interval_s = interval_s
        self.running = False
        self.history: list = []
        self.events: list[queue.Queue] = []      # one per connected browser
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        # Bumped on every stop, and compared across a turn by step_once(): a
        # task decided before the operator hit stop must not be applied after.
        self._epoch = 0

    def stop_agent(self) -> None:
        """Stop the loop and invalidate any turn currently in flight."""
        with self._lock:
            self.running = False
            self._epoch += 1

    # ---- events ------------------------------------------------------------
    def emit(self, kind: str, **payload) -> None:
        ev = {"kind": kind, "at": time.time(), **payload}
        with self._lock:
            dead = []
            for q in self.events:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    dead.append(q)          # a browser tab that stopped reading
            for q in dead:
                self.events.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self.events.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self.events:
                self.events.remove(q)

    # ---- agent loop --------------------------------------------------------
    def step_once(self) -> dict:
        """One agent turn. Also the /api/agent {"action":"step"} handler."""
        epoch = self._epoch
        obs = self.link.observation()
        frames = {c: b for c in obs["cameras"] if (b := self.link.frame(c))}
        if not frames:
            d = {"error": "no frames from the rollout — is it running?"}
            self.emit("error", **d)
            return d

        decision = self.agent.decide(self.goal, frames, obs.get("state", {}), self.history)
        self.history.append(decision)
        self.last_error = decision.error

        applied = None
        if decision.task and self._epoch != epoch:
            # Stopped or reset mid-turn: applying the task would drive the arms
            # off the home pose the operator just asked for.
            applied = {"skipped": "agent was stopped during this turn"}
        elif decision.task:
            # The one thing the agent may do. retarget() clears the queued
            # chunk, so it lands next tick instead of after 30 stale actions.
            applied = self.link.command("task", decision.task)
        if decision.done:
            self.stop_agent()
            self.emit("done", reasoning=decision.reasoning)

        self.emit("decision", **decision.as_dict(), applied=applied)
        return {"ok": True, **decision.as_dict(), "applied": applied}

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self.running:
                continue
            try:
                self.step_once()
            except Exception as e:
                logger.exception("agent loop")
                self.last_error = f"{type(e).__name__}: {e}"
                self.emit("error", error=self.last_error)
                # Stopping would leave the robot on its last instruction with
                # nothing watching — worse than a retry the operator can see.

    def start_thread(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="harness-agent")
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()


def build_app(h: Harness):
    from fastapi import FastAPI, Request
    from fastapi.responses import (FileResponse, JSONResponse, Response,
                                   StreamingResponse)

    app = FastAPI(title="SparkLab harness")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/state")
    def state():
        return {
            "rollout": h.link.status(),
            "observation": h.link.observation(),
            "agent": {
                "kind": getattr(h.agent, "name", "none") if h.agent else "none",
                "running": h.running,
                "goal": h.goal,
                "interval_s": h.interval_s,
                "last_error": h.last_error,
                "history": [d.as_dict() for d in h.history[-30:]],
            },
        }

    @app.get("/api/frame/{cam}")
    def frame(cam: str):
        blob = h.link.frame(cam)
        if blob is None:
            return JSONResponse({"error": f"no frame {cam!r}"}, status_code=404)
        return Response(blob, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    def _unreachable(e: Exception) -> JSONResponse:
        """Report a down rollout as such, which is normal during its model load.

        Without this the error escapes as a bare 500 with no JSON body, and the
        UI can only say "HTTP 500" — which reads like a harness bug.
        """
        return JSONResponse({"error": f"cannot reach the rollout: {e}"}, status_code=502)

    @app.post("/api/task")
    async def set_task(req: Request):
        body = await req.json()
        task = (body.get("task") or "").strip()
        if not task:
            return JSONResponse({"error": "task must be non-empty"}, status_code=400)
        try:
            out = h.link.command("task", task)
        except Exception as e:
            return _unreachable(e)
        h.emit("manual", task=task)
        return out

    @app.post("/api/goal")
    async def set_goal(req: Request):
        body = await req.json()
        h.goal = (body.get("goal") or "").strip()
        h.history.clear()
        # Not just the transcript: clearing history alone left a stateful agent
        # mid-plan, reporting the old goal finished. Optional on the agent.
        reset = getattr(h.agent, "reset", None)
        if callable(reset):
            reset()
        h.stop_agent()
        h.last_error = None
        h.emit("goal", goal=h.goal)
        return {"ok": True, "goal": h.goal}

    @app.post("/api/cmd")
    async def cmd(req: Request):
        body = await req.json()
        c = (body.get("cmd") or "").strip().lower()
        # Operator-only; the agent loop never reaches this route. "quit" is
        # excluded on purpose — it owns the arms.
        if c not in {"pause", "resume", "reset", "park", "home"}:
            return JSONResponse({"error": f"{c!r} not allowed here"}, status_code=400)
        if c in {"pause", "reset"}:
            # A reset that leaves the loop running is undone within one interval:
            # the arms ramp home, then the next decision drives them off again.
            h.stop_agent()
        try:
            return h.link.command(c, body.get("arg"))
        except Exception as e:
            return _unreachable(e)

    @app.post("/api/agent")
    async def agent_ctl(req: Request):
        body = await req.json()
        action = (body.get("action") or "").lower()
        if action == "start":
            if not h.agent:
                return JSONResponse({"error": "server started with --agent none"},
                                    status_code=400)
            if not h.goal:
                return JSONResponse({"error": "set a goal first"}, status_code=400)
            h.running = True
            h.emit("agent", running=True)
        elif action == "stop":
            h.stop_agent()
            h.emit("agent", running=False)
        elif action == "step":
            if not h.agent:
                return JSONResponse({"error": "no agent"}, status_code=400)
            return h.step_once()
        else:
            return JSONResponse({"error": "action: start|stop|step"}, status_code=400)
        return {"ok": True, "running": h.running}

    @app.get("/api/events")
    def events():
        def gen():
            q = h.subscribe()
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                h.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--agent", default="scripted", choices=["scripted", "gemini", "none"])
    ap.add_argument("--goal", default="")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between agent turns")
    ap.add_argument("--model", default=None, help="gemini model id (or HARNESS_MODEL)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    from . import agents
    agent = None
    if args.agent != "none":
        kw = {"model": args.model} if args.agent == "gemini" and args.model else {}
        agent = agents.build(args.agent, **kw)

    # uvicorn owns SIGINT; ROS is shut down after it returns.
    ros.init()
    link = RolloutLink()
    close_link = ros.spin_in_background(link)
    h = Harness(link, agent, goal=args.goal, interval_s=args.interval)
    h.start_thread()

    import uvicorn
    logger.info("harness UI on http://%s:%d  (agent %s)", args.host, args.port, args.agent)
    try:
        uvicorn.run(build_app(h), host=args.host, port=args.port, log_level="warning")
    finally:
        close_link()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
