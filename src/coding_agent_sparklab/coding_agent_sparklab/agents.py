"""Claude Code and Antigravity as subprocesses, each in its CLI's headless stream-json mode.

Both write into one Chat, the transcript the page shows. Each agent may edit
files in its working directory and run the robot's Python interpreter there;
it is denied everything else that needs a permission.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

from . import robot

OUTPUT_CHARS = 5000
"""Tail of each tool result kept for the page."""
SKILLS = Path(__file__).resolve().parent / "saved_skills"
"""The agent's saved skills, in the package source so they are versioned; it may write here."""


def skill_index() -> str:
    """One entry per saved skill, from its SKILL.md front matter, as Claude Code lists skills."""
    entries = []
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        try:
            meta = yaml.safe_load(path.read_text().split("---", 2)[1])
            entries.append(f"- {meta['name']}: {meta['description']}\n  {path}")
        except (IndexError, KeyError, TypeError, yaml.YAMLError):
            entries.append(f"- {path.parent.name}: its SKILL.md has no readable name and description\n  {path}")
    return "\n".join(entries) or "(none yet)"


def instructions() -> str:
    """The agent's brief: prompt.md with the interpreter, the robot API and the calibration filled in."""
    api = "\n\n".join(f"{f.__name__}{inspect.signature(f)}\n{textwrap.indent(inspect.getdoc(f), '    ')}"
                      for f in (robot.get_joint_angles, robot.get_ee_pose, robot.fk, robot.ik, robot.command,
                                robot.get_image, robot.get_depth, robot.get_intrinsics))
    calibration = "\n".join(f"{parent} -> {child}:\n"
                            f"{textwrap.indent(np.array2string(T, precision=4, suppress_small=True), '  ')}"
                            for (parent, child), T in robot.EXTRINSICS.items())
    prompt = (Path(__file__).parent / "prompt.md").read_text()
    return (prompt.replace("{python}", sys.executable).replace("{api}", textwrap.indent(api, "    "))
            .replace("{calibration}", textwrap.indent(calibration, "    "))
            .replace("{skills}", str(SKILLS)).replace("{skill_index}", textwrap.indent(skill_index(), "  ")))


class Chat:
    """The transcript: a list of items, each a dict with a kind and an id.

    Kinds: user (text), text (the agent's markdown), tool (name, input, output,
    error, done) and note (text). An item added with a key is updated in place
    by later adds with the same key.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.items: list[dict[str, Any]] = []
        self._keyed: dict[str, dict[str, Any]] = {}
        self.busy = False
        self.model = ""
        self.version = 0

    def add(self, kind: str, key: str | None = None, **fields: Any) -> None:
        with self.lock:
            item = self._keyed.get(key) if key else None
            if item is None:
                item = {"id": len(self.items), "kind": kind}
                self.items.append(item)
                if key:
                    self._keyed[key] = item
            item.update(fields)
            self.version += 1

    def extend(self, key: str, text: str) -> None:
        """Append streamed text to the text item under key."""
        with self.lock:
            item = self._keyed.get(key)
            if item is None:
                item = self._keyed[key] = {"id": len(self.items), "kind": "text", "text": ""}
                self.items.append(item)
            item["text"] += text
            self.version += 1

    def idle(self, note: str | None = None) -> None:
        if note:
            self.add("note", text=note)
        with self.lock:
            self.busy = False
            self.version += 1


class Agent:
    """One coding-agent CLI, started on the first message and kept for the next.

    chat: the transcript its replies, tool calls and results go to.
    workdir: its working directory, where it keeps its scripts.
    """

    label = ""

    def __init__(self, chat: Chat, workdir: Path) -> None:
        self.chat, self.workdir = chat, workdir
        self.session: str | None = None
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._run = 0
        self._stopping = False

    def argv(self) -> list[str]:
        raise NotImplementedError

    def message(self, text: str) -> dict:
        raise NotImplementedError

    def handle(self, event: dict) -> None:
        raise NotImplementedError

    def send(self, text: str) -> None:
        """Hand the agent one operator message; it works on it in the background."""
        self.chat.add("user", text=text)
        with self.chat.lock:
            self.chat.busy = True
        if self._proc is None or self._proc.poll() is not None:
            self._run += 1
            self._stopping = False
            log = self.workdir / f"{type(self).__name__.lower()}.log"
            try:
                with log.open("a") as stderr:
                    self._proc = subprocess.Popen(self.argv(), cwd=self.workdir, text=True, bufsize=1,
                                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
            except FileNotFoundError:
                self.chat.idle(f"{self.label}: {self.argv()[0]} is not installed")
                return
            self._reader = threading.Thread(target=self._read, args=(self._proc, log), daemon=True)
            self._reader.start()
        # A CLI that dies on start (e.g. not signed in) closes the pipe; _read reports why.
        with contextlib.suppress(BrokenPipeError):
            self._proc.stdin.write(json.dumps(self.message(text)) + "\n")
            self._proc.stdin.flush()

    def stop(self) -> None:
        """Kill the agent and everything it started, its scripts included; the next send resumes.

        Returns once its transcript is closed with a note, so a caller's note lands after it.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        self._stopping = True
        try:
            procs = [psutil.Process(self._proc.pid)]
            procs += procs[0].children(recursive=True)
        except psutil.NoSuchProcess:
            return
        for p in procs:
            with contextlib.suppress(psutil.NoSuchProcess):
                p.terminate()
        for p in psutil.wait_procs(procs, timeout=3)[1]:
            p.kill()
        self._reader.join(timeout=5)

    def _read(self, proc: subprocess.Popen, log: Path) -> None:
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            # One event of an unexpected shape must not end the reader and leave the page busy.
            try:
                self.handle(event)
            except (KeyError, TypeError, AttributeError) as exc:
                self.chat.add("note", text=f"{self.label}: unreadable event, {type(exc).__name__}: {exc}")
        code = proc.wait()
        if self.chat.busy:
            tail = log.read_text(errors="replace").strip().splitlines()[-3:]
            self.chat.idle(f"{self.label} stopped" if self._stopping
                           else f"{self.label} exited with code {code}: {' '.join(tail)}")


class ClaudeCode(Agent):
    label = "Claude Code"

    def argv(self) -> list[str]:
        argv = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--permission-mode", "acceptEdits",
                "--allowedTools", f"Bash({sys.executable}:*)", "--add-dir", str(SKILLS),
                "--append-system-prompt", instructions()]
        return argv + (["--resume", self.session] if self.session else [])

    def message(self, text: str) -> dict:
        return {"type": "user", "message": {"role": "user", "content": text}}

    def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            self.session = event["session_id"]
            self.chat.model = event.get("model", "")
        elif kind == "assistant":
            for block in event["message"]["content"]:
                if block["type"] == "text":
                    self.chat.add("text", text=block["text"])
                elif block["type"] == "tool_use":
                    self.chat.add("tool", key=block["id"], name=block["name"], input=block["input"],
                                  output="", error=False, done=False)
        elif kind == "user" and isinstance(event["message"]["content"], list):
            for block in event["message"]["content"]:
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    if isinstance(content, list):
                        content = "\n".join(c.get("text", "") for c in content)
                    self.chat.add("tool", key=block["tool_use_id"], output=(content or "")[-OUTPUT_CHARS:],
                                  error=bool(block.get("is_error")), done=True)
        elif kind == "result":
            self.chat.idle((event.get("result") or event.get("subtype")) if event.get("is_error") else None)


class Antigravity(Agent):
    label = "Antigravity"

    def argv(self) -> list[str]:
        argv = ["agy", "--input-format", "stream-json", "--output-format", "stream-json", "--mode", "accept-edits",
                "--add-dir", str(SKILLS)]
        return argv + (["--conversation", self.session] if self.session else [])

    def message(self, text: str) -> dict:
        # agy takes no system prompt, so a new conversation opens with the brief.
        if self.session is None:
            text = f"{instructions()}\n\n---\n\n{text}"
        return {"event": "user", "message": {"content": text}}

    def handle(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "init":
            self.session = event["conversation_id"]
            self.chat.model = event["init"].get("model", "")
        elif kind == "step_update":
            step = event["step_update"]
            key = f"agy{self._run}:{step['step_index']}"
            if step.get("step_type") == "agent_response" and step.get("text_delta"):
                self.chat.extend(key, step["text_delta"])
            elif step.get("step_type") == "tool":
                info = step.get("tool_info") or {}
                error = info.get("error")
                self.chat.add("tool", key=key, name=step.get("tool_name") or info.get("name", ""),
                              input=info.get("parameters") or {},
                              output=(info.get("output") or (error or {}).get("message", ""))[-OUTPUT_CHARS:],
                              error=bool(error), done=step.get("state") == "DONE")
        elif kind == "result":
            result = event["result"]
            self.chat.idle(None if result.get("status") == "SUCCESS"
                           else f"{self.label}: {result.get('error') or result.get('status')}")
