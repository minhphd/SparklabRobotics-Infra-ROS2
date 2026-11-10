"""Both CLIs' stream-json events into the chat, and the subprocess plumbing against a fake CLI."""

import sys
import time

import psutil

from coding_agent_sparklab import agents
from coding_agent_sparklab.agents import Antigravity, Chat, ClaudeCode

# Claude Code's stream-json, as seen from claude 2.1: echoes a message, or runs `sleep 60` first on "hang".
FAKE_CLAUDE = r"""
import json, subprocess, sys
for line in sys.stdin:
    text = json.loads(line)["message"]["content"]
    if text == "hang":
        child = subprocess.Popen(["sleep", "60"])
        print(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "sleep 60"}}]}}), flush=True)
        child.wait()
    reply = {"type": "assistant", "message": {"content": [{"type": "text", "text": "echo " + text}]}}
    print(json.dumps(reply), flush=True)
    print(json.dumps({"type": "result", "is_error": False}), flush=True)
"""


class FakeClaude(ClaudeCode):
    def argv(self):
        return [sys.executable, "-c", FAKE_CLAUDE]


class Missing(ClaudeCode):
    def argv(self):
        return ["no-such-coding-agent"]


def _wait(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_claude_events_become_text_and_tool_items(tmp_path):
    chat = Chat()
    agent = ClaudeCode(chat, tmp_path)
    chat.busy = True
    for event in [
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "claude-opus-5-5"},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Moving the left arm."},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "python move.py"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "Traceback"}],
             "is_error": True}]}},
        {"type": "result", "is_error": False},
    ]:
        agent.handle(event)
    assert agent.session == "s1" and chat.model == "claude-opus-5-5" and not chat.busy
    assert [i["kind"] for i in chat.items] == ["text", "tool"]
    assert chat.items[1] == {"id": 1, "kind": "tool", "name": "Bash", "input": {"command": "python move.py"},
                             "output": "Traceback", "error": True, "done": True}
    assert "--resume" in agent.argv() and "s1" in agent.argv()


def test_antigravity_streams_text_and_updates_a_tool_step_in_place(tmp_path):
    chat = Chat()
    agent = Antigravity(chat, tmp_path)
    assert agent.message("hi")["message"]["content"].endswith("---\n\nhi")
    chat.busy = True
    step = {"conversation_id": "c1", "step_index": 1, "state": "ACTIVE", "step_type": "agent_response"}
    tool = {"conversation_id": "c1", "step_index": 2, "step_type": "tool", "tool_name": "run_command",
            "tool_info": {"name": "run_command", "parameters": {"CommandLine": "python move.py"}}}
    for event in [
        {"event": "init", "conversation_id": "c1", "init": {"cwd": str(tmp_path), "tools": []}},
        {"event": "step_update", "step_update": step | {"text_delta": "Moving "}},
        {"event": "step_update", "step_update": step | {"state": "DONE", "text_delta": "now.\n"}},
        {"event": "step_update", "step_update": tool | {"state": "ACTIVE"}},
        {"event": "step_update", "step_update": tool | {"state": "DONE", "tool_info": tool["tool_info"] | {
            "output": "", "error": {"type": "exit", "message": "no joint states"}}}},
        {"event": "result", "result": {"conversation_id": "c1", "status": "ERROR", "error": "quota"}},
    ]:
        agent.handle(event)
    assert agent.session == "c1" and not chat.busy
    assert agent.message("again")["message"]["content"] == "again"
    assert [i["kind"] for i in chat.items] == ["text", "tool", "note"]
    assert chat.items[0]["text"] == "Moving now.\n"
    assert chat.items[1]["done"] and chat.items[1]["error"] and chat.items[1]["output"] == "no joint states"
    assert chat.items[2]["text"] == "Antigravity: quota"
    assert agent.argv()[-2:] == ["--conversation", "c1"]


def test_stop_kills_the_agent_and_what_it_started_then_the_next_message_restarts_it(tmp_path):
    chat = Chat()
    agent = FakeClaude(chat, tmp_path)
    agent.send("hi")
    _wait(lambda: not chat.busy)
    assert chat.items[-1]["text"] == "echo hi"
    agent.send("hang")
    _wait(lambda: any(i["kind"] == "tool" for i in chat.items))
    sleeps = [p for p in psutil.Process(agent._proc.pid).children(recursive=True) if p.name() == "sleep"]
    assert sleeps
    agent.stop()
    _wait(lambda: not chat.busy)
    assert chat.items[-1] == {"id": len(chat.items) - 1, "kind": "note", "text": "Claude Code stopped"}
    assert not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in sleeps)
    agent.send("again")
    _wait(lambda: not chat.busy)
    assert chat.items[-1]["text"] == "echo again"
    agent.stop()


def test_a_missing_cli_says_so(tmp_path):
    chat = Chat()
    Missing(chat, tmp_path).send("hi")
    assert not chat.busy
    assert chat.items[-1]["text"] == "Claude Code: no-such-coding-agent is not installed"


def test_the_skill_index_reads_front_matter_and_both_agents_may_write_the_skills(tmp_path, monkeypatch):
    (tmp_path / "pick").mkdir()
    (tmp_path / "pick" / "SKILL.md").write_text("---\nname: pick\ndescription: Picks things up.\n---\n# pick\n")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("no front matter here\n")
    monkeypatch.setattr(agents, "SKILLS", tmp_path)
    index = agents.skill_index()
    assert "- pick: Picks things up." in index and str(tmp_path / "pick" / "SKILL.md") in index
    assert "- broken: its SKILL.md has no readable name and description" in index
    assert str(tmp_path) in agents.instructions()
    for agent in (ClaudeCode(Chat(), tmp_path), Antigravity(Chat(), tmp_path)):
        argv = agent.argv()
        assert argv[argv.index("--add-dir") + 1] == str(tmp_path)
