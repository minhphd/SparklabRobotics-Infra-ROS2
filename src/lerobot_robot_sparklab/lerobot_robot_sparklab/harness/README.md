# harness

A web page where an AI agent (or you) decides *what* the policy should do next.
The policy only understands one short instruction at a time, so the agent
breaks a goal into steps:

```
goal "clear the table"
  -> agent looks at the cameras -> "pick up the marker"  -> policy -> arms
                                -> "put it in the box"   -> policy -> arms
                                -> done
```

```
server.py   web server + ROS node: shows the cameras, runs the agent, talks to the rollout
agents.py   the agents: ScriptedAgent (fixed list, no API key) and GeminiAgent
web/        the page
```

## Running it

```bash
ros2 launch sparklab_yam yam.launch.py   # terminal 1: the robot
./scripts/rollout.sh --mode=live         # terminal 2: the policy
./scripts/harness.sh                     # terminal 3: open http://127.0.0.1:8099/
```

The harness is a separate process from the rollout, so you can restart it
freely without reloading the model.

- `--agent scripted` (default) — steps through a fixed list. Try this first: if
  the arm misbehaves here, the problem isn't the AI.
- `--agent gemini` — needs `pip install google-genai`, `GOOGLE_API_KEY` and
  `HARNESS_MODEL=<model id>`.
- `--agent none` — just the page and the API below; bring your own agent.

## The page

| button | does |
|---|---|
| **Send task to VLA** | give the policy this task now |
| **Send task to reasoning agent** | set a goal and let the agent work through it |
| **Reset & park home** | stop the agent, send the arms home, wait for a new task |

## The API

```bash
curl -s :8099/api/state                                   # task, status, cameras, joints
curl -s :8099/api/frame/top -o top.jpg                    # latest camera image
curl -s :8099/api/task -d '{"task":"pick up the scissors"}'
curl -s :8099/api/cmd  -d '{"cmd":"reset"}'               # pause | resume | reset | park | home
curl -s :8099/api/agent -d '{"action":"start"}'           # start | stop | step
```

The agent itself can only do two things: set a task, and say the goal is done.
Stopping, resetting and parking are left to the human.
