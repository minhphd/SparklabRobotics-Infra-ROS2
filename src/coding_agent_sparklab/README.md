# coding_agent_sparklab

A web page for directing a coding agent on the bimanual YAM rig: the three
cameras on the left, a chat on the right. The agent is **Claude Code** or
**Antigravity**, picked in the chat box. It gets no skills, only joint-level
Python functions, the camera frames and the rig's calibration, and drives the
arms by writing scripts against them and running them. Unlike
[`capx_sparklab`](../capx_sparklab/README.md), there is no perception, planner or
grasp tool.

```
coding_agent_sparklab/robot.py    the agent's API: joints, fk, ik, command, camera frames, calibration
coding_agent_sparklab/agents.py   Claude Code and Antigravity as subprocesses; Chat, the shared transcript
coding_agent_sparklab/ui.py       the page's server: camera frames from ROS, the chat API; main()
coding_agent_sparklab/page.html   the page
coding_agent_sparklab/prompt.md   the agent's brief; the interpreter, the API's docstrings and the skill index are filled in
coding_agent_sparklab/saved_skills/   skills the agent saved, one folder each, as Claude Code stores skills
test/                             the API against simulated drivers; both CLIs' events against a fake CLI
```

## Running

1. **The rig**, without teleop: `ros2 launch sparklab_yam yam.launch.py`.
2. **The page**, in the workspace's env:

   ```bash
   ros2 run coding_agent_sparklab ui                           # http://localhost:8090
   ros2 run coding_agent_sparklab ui --port 8091 --workdir ~/robot_scripts
   ```

   It listens on localhost only; from a laptop, `ssh -L 8090:localhost:8090 <rig>`.

| flag | default | |
|---|---|---|
| `--port` | `8090` | |
| `--workdir` | a new folder under `~/.local/share/coding_agent_sparklab/` | where the agent writes and runs its scripts; pass an old one to give it its earlier scripts |

To try it on the [MuJoCo sim](https://github.com/minhphd/sparklab_sim) instead,
start `ros2 run sparklab_sim sim` and run the page with `ROS_DOMAIN_ID=42`.

## The page

| part | does |
|---|---|
| cameras | `top`, `left_wrist`, `right_wrist`: the newest frame of `/cameras/<id>/color/image_raw`, refreshed as fast as the page can fetch it |
| chat | your messages, the agent's replies, and each tool call (command, file) with its output; click a call to open it |
| agent menu | Claude Code or Antigravity. Each keeps its own conversation; switching back continues it. Locked while the agent works |
| Stop | always there: kills both agents and everything they started, the script moving an arm included, then parks both arms (all zeros, about 5 s). The chat waits for the park; the next message resumes the conversation |

Enter sends; Shift+Enter is a new line. One message at a time: the box waits
until the agent has finished.

## The agents

Each runs as its CLI's headless stream-json process in `--workdir`, started on
the first message and kept for the next. It may edit files in that folder and in
`saved_skills/`, and run the workspace's Python interpreter; any other command is refused. Its stderr goes
to `claudecode.log` or `antigravity.log` there.

| agent | needs | runs |
|---|---|---|
| Claude Code | `claude` on `PATH`, signed in | `claude -p --input-format stream-json --output-format stream-json --verbose --permission-mode acceptEdits --allowedTools "Bash(<python>:*)" --add-dir <saved_skills> --append-system-prompt <brief>` |
| Antigravity | `agy` on `PATH`, signed in once interactively; a rule letting it run `<python>` | `agy --input-format stream-json --output-format stream-json --mode accept-edits --add-dir <saved_skills>`; the brief opens the conversation |

`agy` has no per-run allow list, so allow the interpreter once in
`~/.gemini/antigravity-cli/settings.json`; without it every script is refused:

```json
{"permissions": {"allow": ["command(/home/<you>/.conda/envs/sparklab-ros/bin/python)"]}}
```

`<python>` is the interpreter the page runs under (`sys.executable`).

## Saved skills

`coding_agent_sparklab/saved_skills/` holds what the agent has learned to do,
laid out the way Claude Code stores skills:

```
<skill-name>/SKILL.md    YAML front matter: name (the folder's name), description (what it does, when to use it);
                         then usage, arguments, outputs, and what was measured on the rig, dated
<skill-name>/scripts/    its code: scripts to run, modules to import
```

| skill | what it is |
|---|---|
| `rig-toolkit` | shared helpers the others import: frames, ik with rotation matrices, guarded and straight-line moves, wrist-camera clouds, localisation, held checks |
| `perceive-table` | top camera: table plane and every object's centre, axis, size and height → `objects.json` |
| `pick` | `fast_pick.py ARM X Y`: look, localise, grasp, lift, check held |
| `place-down` | `place_down.py ARM TABLE_Z`: set the held object down below the gripper and release |

At every start the agent's brief lists each skill's name, description and
`SKILL.md` path, read from the front matter; it opens a `SKILL.md` when it needs
the rest. It is told to look for a saved skill before writing new code, to save
work that succeeded as a new skill in this layout, to improve a skill rather
than copy it, and to keep shared code in `rig-toolkit`. Skill scripts are run
from the session's working directory, so their images and data land there, not
in `saved_skills/`. The folder lives in the package source so it is versioned;
it is found through the symlinked install (`colcon build --symlink-install`).

## The robot API

```python
from coding_agent_sparklab.robot import (get_joint_angles, get_ee_pose, fk, ik, command, get_image,
                                         get_depth, get_intrinsics, EXTRINSICS, FINGERTIPS)
```

| function | returns |
|---|---|
| `get_joint_angles(arm)` | measured joints 1–6 (rad) and the gripper, 1 open .. 0 closed: `(7,)` |
| `get_ee_pose(arm)` | measured tool0 pose: position `(3,)`, quaternion wxyz `(4,)` |
| `fk(joint_angles)` | tool0 pose for joints 1–6 |
| `ik(position, quat_wxyz, seed=None)` | joints 1–6 within the limits, or `ValueError` if none comes within 1 mm and 0.6° |
| `command(arm, joint_angles)` | moves the arm, then the gripper if a 7th value is given; blocks until there; the measured state after |
| `get_image(camera)` | a colour frame received after the call, `(H, W, 3)` uint8 RGB, as the camera sends it |
| `get_depth(camera)` | a depth frame received after the call, `(H, W)` float32 in m along the camera's z axis, aligned pixel for pixel with `get_image`; 0 = no reading |
| `get_intrinsics(camera)` | `K` `(3, 3)` and the distortion coefficients `(5,)`, from its `camera_info` |
| `EXTRINSICS` | `{(parent, child): 4x4}`, the child frame's pose in the parent, loaded from `sparklab_yam/config/extrinsics.yaml` |
| `FINGERTIPS` | 0.075 m: how far past `tool0`, along its z axis, the long fin-ray fingertips end |

- `arm` is `"left"` or `"right"`. Poses are `tool0`, the stock gripper's grasp site, in that
  arm's base frame (`<arm>_base`: x forward, y left, z up); both arms share one model.
- `command` ramps every joint together at ≤ 0.4 rad/s. It refuses targets outside
  the joint limits, and stops and holds the arm if it falls 0.15 rad behind.
  The driver's `max_step` clamp still applies underneath.
- `camera` is `"top"`, `"left_wrist"` or `"right_wrist"`; camera frames are optical
  (z out of the lens, x right, y down). `EXTRINSICS` holds `left_base → right_base`,
  `left_base → top` and `<arm>_tool0 → <arm>_wrist`, where `<arm>_tool0` is `fk`'s tool0.
- The agent looks at a frame by saving it as a PNG in its working directory and
  opening the file.
- Joint states older than 0.5 s count as a dead driver: the functions raise.
- `fk`, `ik` and `EXTRINSICS` need no ROS; the rest connect on first use, about a
  second. A camera's colour, depth and intrinsics are each subscribed only once a script asks for them.

The design reasons are in [`DESIGN.md`](../../DESIGN.md#the-coding-agent).

## Tests

```bash
colcon test --packages-select coding_agent_sparklab && colcon test-result --verbose
```

They start two simulated drivers on ROS domain 94 and feed both CLIs' event
formats through a fake CLI; no agent, account or network is used.
