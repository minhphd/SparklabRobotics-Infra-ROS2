# capx_sparklab

cap-x's SparkLab tool surface on the bimanual YAM rig: an LLM writes Python
against typed tools (perceive, plan a grasp, move, hand over), and the tools run
on both arms through the ROS 2 stack. cap-x itself stays in its own checkout,
`../cap-x` next to this repo with its venv in `.venv` (`CAPX=/path` to use
another); this package is the robot side.

```
capx_sparklab/env.py              YamEnv: cap-x's low-level env over ROS 2 (joints, cameras, TF, commands)
capx_sparklab/scene.py            RosScene: both arms and what the tools draw, as topics for Foxglove
capx_sparklab/api.py              YamSparkLabApi: SparkLabApi on two arms, plus handover; YamCodeEnv
capx_sparklab/agent.py            cap-x's ReAct loop and dashboard with this rig's defaults
capx_sparklab/ui.py               cap-x's browser UI with this rig's camera and defaults
capx_sparklab/robot/__init__.py   gripper constants; curobo_config() places one arm in the world
capx_sparklab/robot/build.py      regenerates everything below from sparklab_yam's model
capx_sparklab/robot/yam.urdf          the arm with i2rt's stock gripper, rooted at world
capx_sparklab/robot/curobo_yam.yml    cuRobo robot config: collision spheres, self-collision matrix
capx_sparklab/robot/graspgenx/        GraspGenX gripper description (sparklab_yam)
capx_sparklab/configs/yam_sparklab.yaml   the cap-x session config and the agent's prompt
test/                             model, env and two-arm tool tests (GPU, simulated arms)
```

Everything here runs under [`scripts/capx_python.sh`](../../scripts/capx_python.sh),
which gives one Python both cap-x's GPU stack (from the cap-x checkout's venv)
and this workspace's ROS 2.

## Running a session

1. **The rig**, without teleop: `ros2 launch sparklab_yam yam.launch.py`. It also
   publishes the camera calibration on TF, which this package needs.
2. **The perception servers**, each in its own terminal (env active):

   ```bash
   CAPX=../cap-x G=../cap-x/capx/third_party/GraspGenX
   ./scripts/capx_python.sh $CAPX/capx/serving/launch_sam3_server.py --port 8114
   ./scripts/capx_python.sh $G/client-server/graspgenx_server.py --port 5556 \
       --config $G/ext/graspgenx_checkpoints/GraspGenXModel/release/gen/config.yaml \
       --assets_dir $G/ext/gripper_descriptions/gripper_descriptions/assets
   ```

   SAM3's weights come from Hugging Face, where they are gated: request access at
   the SAM3 repo, then `huggingface-cli login` once. The config starts SAM3 itself
   if nothing listens on 8114. Check both: `ss -ltn | grep -E ':8114|:5556'`.
3. **Gemini** is an API, not a server: cap-x sends `gemini:` models (the agent's,
   and perception's pointing) to Vertex AI with a service-account file, read from
   `GEMINI_KEY_FILE`. `capx_python.sh` points it at `~/.config/capx/gemini_key.json`
   when that exists and nothing else is set; keep the file mode 600, outside the repo.
   OpenAI models read `OPENAI_API_KEY`, from the environment or cap-x's `.env`.
4. **The agent**, cap-x's ReAct loop: one tool call per step, each result fed back.

```bash
./scripts/capx_python.sh -m capx_sparklab.agent --model gemini:gemini-3.8-flash
```

| page | shows |
|---|---|
| http://localhost:8090 | the dashboard: camera frames with what perception pointed at, the agent's notebook, the step log, and where you type instructions |
| http://localhost:8081 | the point cloud from the depth cameras, every perceived object's points and box (one per perceive prompt), and the last grasp candidates, the planner's first choice bright (embedded in the dashboard; if 8081 is taken, the next free port) |

**In Foxglove**, while a session runs: in the 3D panel (display frame `left_base`)
add two *Custom layers → URDF*, source *Topic*, `/left/robot_description` and
`/right/robot_description` — both arms, grippers and all, follow their joints.
Then turn on:

| topic | shows |
|---|---|
| `/capx/perceived` | the last perceived object's points |
| `/capx/perceived_box` | its bounding box |
| `/capx/grasps` | grasp candidates as gripper outlines, the one the planner prefers bright |
| `/capx/collision` | the planner's world, when the API is built with `show_collision=True` |
| `/capx/<camera>/highlight` | (Image panel) the frame perception used, with the mask and the point it grounded on |

`--cameras` picks what the model sees each step: `top` (the default), `left_wrist`,
`right_wrist`. The robot speaks its `Say:` lines through cap-x's voice when a TTS
model and an audio device are available, and prints them otherwise; `--silent`
turns it off. cap-x's other front-ends also work:
`-m capx_sparklab.ui` for its browser page (the camera panel needs this wrapper;
cap-x's own `flask_ui.py` looks for LIBERO's `agentview` camera), or
`$CAPX/capx/agent/simple_loop.py --config-path <config> --cameras top` in the terminal.

Stop with Ctrl-C. Whenever the process exits, both arms park: every joint,
gripper included, ramps to zero. Anything still held is folded in with the arm.

## The tools

cap-x's 13 SparkLab tools (their docstrings are what the model reads), with these
differences:

| tool | on this rig |
|---|---|
| every tool that moves an arm or its gripper | needs `arm_id`, `"left"` or `"right"`; there is no default arm |
| `get_state(arm_id=None)` | reports both arms when no arm is named |
| `close_gripper` | returns `{final_width, effort}`, no `grasped`: the agent confirms a grasp with `view_cameras`. See [`DESIGN.md`](../../DESIGN.md#cap-x-on-the-rig) |
| `update_collision_world` | attaching takes `arm_id` too: `attach_cloud_id=..., arm_id=...` |
| `handover(from_arm, to_arm, height=0.30)` | new: passes a held, attached object to the other gripper. The giver holds it level 0.30 m in front of the bases' midpoint, fingers turned 60° toward the receiver, which grasps the part sticking out ≥ 6 cm from the giver's fingers. The receiving grasp is not checked |
| `move_relative` | as cap-x's, with a bug fixed: cap-x's copy raises after every successful move |
| `lift(arm_id)` | new: backs out of the grasp `go_to_pose(grasp_ids=...)` just reached, along the 10 cm lift planned with it. After a grasp low on the table it is the only motion that plans. See [`DESIGN.md`](../../DESIGN.md#cap-x-on-the-rig) |

`view_cameras`, `perceive`, `deproject` and `remember` are arm-independent; their
clouds and grasps are shared by both arms.

## Frames and the gripper

| frame | where |
|---|---|
| world | `left_base`, the root of the rig's TF: x forward, y left, z up |
| `grasp_base` | the planners' tool frame: the gripper's base, z out along the fingers; `ee_pose` in the tools |
| grasp centre | 0.1347 m along `grasp_base` z (`TOOL0_DEPTH`, the driver's `tool0`); the fingertips end ~0.145 m out |

The rig has i2rt's stock LINEAR_4310 gripper; its meshes come from `sparklab_yam`'s model.
`TOOL0_DEPTH`, `FINGERTIP_DEPTH` and `GRIPPER_OPENING` in
[`robot/__init__.py`](capx_sparklab/robot/__init__.py) describe it. After changing any of
them, or the gripper, rebuild the robot description and rerun the tests:

```bash
./scripts/capx_python.sh -m capx_sparklab.robot.build
```

The build prints any link pairs it leaves out of self-collision because they
touch in the parked pose.

## What stops a motion

Every motion is planned by cuRobo against the fused camera depth and the table;
a motion that does not plan is an error and nothing moves. While a planned
trajectory streams to a driver, `YamEnv` stops and holds the arm if it falls
0.15 rad behind — blocked, in zero gravity, or its driver gone — and the tool
returns the error. The drivers' own step clamp still applies underneath.

## Tests

```bash
./scripts/capx_python.sh -m pytest src/capx_sparklab/test
```

They start their own simulated drivers on ROS domains 88–90, fake the cameras,
and plan on the GPU with small batches; perception and GraspGenX are stubbed.
`test_config.py` builds a whole session from the shipped config and runs agent
code through it. Without cap-x's stack (plain `colcon test`), only
`test_robot_files.py` runs.
