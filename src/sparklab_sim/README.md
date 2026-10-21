# sparklab_sim

Isaac Sim copies of the rig. Two entry points:

- **`twin.py`, the digital twin.** The calibrated rig with physics, behind the real
  rig's ROS 2 interface. Anything that drives the rig (cap-x, VLM_TAMP) drives the
  twin unchanged, on another ROS domain.
- **`policy_server.py`.** An HTTP server for LeRobot's `yam_ultra_sim` robot. It poses
  the arms in the older hand-authored `scene.usda`.

Both run on Isaac's own Python (Isaac Sim 5.1 in the `behavior` conda env), so this is
not a ROS package (note the `COLCON_IGNORE`). It reads the arm model and rig config
straight from `sparklab_yam`'s and `capx_sparklab`'s folders (`paths.py`).

```
twin.py           the digital twin: build the stage, serve the rig's ROS 2 interface
twin_scene.py     the twin's stage: arms at their calibrated bases, cameras, props; FK
looks.py          the twin's colours, light, room and cardboard box
calibration.py    the rig as calibrated: extrinsics.yaml, cameras.yaml, intrinsics.yaml
objects.py        BEHAVIOR-1K objects, decrypted once into ~/.cache/sparklab_sim/objects
convert.py        URDF -> USD, run once (--twin: the twin's arm, with collisions)
policy_server.py  HTTP server that LeRobot's yam_ultra_sim robot talks to
scene.py          policy server's scene: pose the arms, place cameras
rig.py            policy server's table and camera layout (MEASURED or ASSUMED)
rigcams.py        policy server's cameras
render.py         camera -> image array, image diff
live.py           edit the policy server's scene file and watch it in a browser
viewport.py       the browser view used by live.py
capture.py        on-demand capture server
paths.py          where the shared model, config and object files are
test/             the twin's geometry against the driver's; runs without Isaac
```

## The digital twin

Don't activate a conda env; the wrapper sets up Isaac's Python and its bundled ROS 2:

```bash
./scripts/isaac_python.sh -m sparklab_sim.convert --twin    # once, and after capx_sparklab.robot.build
./scripts/isaac_python.sh -m sparklab_sim.twin              # serves on ROS domain 42
ROS_DOMAIN_ID=42 ./scripts/capx_python.sh -m capx_sparklab.agent
```

Flags: `--domain N` serves another domain; `--gui` opens Isaac's window. Startup takes
under a minute (longer the first time, while shaders compile).

It serves what the two drivers, the camera node and the launch's static TF serve,
with the same names, frames and units:

| topic / service | as on the rig |
|---|---|
| `/<arm>/joint_states` | `joint_1`..`joint_6` + `gripper` (1 open .. 0 closed), 120 Hz |
| `/<arm>/ee_pose`, `/tf` `<arm>_base` → `<arm>_tool0` | the driver's FK of the joint states |
| `/<arm>/arm/command`, `/<arm>/gripper/command` | clamped to `max_step` from `yam.yaml` |
| `/<arm>/park` | ramps to zero over `park_duration_s`; answers at once rather than after |
| `/<arm>/zero_gravity` | refused, as a SimRobot does |
| `/cameras/<id>/{color,depth}/image_raw` | rgb8, and 16UC1 mm aligned to colour, one stamp, 8 Hz |
| `/cameras/<id>/{color,depth}/camera_info` | the real intrinsics, distortion zero |
| `/tf_static` | `extrinsics.yaml` |

Beyond the rig, for scoring practice runs only: `/tf` `left_base` → `twin/<prop>`, each
prop's true pose. Nothing that drives the robot may read it.

The world is `left_base`. The table top is 1.5 cm below the bases. The arms are
`capx_sparklab`'s robot model, the one cuRobo plans with: i2rt's stock gripper. Arm
links feel no gravity, as under the drivers' gravity compensation; the joints use the
real motors' PD gains, and the fingers keep a held object centred, as the gripper's rack does.

**Props.** `twin_scene.DEFAULT_PROPS` lists five BEHAVIOR-1K objects within top-down
reach (a marker, a soda can, a bowl, an apple, a mug), plus the open cardboard box and
the back rail from `looks.py`. A `Prop` names a BEHAVIOR-1K category and an (x, y) in
`left_base`; it is dropped onto the table and settles. The dataset lives in
`~/BEHAVIOR-1K` (override with `BEHAVIOR_1K=`).

## The policy server

```bash
./scripts/isaac_python.sh -m sparklab_sim.convert                         # once
./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics
./scripts/rollout.sh --robot=sim                                          # then, elsewhere
```

Use `--physics` whenever objects are touched; without it the gripper can't hold
anything. Leave the server running between rollouts; it parks the arms by itself
after 45 s idle. It was written against Isaac 6.0 and has not been rerun on 5.1.

## Editing the policy server's scene

```bash
./scripts/isaac_python.sh -m sparklab_sim.live      # open http://127.0.0.1:8080/
```

Drag to orbit, scroll to zoom, shift-drag to pan. Edit
`sparklab_yam/sparklab_yam/sim/scene.usda` in any editor and save; the view
updates in about 0.3 s. `copy pose` copies the current view as code for `rig.py`.
Useful flags: `--once --out x.png` (one snapshot), `--regenerate` (rebuild from
`rig.py`, **discarding your edits**).

## Notebooks

```bash
./scripts/install_isaac_kernel.sh      # once
```

Then in VS Code pick **Select Kernel → Jupyter Kernel → SparkLab Isaac Sim**.
If it doesn't appear, run `./scripts/start_isaac_jupyter.sh` and connect to the
printed URL. The kernel holds the GPU while it's alive.

## Gotchas

- Create `SimulationApp` before importing anything from `omni.*` / `isaacsim.*`.
- Call `Renderer.close()` before opening another stage, or Isaac crashes.
- Kit drops unflushed stdout when it closes: `print(..., flush=True)`.
- The policy server keeps gravity off on purpose: its URDF has no joint gains, so arms would sag.
- Two renders of the same scene differ by about 0.8 (mean abs. error, 0–255);
  that's the noise floor.
- In the URDF a finger joint at 0 is **closed**; open is −0.04695.
- `convert.py` writes `sparklab_yam/sparklab_yam/model/usd/` (gitignored).
