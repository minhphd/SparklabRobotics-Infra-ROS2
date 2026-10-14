# sparklab_sim

An Isaac Sim copy of the rig, for testing policies without hardware. It **poses**
the arms and **renders** the cameras; it does not simulate forces, so it can't
tell you whether a grasp would really succeed.

It runs on Isaac's own Python, which has neither ROS nor LeRobot — so it is not a
ROS package (note the `COLCON_IGNORE`), and nothing here may import either. It
reads the arm model and camera config straight from `sparklab_yam`'s folder
(`paths.py`).

```
policy_server.py  HTTP server that LeRobot's yam_ultra_sim robot talks to
scene.py          build the scene, pose the arms, place cameras
rig.py            table and camera layout (each value marked MEASURED or ASSUMED)
rigcams.py        sim cameras matched to the real ones
render.py         camera -> image array, image diff
convert.py        URDF -> USD, run once
live.py           edit the scene file and watch it in a browser
viewport.py       the browser view used by live.py
capture.py        on-demand capture server
paths.py          where the shared model and config files are
```

## Running

Don't activate a conda env; the wrapper uses Isaac's Python:

```bash
./scripts/isaac_python.sh -m sparklab_sim.convert                         # once
./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics
./scripts/rollout.sh --robot=sim                                          # then, elsewhere
```

Use `--physics` whenever objects are touched — without it the gripper can't
hold anything. Leave the server running between rollouts (Isaac takes ~15 s to
start); it parks the arms by itself after 45 s idle.

## Editing the scene

```bash
./scripts/isaac_python.sh -m sparklab_sim.live      # open http://127.0.0.1:8080/
```

Drag to orbit, scroll to zoom, shift-drag to pan. Edit
`sparklab_yam/sparklab_yam/sim/scene.usda` in any editor and save — the view
updates in about 0.3 s. `copy pose` copies the current view as code for `rig.py`.
Useful flags: `--once --out x.png` (one snapshot), `--regenerate` (rebuild from
`rig.py`, **discarding your edits**). Over VS Code Remote-SSH the port is
forwarded automatically.

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
- Gravity is off on purpose: the URDF has no joint gains, so arms would sag.
- Two renders of the same scene differ by about 0.8 (mean abs. error, 0–255);
  that's the noise floor.
- `convert.py` writes `sparklab_yam/sparklab_yam/model/usd/` (gitignored).

## Still missing

Wrist camera positions are guesses — there's no hand-eye calibration yet, so
rendered wrist views won't match recorded ones exactly.
