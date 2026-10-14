# scripts

Everyday commands. Run them from the repo root with the `sparklab-ros` env
active and the workspace sourced (`source install/setup.bash`).

| script | does |
|---|---|
| `rollout.sh` | run a policy (real arms or Isaac) |
| `record_manual.sh` | record a dataset, episodes started/ended with Space or B/Y |
| `record.sh` | record a dataset with fixed-length episodes (`lerobot-record`) |
| `harness.sh` | web page + AI agent in front of a running policy |
| `make_certs.sh` | make the HTTPS certificate the headset needs |
| `can_health.sh` | check the CAN bus when an arm stops responding |
| `isaac_python.sh` | run something with Isaac Sim's Python |
| `install_isaac_kernel.sh`, `start_isaac_jupyter.sh` | Isaac notebooks in VS Code |
| `analysis/*.py` | tools for figuring out why a policy misbehaves |

Every workflow below needs the robot started first, in its own terminal:

```bash
ros2 launch sparklab_yam yam.launch.py                     # real arms
ros2 launch sparklab_yam yam.launch.py sim:=true viewer:=true cameras:=false   # no hardware
```

## Teleoperate

```bash
ros2 launch sparklab_yam yam.launch.py teleop:=true
```

Headset controls are in [`sparklab_yam`](../src/sparklab_yam/README.md#the-teleop-node).

## Record a dataset

Start the robot *without* `teleop:=true` (the recorder brings its own), then:

```bash
./scripts/record_manual.sh      # Space / B/Y start+end, r discards, q quits
./scripts/record.sh             # fixed 30 s episodes
```

Change the dataset name and task inside the script, or pass extra
`--dataset.*` flags.

## Run a policy

```bash
./scripts/rollout.sh                          # fine-tuned checkpoint on the real arms
./scripts/rollout.sh --policy=stock-typed     # the stock checkpoint, for comparison
./scripts/rollout.sh --mode=live              # ...and change its task while it runs
./scripts/rollout.sh --mode=dagger            # human takes over with the Quest to correct it
./scripts/rollout.sh --dry-run                # print the command, don't run it
```

Unknown flags pass straight to `lerobot-rollout` (e.g. `--duration=120`). The
script checks that the robot is up before loading the model.

**In Isaac instead of on the real arms:**

```bash
./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics   # terminal 1
./scripts/rollout.sh --robot=sim --policy=stock                                 # terminal 2
```

**Changing the task of a live rollout:**

```bash
ros2 topic pub --once /rollout/task std_msgs/msg/String "{data: 'pick up the scissors'}"
ros2 service call /rollout/reset std_srvs/srv/Trigger    # also: pause, resume, home, park
```

Or with the web page: `./scripts/harness.sh`, then open `http://127.0.0.1:8099/`.

**If the arm moves sluggishly**, look for `loop running slower than target` in
the log and try `--no-foxglove`. Two rollouts at once need different ports:
`FOXGLOVE_PORT=8766 ROS_DOMAIN_ID=1 ./scripts/rollout.sh ...`.

## Figure out why a policy misbehaves

Run these in order; each rules out one layer.

```bash
# 1. Is the data pipeline right? (compares the policy to the recorded human action)
python scripts/analysis/validate_policy_pipeline.py --checkpoint <ckpt>/pretrained_model
# 2. Does it follow the instruction? (--in-scene: only objects that are visible)
python scripts/analysis/lang_probe.py --checkpoint <ckpt>/pretrained_model --in-scene
# 3. Is it jerky?
python scripts/analysis/compare_action_noise.py --checkpoints <ckpt1> <ckpt2> --episodes 0 5 12 --frames 2
# 4. Does it lunge toward the rest pose?
python scripts/analysis/inspect_policy_chunks.py --checkpoint <ckpt>/pretrained_model --episode 5
# 5. Does sim look different enough to matter? (needs the Isaac server)
python scripts/analysis/compare_real_vs_sim_images.py --checkpoint <ckpt>/pretrained_model --port 8081
```

Why is the loop slow? `profile_rollout.py` times each stage of a tick; it takes
the `lerobot-rollout` flags plus `--ticks`.

## Make the headset certificate

```bash
./scripts/make_certs.sh                  # covers every IP of this machine
./scripts/make_certs.sh 10.195.64.139    # or only the ones you name
```

Writes `cert.pem` / `key.pem` in the repo root, where the launch file looks.
The headset must accept a new certificate once, so don't regenerate casually.

## Good to know

- **`--rename_map`**: some checkpoints call the wrist cameras `left`/`right`
  instead of `left_wrist`/`right_wrist`. `rollout.sh` adds the rename only for
  the `--policy` choices that need it.
- **GPU memory**: a rollout (~13 GB), the Isaac server (~4 GB) and an analysis
  script (~13–27 GB) won't all fit in 32 GB at once.
