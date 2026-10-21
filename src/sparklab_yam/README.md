# sparklab_yam

Everything specific to the YAM-Ultra robot: the arm driver, the Quest teleop
node, the inverse kinematics, the arm model, and the launch file that starts
the whole rig.

```
launch/yam.launch.py        starts both drivers, the cameras, the headset bridge, the camera TF, optionally teleop
config/yam.yaml             parameters for the driver, teleop and calibration nodes
config/cameras.yaml         which cameras exist (see config/README.md)
config/extrinsics.yaml      calibrated camera and arm-base poses, published as static TF
sparklab_yam/driver.py      ROS node: one arm
sparklab_yam/teleop_node.py ROS node: Quest controllers drive the arms
sparklab_yam/teleop.py      QuestTeleop: controller pose -> joint targets (used by the node and by LeRobot)
sparklab_yam/keyboard.py    ROS node: drive the arms from the keyboard, one joint at a time
sparklab_yam/calibration.py ROS node: camera-to-arm (hand-eye) calibration, and the board generator
sparklab_yam/ik/            inverse kinematics
sparklab_yam/model/         the arm model (MuJoCo XML, URDF, meshes)
sparklab_yam/sim/           Isaac scene files, read by sparklab_sim
sparklab_yam/run.py         shared main() that lets a node park the arm on Ctrl-C
test/                       driver, teleop, keyboard and calibration tests, no hardware
```

## Starting the rig

```bash
ros2 launch sparklab_yam yam.launch.py                    # real arms
ros2 launch sparklab_yam yam.launch.py teleop:=true       # + Quest teleop
ros2 launch sparklab_yam yam.launch.py sim:=true viewer:=true cameras:=false
```

| argument | default | |
|---|---|---|
| `sim` | `false` | simulated arms instead of CAN |
| `viewer` | `false` | a MuJoCo window per simulated arm |
| `cameras` | `true` | start the camera node |
| `teleop` | `false` | start the teleop node |
| `certfile`, `keyfile` | `./cert.pem`, `./key.pem` | HTTPS cert for the headset page |

## The driver

One process per arm, in a `/left` or `/right` namespace. It is the only thing
that talks to the motors.

| topic | direction | |
|---|---|---|
| `joint_states` | out | joints 1–6 + gripper: position, velocity, effort; 100 Hz |
| `ee_pose` | out | gripper pose (`tool0`) in the arm base frame |
| `/tf` | out | the same pose as the transform `<arm>_base → <arm>_tool0` |
| `arm/command` | in | 6 joint targets, rad |
| `gripper/command` | in | gripper target, 0 closed .. 1 open |
| `park` | service | slowly move every joint to 0 (the folded rest pose) |
| `zero_gravity` | service (`SetBool`) | `true`: arm joints float (damped), the gripper keeps holding, commands ignored; `false`: hold where it is |

Things to know:

- Each command is **clamped** so no joint moves more than `max_step` (default
  0.133 rad for joints 1–3, 0.15 for 4–6) away from where it is now.
- If the motors stop responding, `joint_states` **stops**. Silence means fault.
- On Ctrl-C it **parks first**, then cuts torque.
- At startup the gripper homes itself — keep fingers clear.

Parameters (in `config/yam.yaml`): `channel`, `sim`, `rate_hz`, `max_step`,
`park_duration_s`, `park_on_exit`, `enable_auto_recovery`, `viewer`,
`ee_mass`, `gravity_comp_factor`, `zero_gravity_comp_factor`.

**Tuning gravity compensation.** All three settings are in `config/yam.yaml`:

| parameter | what it is |
|---|---|
| `ee_mass` | the gripper's mass in kg, as i2rt's gravity model sees it; 0 keeps the stock LINEAR_4310's 0.553, which the rig uses. Fix this first if both modes are off, or after changing the gripper. |
| `gravity_comp_factor` | 6 numbers, joints 1–6: scale on the modelled gravity torque while **holding position** |
| `zero_gravity_comp_factor` | the same while **floating** in zero gravity; 1.0 holds exactly the arm's weight |

In zero gravity: if the arm **sinks**, raise `zero_gravity_comp_factor`; if it
**drifts up**, lower it — in steps of about 0.05, mostly on joints 2, 3 and 5,
which carry the load. To tune one arm, add the line under `/left/yam_driver`
or `/right/yam_driver` instead. Restart the launch to apply.

## The teleop node

Reads the Quest topics and publishes arm and gripper commands. Its settings
(`hands`, speeds, scales) are in `config/yam.yaml`; the headset page's Settings
sliders can change the scales live.

**Headset controls:**

| control | does |
|---|---|
| **B/Y** | arm / disarm. Starts disarmed; disarming moves the arms back home |
| **Grip** (hold) | clutch: the arm follows your hand only while held |
| **Trigger** | close the gripper |
| **A/X** (hold) | precision mode: slower, finer motion |
| **Thumbstick click** | send that arm home |

"Home" is wherever each arm was when the node started. Ctrl-C also sends the
arms home first.

Don't run the teleop node while recording or running a policy — those bring
their own teleoperator.

## The keyboard node

The simplest way to move an arm, like turtlesim's `teleop_key`. Start the
robot without teleop, then in its own terminal:

```bash
ros2 run sparklab_yam keyboard                     # 0.02 rad per press
ros2 run sparklab_yam keyboard --ros-args -p step:=0.05 -p arm:=right
```

| keys | do |
|---|---|
| `q w e r t y` | joint 1–6 **+** |
| `a s d f g h` | joint 1–6 **−** |
| `z` / `x` | open / close the gripper |
| space | switch between the left and right arm |
| Ctrl-C | quit — the arm holds where it is |

Hold a key to keep moving. Each press publishes the whole arm's target on
`/<arm>/arm/command`, so the driver's `max_step` clamp still applies, and
targets stop at the joint limits. If something else moved the arm, the next
press after a pause starts from where it now is.

## Camera calibration

`calibration.py` finds where a camera is relative to an arm with a ChArUco
board: `eye_in_hand` for a wrist camera (pose in `<arm>_tool0`), `eye_to_hand`
for the top camera (pose in `<arm>_base`). It switches the arm to zero gravity
through the driver's `zero_gravity` service, shows the detection on
`/calibration/image`, and saves `calibration_<camera>.yaml`. The step-by-step
guide is in the [top-level README](../../README.md#calibrating-a-camera); the
board size is in `config/yam.yaml`. Results go into `config/extrinsics.yaml`.

## QuestTeleop

The core of teleoperation, shared by the teleop node and LeRobot's
`bi_quest_teleop`. Per arm it keeps a *clutch* (see
[`sparklab_quest`](../sparklab_quest/README.md#clutch-pose-mapping)) and an IK
solver. Each `step()` returns joint targets in LeRobot's format
(`left_joint_1.pos` … `right_gripper.pos`) and vibrates the controller when the
arm is near a limit. Call `seed(obs)` whenever something else has moved the
arms, so teleop resumes from where they actually are.

## More

- [`ik/`](sparklab_yam/ik/README.md) — the inverse kinematics
- [`model/`](sparklab_yam/model/README.md) — the arm model and where it came from
- [`config/`](config/README.md) — camera serials and settings
