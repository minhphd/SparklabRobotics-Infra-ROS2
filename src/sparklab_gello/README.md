# sparklab_gello

One GELLO (YAM Active Leader) driving one YAM arm, joint to joint, through the
YAM driver's command topics. Never writes the leader's EEPROM; its torque is on
only while homing at start-up, while it follows the arm disarmed, while it
folds at shutdown, and while `hold` locks it.

```
sparklab_gello/reader.py   read, validate and convert one leader reading to YAM targets
sparklab_gello/guard.py    DISARMED / ARMED / FAULT latch with time-based rate limits
sparklab_gello/node.py     the ROS 2 node: reader -> guard -> command topics
sparklab_gello/pair.py     /gello/arm and /gello/disarm for both of the launch's GELLO nodes at once
config/gello_to_yam.yaml   GELLO -> YAM joint sign, zero and lever end stops (provisional)
config/yam_active_leader_eeprom.json  the leader's EEPROM calibration, read back
config/*_right.*           the same for the right leader
launch/teleop.launch.py    both drivers and both GELLO nodes
test/                      reader, guard and node tests; no hardware needed
```

Needs `lerobot_teleoperator_yamactiveleader` (pip) in the ROS env.

## Running

Start the robot without `teleop:=true`, then:

```bash
ros2 run sparklab_gello gello --ros-args -p arm:=left     # or arm:=right
ros2 service call /gello/arm std_srvs/srv/Trigger         # start following
ros2 service call /gello/disarm std_srvs/srv/Trigger      # stop following
```

Both arms and both of the lab's leaders, drivers included, from one launch:

```bash
ros2 launch sparklab_gello teleop.launch.py                          # real arms
ros2 launch sparklab_gello teleop.launch.py sim:=true viewer:=true   # simulated arms
ros2 service call /gello/arm std_srvs/srv/Trigger                    # both arms follow, or neither
ros2 service call /gello/disarm std_srvs/srv/Trigger                 # both stop
```

`/gello/arm` arms both arms or neither: if one is refused, both are disarmed
again, and the reply says why for each. One arm alone: `/gello/left/arm`,
`/gello/right/arm` (and `.../disarm`). It picks each leader by USB serial (left `5B8E115998`, right `5B79018636`), sets j5/j6
`max_step` to 0.3 rad for its drivers ([`DESIGN.md`](../../DESIGN.md#wrist-max_step-in-teleoplaunchpy))
and, in sim, `state_timeout_s` to 0.5 s. Ctrl-C parks both arms.

On start-up the node drives the leader to the pose that maps to the arm's measured pose and the
lever to the arm's measured gripper, at ~26 deg/s and half torque, then releases them: keep hands off it until
`leader homed and released` is logged. `~/arm` then passes without posing by hand. The arm's driver must be running first; the node homes only once the arm has held still for 1 s, after the driver's startup ramp to the rest pose, and gives up after 30 s. `-p home:=false` skips this. See
[`DESIGN.md`](../../DESIGN.md#homing-the-leader).

Before `~/arm`, pose the leader like the arm: the request is refused, naming
the joint, while any joint is more than 0.15 rad off.

### In simulation on macOS

Every terminal:

```bash
conda activate sparklab-ros && source install/setup.zsh
export ROS_LOCALHOST_ONLY=1    # without it, nodes in different processes do not find each other
```

```bash
# terminal 1: one simulated arm with a MuJoCo window (launch_passive needs mjpython on macOS)
mjpython -m sparklab_yam.driver --ros-args -r __ns:=/left -r __node:=yam_driver -p sim:=true -p viewer:=true
# terminal 2
ros2 run sparklab_gello gello --ros-args -p arm:=left
# terminal 3
ros2 service call /gello/arm std_srvs/srv/Trigger
ros2 service call /gello/disarm std_srvs/srv/Trigger
```

The simulated arm starts at all zeros. Stop terminal 2 before terminal 1.
Without a window, `ros2 launch sparklab_yam yam.launch.py sim:=true cameras:=false`
runs both arms; its Quest bridge exits unless its web dependencies are installed,
which does not affect this.

## Topics and services

| name | type | |
|---|---|---|
| `/<arm>/joint_states` | in, `JointState` | measured pose, joint_1..6 + gripper |
| `/<arm>/arm/command` | out, `Float64MultiArray` | 6 joint targets (rad), joint_1..6 |
| `/<arm>/gripper/command` | out, `Float64` | 0 closed .. 1 open |
| `~/arm`, `~/disarm` | `std_srvs/Trigger` | the reply says why an arm request was refused; `/gello/left/...` and `/gello/right/...` in the launch |
| `/gello/arm`, `/gello/disarm` | `std_srvs/Trigger` | the launch's `pair` node: both of the above at once, both arms or neither |

Only the chosen arm is commanded, at `rate_hz` (100) with depth-10 QoS, as `yam_teleop` does.

## Behaviour

- Starts DISARMED and publishes nothing.
- While DISARMED the leader follows the arm: its servos are driven, at a torque
  limit of 300 of 1000, to the pose that reads back as the arm's measured pose,
  and the lever to the arm's gripper; the goal is rewritten only once the arm
  has moved 0.5 deg. A hand overrides it. `~/arm` then finds the two matched,
  and the leader goes limp once ARMED, on a fault, or when `joint_states` goes
  stale. `-p mirror:=false` turns this off.
- At shutdown (Ctrl-C) a following leader folds to YAM zero, where the drivers
  park the arms, at homing speed, then goes limp; keep hands off it. A limp
  leader is left where it is.
- `~/arm` is refused while another node publishes the same command topics
  (Quest teleop, keyboard) or `joint_states` is not fresh. Otherwise the next
  GELLO reading must be within `arm_tol` (0.15 rad) of the measured pose per
  joint. `gripper_arm_tol` (1.0) lets any lever position arm; the gripper then
  moves to the lever at `max_gripper_speed`.
- While ARMED, every target moves at most `max_joint_speed` (rad/s) and
  `max_gripper_speed` (1/s) from the previous target, scaled by elapsed time,
  so the publish rate does not change the speed. The driver's `max_step` bounds
  how far a target leads the measured pose. The gripper target is also clipped
  to 0..1. See [`DESIGN.md`](../../DESIGN.md#the-gello-node).
- A failed, late, non-finite, out-of-range or jumping GELLO reading → FAULT.
  `joint_states` silent for `state_timeout_s` (0.1 s) → FAULT. FAULT needs a
  new `~/arm`. The error repeats once a second; the node keeps running.
- After a USB replug, restart the node, then arm.
- Leaving ARMED (disarm, fault, Ctrl-C) publishes the measured pose once, then
  nothing, so the driver holds there. If `joint_states` is stale, nothing is
  published and the driver keeps its last target.
- If this node is killed, the driver keeps its last target, at most one
  `max_step` from where the arm was. The driver has no command timeout. See
  [`DESIGN.md`](../../DESIGN.md#the-gello-node).

## Parameters

| parameter | default | |
|---|---|---|
| `arm` | `left` | `left` or `right` |
| `port` | auto | servo adapter; auto when exactly one `/dev/tty.usbmodem*`, `ttyACM*` or `ttyUSB*` exists |
| `home` | true | drive the leader to the arm's measured pose and the lever to the measured gripper on start-up, then release it |
| `mirror` | true | while DISARMED, drive the leader to follow the arm at low torque, so arming finds them matched |
| `hold` | false | lock the leader's arm joints at half torque once every joint has been within 0.5 deg for 1 s; pushing any joint 2 deg releases them. The lever stays free |
| `calibration_path`, `mapping_path` | this package's `config/` | per leader unit |
| `rate_hz` | 100.0 | |
| `state_timeout_s` | 0.1 | |
| `read_retries`, `max_read_latency_s`, `max_input_speed_deg_s`, `range_margin_deg` | 1, 0.1, 720, 10 | input validation |
| `max_joint_speed`, `max_gripper_speed`, `max_dt`, `command_timeout`, `arm_tol`, `gripper_arm_tol` | 4.0, 4.0, 0.05, 0.25, 0.15, 1.0 | guard; simulation defaults, not validated for hardware |

## Tested

Simulation only; not yet on the real YAM.

| | how |
|---|---|
| joint order, rad, gripper 0..1, only the chosen arm commanded | `test_node.py`, and live topics |
| arm refused on pose mismatch; arm, disarm, re-arm | `test_node.py`, and by hand with the leader |
| arm refused while another node publishes the command topics | `test_node.py` |
| read failure, silent `joint_states` → FAULT, hold published once or not at all | `test_node.py` |
| time-based joint and gripper rate limits | `test_node.py` |
| leader follows; both gripper fingers move together | by hand, MuJoCo viewer |
| USB pulled while ARMED → FAULT, arm stops, node survives, clean Ctrl-C | by hand |
| after replug: same port name, the leader reads cleanly in a fresh process | read-only check; node restart and re-arm not yet run by hand |
| leader read time | median 1.27 ms, max 1.64 ms over 300 reads (macOS) |

## Mapping

`q_yam = sign * radians(leader_deg - zero_deg)` per joint, clipped to
`sparklab_yam/model/yam_ultra.xml`'s limits. The lever maps raw ticks `raw_a`
→ 0 (closed) to `raw_b` → 1 (open). Both files describe the lab's one leader
unit (Waveshare adapter, USB serial `5B8E1159981`); `zero_deg` was matched by
eye to the model at `qpos = 0`, so it is provisional. Another unit needs its
own EEPROM read-back and mapping.
