# sparklab_gello

One GELLO (YAM Active Leader) driving one YAM arm, joint to joint, through the
YAM driver's command topics. Reads the leader without enabling torque or
writing EEPROM.

```
sparklab_gello/reader.py   read, validate and convert one leader reading to YAM targets
sparklab_gello/guard.py    DISARMED / ARMED / FAULT latch with time-based rate limits
sparklab_gello/node.py     the ROS 2 node: reader -> guard -> command topics
config/gello_to_yam.yaml   GELLO -> YAM joint sign, zero and lever end stops (provisional)
config/yam_active_leader_eeprom.json  the leader's EEPROM calibration, read back
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

A second leader on the other arm needs its own node name and files:
`-r __node:=gello_right -p arm:=right -p calibration_path:=... -p mapping_path:=...`.

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
| `~/arm`, `~/disarm` | `std_srvs/Trigger` | the reply says why an arm request was refused |

Only the chosen arm is commanded, at `rate_hz` (100) with depth-10 QoS, as `yam_teleop` does.

## Behaviour

- Starts DISARMED and publishes nothing.
- `~/arm` is refused while another node publishes the same command topics
  (Quest teleop, keyboard) or `joint_states` is not fresh. Otherwise the next
  GELLO reading must be within `arm_tol` (0.15 rad) of the measured pose per
  joint and `gripper_arm_tol` (0.2) on the gripper.
- While ARMED, every target moves at most `max_joint_speed` (rad/s) and
  `max_gripper_speed` (1/s) from the measured pose, scaled by elapsed time,
  so the publish rate does not change the speed. The gripper target is also
  clipped to 0..1.
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
| `calibration_path`, `mapping_path` | this package's `config/` | per leader unit |
| `rate_hz` | 100.0 | |
| `state_timeout_s` | 0.1 | |
| `read_retries`, `max_read_latency_s`, `max_input_speed_deg_s`, `range_margin_deg` | 1, 0.1, 720, 10 | input validation |
| `max_joint_speed`, `max_gripper_speed`, `max_dt`, `command_timeout`, `arm_tol`, `gripper_arm_tol` | 4.0, 4.0, 0.05, 0.25, 0.15, 0.20 | guard; simulation defaults, not validated for hardware |

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
