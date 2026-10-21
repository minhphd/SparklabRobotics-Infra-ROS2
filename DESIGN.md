# Design notes

Why things are the way they are. The READMEs say *what* is where; this file
keeps the reasons and measurements, so decisions aren't re-argued or quietly
undone. You don't need any of it to run the robot.

## The big picture

**Why ROS 2.** Every part — arms, cameras, headset, policy — is a separate
process that publishes and subscribes. Any of them can be restarted,
replaced or inspected (`ros2 topic echo`) without touching the others.

**One Python for everything.** ROS 2 Humble's normal packages use Python 3.10,
but LeRobot needs 3.12. RoboStack builds Humble for 3.12, so ROS, LeRobot and
i2rt load in one process — which the LeRobot robot and teleoperator need, since
they are both LeRobot plugins and ROS nodes. Catch: torch loads the system's
older C++ library first and ROS then fails (`CXXABI_1.3.15`). Putting the env's
`lib/` first on `LD_LIBRARY_PATH` fixes it.

**Package split.** The ROS packages (`sparklab_yam`, `sparklab_quest`,
`sparklab_cameras`) never import LeRobot, so they start in a second and never
load torch. That is why the teleop logic (`QuestTeleop`) lives in
`sparklab_yam` and LeRobot's teleoperator is a thin wrapper. `sparklab_sim`
runs on Isaac's Python, which has neither ROS nor LeRobot, so the rule there is
mechanical: nothing in it may import either. It reads the shared model files by
path instead.

**Standard messages only.** Custom `.msg` types need a C++ build step; every
topic here fits standard types. The two JSON-in-a-string topics
(`/quest/settings`, `/rollout/status`) are UI payloads only the UI reads.

## The arm drivers

**One process per arm.** i2rt's CAN thread must send within each motor's
watchdog window. When it shared a process (and Python's GIL) with policy
inference, it got starved and the motors reported `loss communication` (0xD)
mid-run, while every CAN error counter read zero — the wire was fine, the
frames just weren't sent in time. `scripts/can_health.sh` tells the two apart:
nonzero counters mean wiring, zero counters with errors mean timing.

**Two clamps.** The driver clamps every command to `max_step` from the measured
pose, so nothing — not even a typo in `ros2 topic pub` — can jump an arm. The
LeRobot robot clamps the same way first, so the action it records is the one
the arm actually got.

**Silence means dead.** When i2rt's control loop dies, `get_joint_pos()` keeps
returning the last pose, so the arm *looks* fine while ignoring commands. So the
driver stops publishing `joint_states` the moment the loop dies, and clients
refuse joint states older than 0.5 s.

**Parking.** i2rt's `close()` cuts torque wherever the arm is, so the driver
first ramps every joint to zero — the folded rest pose, where every recorded
episode starts. It parks on its own shutdown because nothing else can if its
process is killed.

## Teleoperation

**Forward is where you're facing.** Each time you press grip, the mapping takes
the headset's current heading, so pushing "forward" moves the arm away from you
even after you turn around. Only yaw is used, and turning your head mid-grip
changes nothing until the next press.

**Grip re-anchors every time**, on the arm's last *commanded* pose, not its
measured one. The measured pose lags by a slightly different amount each time,
so anchoring on it made the arm drift on every press. A background re-sync
thread was tried and removed: it raced LeRobot's startup and tangled the
teleop with the robot. Now whoever moves the arms by other means calls
`seed()` (the teleop node after arm/disarm, `record_manual` after parking).

**Reach limits absorb extra motion.** The target can't run more than
`pos_reach_limit` / `rot_reach_limit` ahead of the arm; extra hand motion is
dropped, like a mouse at the edge of the screen. Without this, pushing past a
joint stop built up a huge error that made the arm shake at full speed, and
twisting far enough snapped the wrist round the other way. The cost: dropped
motion is gone, and with a rotation gain ≠ 1 curved hand paths don't map
exactly (a 60°+60° loop at 1.5× leaves ~23°). Re-gripping realigns.

## Inverse kinematics

Joints 1–3 handle position, 4–6 orientation, one damped step per tick. The
wrist isn't perfectly spherical (6.2 cm offset), so twisting adds a small
position error; the operator corrects by eye.

`solve()` never fails or freezes. Four edge cases:

1. **Edge of the workspace** — damping grows so the step shrinks smoothly to zero.
2. **Wrist gimbal lock** (joint 5 near ±90°) — same trick for the wrist; the
   old exact solution demanded 68° of joint motion per 1° of twist there.
3. **Joint limits** — clamped, so the arm reaches as far as it can.
4. **Twist demanded nearly 180° away** — the wrist holds still. The
   shortest-way direction flips with tiny hand tremor there, which made the
   arm shake (reproduced in sim at ~150° with 0.2° of tremor).

## The arm model

Copied byte-for-byte from i2rt v1.2.4, with one deliberate change: joint 3's
upper limit is **3.0 rad**, not π, because this rig's elbow hits its stop
around 3.0. Wrong limits fail silently both ways — too tight is an invisible
wall, too loose grinds into the stop — so after any i2rt upgrade run
`python -m sparklab_yam.model.kinematics`. Meshes don't affect kinematics.

## Cameras

**One owner.** A RealSense can only be opened by one process. The old setup had
the headset relay and the recorder each opening them, so recording meant
turning off the headset video. Now the camera node owns them and everyone
subscribes.

**Depth.** Every camera also publishes depth, aligned to its colour image, so
one pixel means the same point in both and one set of intrinsics serves both.
Alignment costs 1.9 ms per wrist frame and 6.6 ms per top frame. Depth is
converted to millimetres (ROS's convention for 16-bit depth) because the
cameras disagree on units: 1 mm on the D435F, 0.1 mm on the D405s. All three
sustain colour + depth at full rate, the D405s even on USB 2.1.

**Image speed.** Assigning `bytes` to `Image.data` takes ~90 ms for a 720p
frame (rclpy copies it element by element) and capped the top camera near
4 Hz. Assigning an `array.array('B')` takes 0.1 ms; with it all three cameras
run at full rate (30/15/30 Hz, ≤ 3 ms latency) on the default DDS. CycloneDDS
was no better.

**Point clouds in the camera node.** The node already holds each aligned frame,
so building a cloud there costs 0.45 ms (wrist) / 0.9 ms (top); a separate
`depth_image_proc` node would add a dependency and a second copy of every image.
Every 4th pixel at ≤ 10 Hz keeps a cloud at 0.3 / 0.9 MB, light enough for
Foxglove over a laptop link, and nothing is built unless someone subscribes, so
recording is unaffected. TF is rooted at `left_base` because the top camera was
calibrated against the left arm; `right_base` hangs off it through the top camera.

## Camera calibration

**Zero gravity is a driver service**, because only the driver may touch the
motors. While it's on, the driver ignores commands, so a forgotten teleop can't
fight the person moving the arm. Turning it off holds the arm where it is — no
ramp anywhere.

**Zero gravity follows Raiden's tested recipe**, after i2rt's own mode let the
arm fall: arm kp to 0 with kd kept (a released arm can't run away), the
gripper's gains untouched, and the gravity factor at 1.0 while floating. The
real culprit was mass: the long gripper weighed 0.953 kg (Raiden's `arms.json`), but
i2rt models the stock 0.553 kg one. The holding factors stay separate, since
they also cancel friction and tracking error under kp. With the stock gripper back
(see [cap-x on the rig](#cap-x-on-the-rig)), the drivers use i2rt's
mass and factors again.

**The left arm's joint 4 zero was 5–7° off.** Hand-eye calibration stalled at
7.5 mm / 1.6° with a 0.2–0.3 px board fit, so the camera was fine; a fitted
joint-4 offset of −6.6° brought it to 1.7 mm / 0.28°. With the gripper held
level it read +5.04°, so motor 4 on `can_left` was re-zeroed there on
2026-10-14 (shift 0.0879 rad): left joint 4 recorded before then reads 0.0879
rad higher. Joint 5 read −3.4° in that by-eye pose but the fit put it within
0.4°, so it was left alone. Recalibrating then gave 4.0 mm / 0.61°, with −1.3°
still fitted on joint 4; even with joints 2–5 free, ~3 px remain against a
0.25 px board fit — the arm's own accuracy. The right arm's joint 4 read +4.36°
with its gripper level, the same way, and was re-zeroed there the same day
(shift 0.0761 rad). Both arms erring alike points at how they were zeroed.
Calibrating afterwards still fitted −1.3° (left) and −1.6° (right) on joint 4:
a phone level on the gripper lands ~1.5° short of the model's zero.

**Captures need a still arm.** The arm pose (100 Hz) and the image (15–30 Hz,
tens of ms late) are matched by timestamp, so motion during a capture becomes
calibration error. The node refuses unless the gripper moved less than 1 mm and
0.2° in the last half second.

**How the error is measured.** The board never moves (on the table, or on the
gripper), so every capture should predict the same board pose through the
answer. The spread of those predictions (mm, °) and the reprojection error (px)
say how well the answer fits. Hand-eye calibration needs rotation about at
least two different axes, hence the warning below 30° of rotation spread.

Solved with OpenCV's Park method. Checked on synthetic captures with 0.3 px
noise (within 2 mm / 0.3°) and end to end against the sim arm with a rendered
camera (1.1 mm / 0.28°).

## The Quest bridge

**HTTPS with the right certificate.** The headset only allows VR on secure
pages, and it rejects certificates without a `subjectAltName` outright — which
a plain `openssl req -x509` produces. `make_certs.sh` puts every local IP in
it, so one cert works on the LAN, the campus network and a USB tunnel.

**H.264 video**, because the Quest decodes it in hardware; VP8 is software.

**Settings are latched.** The page sends its slider values when it connects.
Publishing them as a latched topic means a teleop that starts later still gets
them.

## Shutting down safely

**Signals.** By default, Ctrl-C makes rclpy shut ROS down immediately — after
which the teleop can't send the arms home and LeRobot can't ask the drivers to
park. So every process here handles Ctrl-C itself and ignores a second Ctrl-C
while parking. The launch file gives the drivers 20 s (instead of 5) before
escalating, because parking takes 5 s. A background job starts with Ctrl-C
ignored, so the nodes re-enable it explicitly.

**Background spinning.** In Humble, shutting an executor down while another
thread is spinning it crashes the process ("terminate called without an active
exception"). Background spinners loop on `spin_once` and are joined first; an
`atexit` hook does the same if a process dies with one still running.

## The Isaac twin

**Poses, not physics.** The policy reads images and joint angles and outputs
joint angles; nothing asks the sim for forces. Skipping dynamics also skips
tuning friction and gains that no recording could tell us. A policy can
"grasp" through the box — good for checking actions are sane, not for judging
grasps.

**A separate robot type.** `sim:=true` on the launch only fakes the arms; the
cameras stay real. A policy without hardware needs *rendered* cameras, hence
`yam_ultra_sim`. It stays on HTTP because Isaac's Python can't load ROS.

**Same clamps as hardware.** An unclamped sim flatters the policy: a jump that
hardware would cut to 0.15 rad runs in full, hiding the bug you're looking for.

**Park between runs.** The sim server stays up between rollouts, so a run's
last pose is the next run's start — and LeRobot then restores to it. One run
that ended unparked left an arm 44 sigma away from any demo's start pose, and
the policy flailed. All 49 demos start at zero. A clean disconnect parks; the
server's idle auto-park covers crashes (only `/step` counts as activity).

**Isaac on the main thread only.** `app.update()` called from a request thread
never returns — it hangs silently while `/health` keeps answering. So requests
queue work and the main loop does it. Rendering all cameras on one update
instead of one per camera took an observation from ~116 ms to ~40 ms.

**Fast scene editing.** `live.py` loads the scene as a sublayer and reloads only
that layer: ~0.04 s per edit, against ~0.38 s to reopen the stage or a full
reboot. Reopening the root layer deletes the render products and segfaults,
hence `Renderer.close()`. The timeline is always playing (joint writes need it),
so a reload bounces it to reset physics-owned objects.

**Not Isaac's WebRTC stream**: it has no browser client here, needs extra ports,
and uses UDP, which SSH tunnels can't carry. The MJPEG view is one TCP port.

**The digital twin speaks ROS.** The policy server stays on HTTP because Isaac's
Python can't load the workspace's rclpy. Isaac 5.1's ROS 2 bridge ships its own
Humble rclpy for Python 3.11, though, and it talks DDS to sparklab-ros like any other
node. So `twin.py` serves the drivers', camera node's and launch's topics itself, and
cap-x and VLM_TAMP run on the twin with nothing changed but `ROS_DOMAIN_ID`. Its world
frame is `left_base`, the frame every client already works in. The stage is built from
the calibration files every launch, never hand-edited, so it can't drift from them.

**Checked against the real cameras.** Forward-kinematics points projected through the
calibration land on the same gripper features in real and twin frames, and the twin's
`tool0` equals the driver's FK to 0.1 mm (`test_twin.py`). `intrinsics.yaml` had been
exported at the D4xx's default 848x480 and rescaled by width, which is wrong for the
D405: its 640x480 mode is a different crop (fx 392, against 327 rescaled). It now holds
the lens at the captured resolution.

**The finger direction was backwards.** The twin rendered "open" fingers meeting in the
middle. In the URDF a finger joint at 0 is closed, with the tips touching, and −0.04695
is open, 9.4 cm apart. cap-x had it the other way round. cuRobo's model locked the
fingers "open" at 0, so it planned with closed fingers, and the Foxglove view and the
GraspGenX collision mesh drew the hand inverted. All three are fixed. With the wider
hand modelled, the handover giver can't back out 8 cm from the meeting pose, so it backs
out 5 cm.

**Looks fitted, not guessed.** Colours are the sRGB the real top camera records. The
ambient, ceiling and window light levels were fitted with Nelder-Mead against six table
regions of a real frame, to 12/255 mean error, with auto-exposure off so the levels
mean what they did when fitted. The real table is markedly brighter on the right arm's
side, hence the window. The stock fingers take the gripper body's grey until they
are sampled from a frame.

**The twin's joints are the real motors'.** Force drives with i2rt's kp and kd
(`yam_ultra.yml`), against no link gravity, stand in for the drivers' gravity-
compensated PD, so a payload sags as on the rig. The importer's acceleration drives
had scaled with link mass: the wrists held a 0.35 kg can at about 1.5 N m/rad
instead of 10. The URDF's `effort="1"` had capped every drive at 1 N; the caps are
now the DM4340s' and DM4310s' peak torques. The real fingers share one motor through
a rack. Two drives that saturate let a gripped object float between them, and a
PhysX mimic joint coupling them pushed 1.5 N m into the wrist pitch whenever they
squeezed something: joint 5 sat 0.15 rad off, and every carry tripped the lag guard,
which the real arm never did. So each finger has its own drive, capped above
anything its stiffness asks: unsaturated, the two centre a held object. Props weigh
their BEHAVIOR category's average, not the files' 1 kg.

## Rollout control

- **A new task clears the queue.** The policy plans 30 steps at a time; without
  clearing, up to 30 old-task actions (1.5 s) run first and the arm seems to
  ignore you, then lurches.
- **Going home is a smooth move**, not one big target. Letting the clamp limit
  a single jump meant moving home at the fastest allowed speed (~4.5 rad/s),
  which felt like a jolt. Peak speed is now ~0.9 rad/s.
- **Reset holds.** Otherwise the next tick runs the policy again and the arm
  climbs straight back out, so reset looked like it did nothing.
- **ROS callbacks only queue.** Neither the robot nor the model is thread-safe;
  the control loop applies commands between ticks.

## The agent harness

Separate from the rollout on purpose: the rollout holds a ~21 GB model and
takes ~90 s to restart, while the harness can restart any time. If it crashes,
the robot keeps doing its last task. An agent can only set a task or say it's
done — it can't park or stop the robot, so a confused agent picks a wrong step
rather than doing something drastic. A generation counter drops any decision
that was in flight when you pressed stop.

## cap-x on the rig

**One process, two environments.** cap-x needs CUDA torch, cuRobo and GraspGenX
from its own venv; ROS 2 only installs into the conda env. Both are CPython 3.12,
so `scripts/capx_python.sh` runs the conda interpreter isolated, cap-x's
site-packages first and ROS 2's after. rclpy, messages with numpy fields and tf2
all work next to cap-x's numpy 1.26: extensions built against numpy 2 target the
1.x API by default. The alternative, a bridge server between two processes, would
have brought back the `arm_server` the migration removed.

**The robot model is generated, not hand-written.** `capx_sparklab.robot.build`
starts from sparklab_yam's URDF, i2rt's stock gripper included, and adds the
planning frames; the collision spheres and GraspGenX's collision mesh are fitted
from it, so a change of hand is a rebuild. The
planners' tool frame is the gripper base (`grasp_base`), not `tool0`: GraspGenX
returns gripper-base poses, as it does `panda_hand` for the Franka. Both arms plan
in `left_base`; each arm's URDF gets a world joint set from the calibration.

**Collision at park.** The arms rest folded, so every plan starts there. cuRobo's
fitted spheres made park look like a collision twice over: the base's hung 3 cm
below its mounting face into the table (now clipped at the face), and folded
links overlap by a few millimetres as spheres, a different pair on each build.
The same happens fingers-down: one fit left link3 and link5 overlapping there, and
no fingers-down pose planned. So every pair whose spheres overlap at a pose the real
arm reaches without touching itself (parked, and fingers-down 0.3 m over the table)
is left out of self-collision, and the build prints which.

**Wrist images are undistorted in `YamEnv`.** cap-x deprojects depth as a pinhole
camera; the D405s have k1 ≈ -0.05, a few pixels at the edges.

**Handover geometry was measured on the model, not chosen.** The YAM wrist rolls
about ±120°, which rules out a lot:

- Fingers straight down must be the pitch-down of the parked orientation; the
  same direction rolled half a turn is out of reach.
- Presented fingers-down, an object held near one end hangs to table height at
  any height the arm reaches, so the giver holds it level instead.
- Level, the giver cannot point straight at the other arm or straight ahead;
  30–60° round from forward works from 0.20 m up.
- Of those, only 60° at 0.30 m in front of the bases' midpoint leaves the receiver
  a plannable level grasp: at 30°, or 0.22 m out, it has none.

**Worked around in cap-x, not fixed there** (the checkout belongs to another
account): its `move_relative` binds `quat_wxyz` to an array and then calls it, so
every successful move raised after moving — `YamSparkLabApi` carries a corrected
copy; LIBERO asks on stdin for a dataset folder the first time an account imports
it, which hung runs silently — the launcher writes its default config; cap-x's
model client loads the checkout's `.env` on import — the tests turn that off; and
`Voice` loads a TTS model and needs an audio device — lines are printed instead.

**cap-x lives next to the repo, in this account.** The checkout and its venv were
copied from the account that set them up, and the venv's editable-install paths
rewritten, so nothing here reaches into another user's home.

**The scene goes to Foxglove, not viser.** One viewer for the rig: `RosScene`
answers cap-x's drawing calls with ROS topics and publishes both arms' links on TF.
`YamEnv`'s ROS thread logs a failing callback and keeps spinning: a drawing bug
once froze the joint states every motion check reads.

**Three rig facts learnt the hard way.** The drivers' gripper is 1 open, 0 closed;
read the other way round, `open_gripper` closed it. cuRobo's fusion takes one
resolution for every camera, while the top camera is 1280x720 and the wrists
640x480, so frames are resampled to 640x480, intrinsics scaled, before fusing.
And fusion fixes its camera count on the first refresh, while the newest colour
frame lacks its depth on 11-16% of reads per camera (measured live), so a camera
that skipped one refresh failed every later one ("Expected num_cameras=2"). The
env keeps three colour frames as well as three depth, and pairs the newest match.

**The gripper has a GraspGenX collision mesh.** cap-x screens candidates against
the scene with the gripper's `coll_mesh.obj` before keeping the top 20, and skips
the screen when the mesh is missing. Without it, a marker lying flat got 20 side
grasps through the table and none planned. `build.py` now writes the open hand's
hulls. Offline, with a marker in reach: 12 of 20 planned without the mesh, 19 with.

**Top-down reach is about 0.45 m.** With the stock gripper, a grasp from above on
the table (10 cm standoff and lift) plans out to 0.46 m straight ahead of the arm's
base and 0.44 m at 15 cm across, and as close as 0.12 m. The grasp centre gets no
higher than 0.12 m at 25–30 cm out, 0.10 m at 36 cm and 0.07 m at 42 cm. With the
14.5 cm fingers it was 0.25 to 0.43 m. The prompt had once said 0.55 m, and the
agent retried a marker at 0.56 m for dozens of turns.

**Grasps stop at 0.35 m: the shoulder's rating, not the planner's reach.** With the
stock gripper the agent took a grasp 0.48 m out. Through the grasp and lift, joint 2
read 14–16 N m on the rig, and the arm whined. A DM4340 is rated 9 N m continuous
(27 N m peak), and the agent holds each pose for a model's turn, some 30 s. i2rt's
gravity model, which the driver uses, predicted 13.7 N m there against the 14.4
measured. Fingers down at the table, it puts joint 2 at 7.0 N m 0.28 m out, 8.4 at
0.32, 9.7 at 0.36 and 11.0 at 0.40; a 10 cm lift takes off about 0.5. So capx's
`get_grasp_poses` refuses objects past 0.35 m, and TAMP's `pick` does the same.

**Plans play at half speed.** cuRobo times plans at up to 0.9 rad/s per joint, the
URDF's 1 rad/s. Played in real time, the right shoulder with the 0.95 kg gripper
trailed `home` by 0.16 rad, and `YamEnv`'s 0.15 rad lag guard stopped it; the
driver's 0.133 rad `max_step` clamp fired at the same moment. The robot config's
`velocity_scale` lowered the limit cuRobo reports, but not the trajectory: still
81 waypoints at 0.8 rad/s. So `_execute` resamples each plan to twice the
waypoints and streams them at the same 40 Hz.

**Grasps are confirmed by eye.** cap-x's `close_gripper` reports `grasped` from
the closed width, and `handover` reopened the receiver on the same test. The rig's
fingers are soft with no tactile sensing, so the width does not separate an object
from nothing. Both checks are gone: `close_gripper` tells the agent to look with
`view_cameras`, before attaching and again after a short lift. Looking comes after
closing because the unattached object's cloud surrounds the gripper, and nothing
plans until it is attached.

**Back to the stock gripper.** The rig ran SparkLab's long gripper for a while: TPU
fin-rays from [YAM-iPhUMI](https://github.com/sparklabutah/YAM-iPhUMI) on the stock
sliders, 14.5 cm base to tip against 8 cm, and 0.953 kg against 0.553. What it cost,
measured in the twin and on the rig on 2026-10-15: reach from above ended at 0.43 m
and about 10 cm-tall objects; the wrists, at kp 10, carried 0.4 kg more on a longer
lever; the fitted fingertip spheres reached 1 cm past the tips, which trapped a low
marker grasp; the fingers filled the wrist views, whose ray-shadows fuse as phantom
surfaces; and the 4 cm-wide fingers left little of a held object for a handover. The
fin-rays' compliance did not outweigh that for planning work, so the rig went back
to i2rt's LINEAR_4310 on the same camera mounts.

**A grasp is left by the lift planned with it.** On the rig, a GraspGenX grasp on a
1.9 cm marker put the fingertips at z ≈ −0.012. The fitted fingertip spheres
reached −0.022, 1.2 cm into the planner's table, whose top deliberately errs high
at −0.010. cap-x plans the final descent with the fingers exempt, so the grasp was
reached. From that start, though, every later motion is in collision: the
`move_relative` lift, the pregrasp and `home` all failed, and the arm was stuck.
cap-x does plan a 10 cm lift with the same exemption, to prove the grasp can be
left, but then discards it. `lift` plays that lift. The twin reproduced the
failure and the fix: the marker rose 10 cm, and `home` then planned. Closing on
an object pinned to the table deflects the wrist about 0.05 rad. So `lift`
accepts any start within the lag guard's 0.15 rad of the plan's first waypoint.

**Tests without cap-x are not imported at all.** ROS's launch_testing pytest
plugin imports every test module; a module-level `importorskip` inside it skips
the whole session, every package's tests. `conftest.py` leaves them out instead.

## Two rules plugin discovery depends on

- **The package name starts with `lerobot_robot_`.** LeRobot imports every
  installed package with that prefix; that import registers our robot types.
- **The package `__init__.py` re-exports every Robot and Teleoperator class.**
  LeRobot looks for the class next to its config; without the export you get
  `Could not locate device class 'YamUltraFollower'`.

## Measured on this rig

- **Trust the checkpoint's `config.json`.** Forcing `--policy.use_amp=true` on a
  bfloat16 model wraps it in float16 autocast: slower *and* worse actions.
- **Loop speed is arm speed.** One tick = one action of a trajectory learned at
  30 fps, so a slow loop plays it in slow motion. Foxglove's `--display_data` and
  CPU pinning both slowed the arm visibly.
- **CPU pinning backfired**: 30 Hz unpinned vs ~4 Hz with `taskset -c 0-5` — a
  rollout alone uses ~400% CPU.
- **Skipping "wasted" preprocessing made the policy worse.** The sync engine
  preprocesses images every tick and uses 1 in 30 (12.5 ms each). Bypassing it
  was tried and reverted.
- **`max_relative_target` caps step size, not speed.** Speed comes from the motor
  gains; capping the step bounds the error spike.
- **RTC inference needs `n_action_steps / fps > inference time`**, or whole
  chunks get dropped and the arm doesn't move. Sync inference instead pauses
  the loop ~275 ms every 30 ticks.
- **Validate a checkpoint offline first** with
  `scripts/analysis/validate_policy_pipeline.py`.

## Removed in the ROS 2 migration

So they aren't re-added without knowing why they went:

- **`arm_server` / `arm_client`** (portal RPC) → the driver node.
- **The WebSocket relay and `XRFrameClient`** → the bridge publishes topics.
- **`sparklab-rollout-ctl`** → `ros2 topic pub` / `ros2 service call`.
- **Single-arm teleop scripts and `pure_sim.py`** → the teleop node with
  `hands:=[right]`, or `sim:=true`.
- **`--robot.sim`** (the robot starting its own sim servers) → `sim:=true` on the
  launch. That also removed the "is this server sim or real?" check, which only
  existed to catch leftover servers it had started.
- **`gripper_flip`** — never used.
- **The Rerun pose viewer** — the poses are topics now; any ROS viewer shows them.
- **Gripper force haptics** — nothing ever supplied gripper torque, so it never
  ran. Gripper effort is in `/<arm>/joint_states` if it comes back.
- **The DAgger hooks** (`is_engaged`, `is_handoff_pressed`, …) — on request;
  `record_manual` reads B/Y from `/quest/<arm>/joy` instead.
- **`cameras/streaming.py`** — any ROS image viewer shows `/cameras/*`.
