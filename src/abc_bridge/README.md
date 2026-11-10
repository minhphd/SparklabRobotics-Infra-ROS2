# abc_bridge

Runs [ABC](https://github.com/amazon-far/abc)'s released policy on the rig. ABC
runs unchanged as its own websocket server (`deploy/serve_policy.py`, in ABC's
uv env); this node is its client on the ROS side.

```
abc_bridge/node.py   ROS node: observations to the server, 30 Hz RTC execution of its action chunks
```

## Running it

```bash
# ABC's server, from ABC's clone (fast inference: ~45 ms per call, ~3 min to start)
uv run deploy/serve_policy.py --policy.fast-inference \
  --policy.checkpoint-path=$ABC_CACHE/abc_dit_xl_200k_model.pt --port 8000

# the rig, then the bridge (dry run)
ros2 launch sparklab_yam yam.launch.py
ros2 run abc_bridge bridge --ros-args -p prompt:="throw plastic bottles in bin"
```

| parameter | default | |
|---|---|---|
| `prompt` | — | required; must match a task name the checkpoint trained on |
| `dry_run` | `true` | publish to `/abc/preview/<arm>/{arm,gripper}/command` instead of the command topics, and never park |
| `host`, `port` | `localhost`, `8000` | ABC's server |
| `record_dir` | — | if set, record the episode to `<record_dir>/<time>_<prompt>/` as MCAP |
| `noise_seed` | `-1` | `>= 0`: send a fresh (30, 14) N(0, 1) `noise` with each request, drawn from this seed. `-1`: send none; the server then uses zeros. See [`DESIGN.md`](../../DESIGN.md#the-abc-bridge) |
| `start_pose` | ABC's | per-arm ramp target, 6 joints then gripper, e.g. `[0.,0.,0.,0.,0.,0.,1.]` (park, gripper open) for policies trained on our GELLO demos. See [`DESIGN.md`](../../DESIGN.md#the-abc-bridge) |

With fast inference, the first request carrying `action_prefix` recompiles the
model (several minutes, the server unresponsive meanwhile); send one before
starting the bridge, or its first chunk waits that long.

On start the bridge ramps both arms over 5 s to `start_pose` (default ABC's,
`[0, 0.5, 1.0, -1.0, 0, 0]` with the gripper open), asks for the first chunk,
prints each joint's first-action delta and waits for Enter. The episode ends
on Ctrl-C: the bridge stops publishing and, unless dry-running, parks both
arms. It also stops, and parks, if a joint state or image is older than 0.5 s
or a target for any arm joint lies more than 0.5 rad from where that joint
measures, the first action included.

Start it after the drivers log "at the rest pose": during their startup ramp
they ignore commands, so the bridge's own ramp would be dropped.

## Mapping

| ABC | rig |
|---|---|
| `state` (14,) float32 | `/left/joint_states` then `/right/joint_states`: joints 1–6, gripper |
| `actions[:, 0:6]`, `actions[:, 6]` | `/left/arm/command`, `/left/gripper/command` |
| `actions[:, 7:13]`, `actions[:, 13]` | `/right/arm/command`, `/right/gripper/command` |
| `images.top`, `.left`, `.right` | `/cameras/{top,left_wrist,right_wrist}/color/image_raw`, centre-cropped to 4:3 and resized to 640x480, sent as (3, H, W) uint8 |

Joints are absolute positions in rad; the gripper is 0 closed .. 1 open on both
sides.

## Timing

ABC's own RTC settings (`--execute-chunk-dim=16 --rtc-prefix-length=4
--rtc-inference-lead-steps=7`): each reply is 30 actions, of which
`actions[4:20]` run at 30 Hz. At step 9 of 16 the next request goes out with
`action_prefix` = the chunk's last 4 actions; the first request uses the
current state repeated 4 times. A reply that is not back by the end of the
chunk is logged as late and the arms hold their last command until it arrives.

Each chunk logs its inference time, the largest joint jump at the chunk
boundary and the running count of late chunks.

## Recording

With `record_dir` set, the bridge runs `ros2 bag record` itself: from the end
of the ramp, so the first request and reply are in the bag, until the arms
have parked. Each episode gets its own folder. Recorded alongside the topics
below: `/<arm>/joint_states` and the command topics the bridge publishes to.
If `ros2 bag record` exits or does not subscribe within 15 s (its MCAP plugin,
`rosbag2_storage_mcap`, missing for one), the bridge stops and parks rather
than run unrecorded.

Every request and reply is also published:

| topic | type | |
|---|---|---|
| `/abc/obs/images/{top,left,right}` | `Image` `rgb8` | the 640x480 frames sent, after the crop; `header.stamp` is the camera node's stamp (taken when it publishes the frame, after alignment), so record time minus stamp is the frame's age when sent, excluding sensor and USB latency |
| `/abc/obs/state`, `/abc/obs/action_prefix` | `Float32MultiArray` | (14,) and (4, 14), as sent |
| `/abc/obs/noise` | `Float32MultiArray` | (30, 14), as sent; only with `noise_seed >= 0` |
| `/abc/obs/prompt` | `String` | the prompt sent |
| `/abc/actions` | `Float32MultiArray` | the server's full (30, 14) reply; `actions[4:20]` are executed |
