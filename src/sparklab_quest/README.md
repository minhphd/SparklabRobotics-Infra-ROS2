# sparklab_quest

Everything for the Meta Quest headset. The headset runs a web page (no app);
this package serves that page and turns what it sends into ROS topics.

```
sparklab_quest/bridge.py        ROS node: serves the page, publishes /quest/*, streams camera video
sparklab_quest/input.py         QuestInput: read the latest controller state from the topics
sparklab_quest/buttons.py       button numbers (trigger, grip, A/X, B/Y, thumbstick)
sparklab_quest/pose_mapping.py  ClutchPoseMapper: controller motion -> gripper target
web/                            the page the headset loads (HTML + JS)
test/
```

## The bridge

```bash
ros2 run sparklab_quest bridge --ros-args -p certfile:=cert.pem -p keyfile:=key.pem \
    -p rig:=src/sparklab_yam/config/cameras.yaml
```

The launch file starts it for you. It serves `https://<this-machine>:8443/`;
open that in the Quest browser.

```
headset page ── poses, buttons ─► bridge ─► /quest/head/pose, /quest/<arm>/pose, /quest/<arm>/joy
             ── settings ───────► bridge ─► /quest/settings
             ◄─ vibration, status  bridge ◄─ /quest/<arm>/haptic, /quest/<arm>/engaged
             ◄─ camera video ════  bridge ◄─ /cameras/<id>/color/image_raw
```

**HTTPS:** the headset only allows VR on a secure page. Make a certificate once
with `./scripts/make_certs.sh` and accept it once in the headset browser. Over a
USB cable (`adb reverse tcp:8443 tcp:8443`) plain `http://localhost:8443` works.

Parameters: `host`, `port` (8443), `certfile`, `keyfile`, `rig` (the camera
file; cameras without a running publisher are hidden).

## The topics

| topic | type | |
|---|---|---|
| `/quest/head/pose` | `PoseStamped` | headset pose |
| `/quest/<arm>/pose` | `PoseStamped` | controller pose, moved to your wrist so twisting doesn't translate |
| `/quest/<arm>/joy` | `Joy` | `buttons[i]` = pressed, `axes[i]` = how far (0..1), for button `i` |
| `/quest/settings` | `String` | JSON of the page's sliders; late subscribers still get it |
| `/quest/<arm>/haptic` | `Float32` | vibration strength 0..1 (from the teleop) |
| `/quest/<arm>/engaged` | `Bool` | clutch held (from the teleop) |

**Coordinates:** the frame is called `quest`. Y is up, −Z is the direction you
faced when you entered VR, origin on the floor. Quaternions are `[x, y, z, w]`.

**Buttons** (index into `joy.buttons` / `joy.axes`):

| name | index | used for |
|---|---|---|
| `TRIGGER` | 0 | gripper |
| `GRIP` | 1 | clutch |
| `THUMBSTICK` | 3 | send arm home |
| `A_X` | 4 | precision mode |
| `B_Y` | 5 | arm/disarm teleop; start/end a recording episode |

## Reading the Quest from your own node

```python
from sparklab_quest import buttons
from sparklab_quest.input import QuestInput

quest = QuestInput(node)               # subscribes on your node
ctrl = quest.controller("right")       # None until the first pose arrives
if ctrl and ctrl.age_s < buttons.XR_FRAME_STALE_TIMEOUT_S:
    holding = buttons.pressed(ctrl.buttons, buttons.GRIP)
    squeeze = buttons.value(ctrl.axes, buttons.TRIGGER)
```

## Clutch pose mapping

`ClutchPoseMapper` works like lifting a mouse off the desk:

- **Press grip** — remember where the controller and the gripper are.
- **Hold grip** — move the gripper by however much the controller has moved since.
- **Release** — the arm stops; reposition your hand freely.

The target is never allowed to run more than a set distance or angle ahead of
the arm; extra motion is dropped. Run `python -m sparklab_quest.pose_mapping`
for its self-test.

## The web page

`web/index.html` and `web/client.js` handle VR, the controller readout, the
wrist calibration and the camera panels. Add `?latency=1` to the URL to measure
round-trip time; the bridge logs the results.
