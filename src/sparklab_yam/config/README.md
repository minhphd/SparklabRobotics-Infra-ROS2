# config

Settings for *this* physical rig.

```
yam.yaml                  ROS parameters: drivers (CAN channels, limits), teleop speeds, calibration board size
cameras.yaml              which cameras exist: id, serial number, resolution
extrinsics.yaml           calibrated camera and arm-base poses, published as static TF by the launch
hand_default.json         saved RealSense settings for the two wrist cameras
thirdperson_default.json  saved RealSense settings for the top camera
intrinsics.yaml           camera intrinsics, exported once
```

## cameras.yaml

| id | serial | model | capture |
|---|---|---|---|
| `left_wrist` | 353322271147 | D405 | 640×480 @ 30 |
| `right_wrist` | 323622272781 | D405 | 640×480 @ 30 |
| `top` | 243322071190 | D435F | 1280×720 @ 15 |

The camera node, the headset bridge and the LeRobot robot all read this file.

**The ids matter:** they are the topic names (`/cameras/<id>/color/...`) and the
dataset column names (`observation.images.<id>`). Renaming an id renames a
dataset column.

Per-camera keys: `serial`, `width`, `height`, `fps`, `crop_height` (cut rows off
top and bottom after capture), `advanced_json` (a saved settings file), or
`exposure` / `white_balance` / `gain`. `height` must be a mode the camera
actually supports.

To record 640×360 like the `allenai/MolmoAct2-BimanualYAM-Dataset`, set
`crop_height: 360` on the wrist cameras and switch `top` to the commented-out
640×360 line.

## Replacing a camera

Update its serial in `cameras.yaml`, load its settings again (`apply_presets`),
and re-export the intrinsics. A wrong serial shows up as a camera that won't
open — or as left and right wrists silently swapped.
