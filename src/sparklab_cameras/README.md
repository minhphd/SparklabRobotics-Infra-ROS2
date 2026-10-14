# sparklab_cameras

The RealSense cameras on ROS 2, plus a few tools you run by hand to set them up.
Which cameras exist is configured per robot, in
[`sparklab_yam/config/cameras.yaml`](../sparklab_yam/config/README.md).

```
sparklab_cameras/node.py               ROS node: publishes every camera
sparklab_cameras/apply_presets.py      load saved camera settings onto the hardware
sparklab_cameras/calibrate.py          try exposure / white balance / gain values
sparklab_cameras/export_intrinsics.py  save every camera's intrinsics to a YAML file
```

## The camera node

```bash
ros2 run sparklab_cameras cameras --ros-args -p rig:=src/sparklab_yam/config/cameras.yaml
```

The launch file starts it for you. This node is the only thing that opens the
cameras — the headset bridge and LeRobot just subscribe. A camera that can't be
opened is skipped with an error in the log.

Each camera publishes these topics, all with the same timestamp:

| topic | type | |
|---|---|---|
| `/cameras/<id>/color/image_raw` | `Image` `rgb8` | colour image |
| `/cameras/<id>/depth/image_raw` | `Image` `16UC1` | depth in **millimetres**, 0 = no reading |
| `/cameras/<id>/color/camera_info` | `CameraInfo` | intrinsics (focal length, centre, distortion) |
| `/cameras/<id>/depth/camera_info` | `CameraInfo` | the same intrinsics |
| `/cameras/<id>/points` | `PointCloud2` | xyz (m) + rgb in the camera frame, every 4th pixel, ≤ 10 Hz |

Depth is **aligned to colour**: same size, and pixel (u, v) in the depth image
is the same point as pixel (u, v) in the colour image. That's why both share
one set of intrinsics. `frame_id` is the camera id. The point cloud is only
built while something subscribes to it, and corrects for lens distortion.

## Freezing camera settings

Auto-exposure flickers during recording, so settings are fixed once. Pick
**one** of these per camera:

**A saved preset** (what the rig uses). Save settings from `realsense-viewer`,
point the camera's `advanced_json` at the file, and load it after each power-up:

```bash
ros2 run sparklab_cameras apply_presets src/sparklab_yam/config/cameras.yaml
```

**Three values in the camera file** — `exposure`, `white_balance`, `gain`. The
camera node applies them on every start. Find good values with:

```bash
ros2 run sparklab_cameras calibrate --serial 323622272781
```

## Intrinsics

```bash
ros2 run sparklab_cameras export_intrinsics intrinsics.yaml
```

Only needed for 3D work; policies that use raw pixels don't need it.
