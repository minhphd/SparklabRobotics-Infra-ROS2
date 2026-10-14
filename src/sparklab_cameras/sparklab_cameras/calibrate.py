"""One-time RealSense exposure / white-balance / gain calibration.

Autoexposure hunts and flickers when the camera is read at a fixed frame rate,
so each camera's settings are frozen by eye and copied into
the rig's ``cameras.yaml``, which the camera node reads::

    ros2 run sparklab_cameras calibrate --serial 323622272781

Re-run with different --exposure/--white-balance/--gain until the feed looks
right, then paste the values into cameras.yaml.
"""

from __future__ import annotations

import argparse

import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", required=True, help="camera serial, e.g. from cameras.yaml")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=150)
    ap.add_argument("--white-balance", type=int, default=4600)
    ap.add_argument("--gain", type=int, default=64)
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)

    # D405/D415: a single sensor carries color (and depth, for the D405).
    sensor = pipe.get_active_profile().get_device().query_sensors()[0]
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.enable_auto_white_balance, 0)
    sensor.set_option(rs.option.exposure, args.exposure)
    sensor.set_option(rs.option.white_balance, args.white_balance)
    sensor.set_option(rs.option.gain, args.gain)

    print(f"applied exposure={args.exposure} white_balance={args.white_balance} "
          f"gain={args.gain} to {args.serial}")
    print("eyeball the feed (realsense-viewer, pointed at the same serial), Ctrl-C when happy, "
          "then copy these values into this camera's entry in config/cameras.yaml.")
    try:
        while True:
            pipe.wait_for_frames(timeout_ms=1000)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
