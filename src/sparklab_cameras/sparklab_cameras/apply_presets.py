"""Push each camera's RealSense advanced-mode JSON preset onto the hardware.

Reads ``advanced_json`` from a rig's cameras.yaml, resolved beside it. Run once
per power-up, before the camera node — the settings persist in the device's own
memory until it loses power::

    ros2 run sparklab_cameras apply_presets src/sparklab_yam/config/cameras.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("apply_camera_presets")

def _find_device(ctx, serial: str):
    import pyrealsense2 as rs

    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return dev
    return None


def apply(serial: str, json_path: Path, *, timeout_s: float = 8.0) -> None:
    import pyrealsense2 as rs

    dev = _find_device(rs.context(), serial)
    if dev is None:
        raise RuntimeError(f"realsense device {serial} not found (unplugged, or held by another process)")

    adv = rs.rs400_advanced_mode(dev)
    if not adv.is_enabled():
        logger.info("%s: enabling advanced mode (device will briefly reset)", serial)
        adv.toggle_advanced_mode(True)
        deadline = time.time() + timeout_s
        dev = None
        while dev is None and time.time() < deadline:
            time.sleep(0.5)
            dev = _find_device(rs.context(), serial)
        if dev is None:
            raise RuntimeError(f"realsense device {serial} did not reappear after enabling advanced mode")
        adv = rs.rs400_advanced_mode(dev)

    adv.load_json(json_path.read_text())
    logger.info("%s: applied %s", serial, json_path.name)


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rig", type=Path, help="the rig's cameras.yaml")
    rig = ap.parse_args().rig
    data = yaml.safe_load(rig.read_text()) or {}
    applied = 0
    for cam_id, cfg in (data.get("cameras") or {}).items():
        preset = cfg.get("advanced_json")
        serial = cfg.get("serial")
        if not preset:
            continue
        if not serial:
            logger.warning("%s: has advanced_json but no serial, skipping", cam_id)
            continue
        path = Path(preset)
        if not path.is_absolute():
            path = rig.parent / path
        logger.info("%s (serial %s): applying %s", cam_id, serial, path)
        apply(serial, path)
        applied += 1

    if applied == 0:
        logger.warning("no camera in %s has an advanced_json preset set", rig)


if __name__ == "__main__":
    main()
