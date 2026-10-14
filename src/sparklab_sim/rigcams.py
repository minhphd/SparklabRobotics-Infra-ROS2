"""Sim cameras matching the real rig's resolution and field of view.

Reads ``sparklab_yam/config/{cameras,intrinsics}.yaml``, the same files the
camera node and the follower use. FOV is derived from the measured ``fx``.

The wrist camera POSES are guesses: they ride each arm's link6, but that mount
transform is an un-done hand-eye calibration and the URDF has no camera links.
Every wrist render is the right lens in the wrong place.
"""

from __future__ import annotations

import numpy as np
import yaml

from . import paths, scene

# USD's default horizontal aperture (mm). Arbitrary but must stay consistent
# with the focal length computed against it — only the ratio matters.
_H_APERTURE_MM = 20.955

# ASSUMED, in the gripper link's frame: above and behind the grasp point,
# looking along the approach. Replace with a hand-eye calibration.
_WRIST_MOUNT_XYZ = (0.04, 0.0, -0.03)
_WRIST_MOUNT_RPY_DEG = (0.0, 25.0, 0.0)


def load_specs() -> dict:
    """``{camera_id: {...}}`` merging cameras.yaml with intrinsics.yaml.

    Rescales ``fx``/``fy`` when the intrinsics were exported at a different
    resolution than the camera is captured at.
    """
    cams = yaml.safe_load(paths.YAM_ULTRA_CONFIG_DIR.joinpath("cameras.yaml")
                          .read_text())["cameras"]
    intr = yaml.safe_load(paths.YAM_ULTRA_CONFIG_DIR.joinpath("intrinsics.yaml")
                          .read_text())["cameras"]
    by_serial = {str(v["serial_number"]): v for v in intr.values()}

    out = {}
    for cid, c in cams.items():
        w, h = int(c["width"]), int(c["height"])
        out_h = int(c.get("crop_height") or h)
        entry = {"model": c["model"], "serial": str(c["serial"]),
                 "capture": (w, h), "out": (w, out_h),
                 "fx": None, "fy": None, "source": "none"}
        col = (by_serial.get(str(c["serial"]), {}).get("streams", {}) or {}).get("color")
        if col:
            iw = int(col["resolution"]["width"])
            s = w / iw                      # rescale to the captured width
            entry.update(fx=float(col["focal_length"]["fx"]) * s,
                         fy=float(col["focal_length"]["fy"]) * s,
                         source=f"intrinsics.yaml ({iw}px -> {w}px, x{s:.4f})")
        out[cid] = entry
    return out


def _apply_intrinsics(cam, fx: float, fy: float, width: int, height: int) -> None:
    """Set focal length and apertures so the USD camera's FOV matches fx/fy."""
    focal_mm = fx * _H_APERTURE_MM / width
    # Vertical aperture from the pixel aspect, so a non-square frame is not
    # silently stretched.
    v_aperture = _H_APERTURE_MM * (height / width) * (fx / fy)
    cam.CreateFocalLengthAttr(float(focal_mm))
    cam.CreateHorizontalApertureAttr(float(_H_APERTURE_MM))
    cam.CreateVerticalApertureAttr(float(v_aperture))


def add_rig_cameras(stage, specs: dict | None = None) -> dict:
    """Create sim cameras for every camera in the rig config.

    ``top`` is placed from ``rig.py``; the wrist cameras are parented to their
    arm's gripper link so they follow FK, with a placeholder mount.

    Returns: ``{camera_id: {"path": prim_path, "resolution": (w, h)}}``.
    """
    from pxr import Gf, UsdGeom

    specs = specs or load_specs()
    made = {}

    for cid, spec in specs.items():
        w, h = spec["out"]
        if cid == "top":
            cam = scene.add_top_camera()
            path = scene.CAMERA_PRIM
        else:
            arm = scene.LEFT_PRIM if cid.startswith("left") else scene.RIGHT_PRIM
            parent = scene.link_path(arm, "gripper")
            path = f"{parent}/{cid}_cam"
            cam = UsdGeom.Camera.Define(stage, path)
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*_WRIST_MOUNT_XYZ))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*_WRIST_MOUNT_RPY_DEG))

        if spec["fx"]:
            _apply_intrinsics(cam, spec["fx"], spec["fy"], w, h)
        made[cid] = {"path": path, "resolution": (w, h)}

    return made


def summary(specs: dict | None = None) -> str:
    specs = specs or load_specs()
    lines = ["rig cameras (sim)"]
    for cid, s in specs.items():
        w, h = s["out"]
        if s["fx"]:
            hfov = np.rad2deg(2 * np.arctan(w / (2 * s["fx"])))
            lines.append(f"  {cid:12s} {s['model']:5s} {w}x{h}  "
                         f"fx={s['fx']:.1f}  hfov={hfov:.1f}deg  [{s['source']}]")
        else:
            lines.append(f"  {cid:12s} {s['model']:5s} {w}x{h}  NO INTRINSICS")
    lines.append("  NOTE wrist camera mounts are PLACEHOLDERS (no hand-eye calibration)")
    return "\n".join(lines)
