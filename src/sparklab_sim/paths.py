"""Locate shared assets in the sparklab_yam source tree by path, never by import.

Isaac's bundled Python has neither rclpy nor lerobot. See DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

# src/sparklab_sim/paths.py -> src/
_SRC = Path(__file__).resolve().parent.parent

YAM_ULTRA_MODEL_DIR = _SRC / "sparklab_yam" / "sparklab_yam" / "model"
YAM_ULTRA_URDF = YAM_ULTRA_MODEL_DIR / "yam_ultra.urdf"
YAM_ULTRA_CONFIG_DIR = _SRC / "sparklab_yam" / "config"

# Beside the source model rather than a temp dir: conversion is slow and
# deterministic, so it should survive across runs. Gitignored build artifact.
USD_DIR = YAM_ULTRA_MODEL_DIR / "usd"
# The importer writes <usd_path>/<stem>/<stem>.usda, one level deeper than given.
YAM_ULTRA_USD = USD_DIR / "yam_ultra" / "yam_ultra.usda"


def require(path: Path) -> Path:
    """Return *path*, or raise FileNotFoundError naming the command that builds it."""
    if not path.exists():
        if path == YAM_ULTRA_USD:
            raise FileNotFoundError(
                f"{path} not found — run the URDF->USD conversion first:\n"
                f"    <isaac>/python.sh -m sparklab_sim.convert")
        raise FileNotFoundError(f"missing asset: {path}")
    return path
