"""Locate shared assets in the sparklab_yam source tree by path, never by import.

Isaac's bundled Python has neither rclpy nor lerobot. See DESIGN.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/sparklab_sim/paths.py -> src/
_SRC = Path(__file__).resolve().parent.parent

YAM_ULTRA_MODEL_DIR = _SRC / "sparklab_yam" / "sparklab_yam" / "model"
YAM_ULTRA_URDF = YAM_ULTRA_MODEL_DIR / "yam_ultra.urdf"
YAM_ULTRA_CONFIG_DIR = _SRC / "sparklab_yam" / "config"
# The arm cuRobo plans with, written by capx_sparklab.robot.build.
CAPX_URDF = _SRC / "capx_sparklab" / "capx_sparklab" / "robot" / "yam.urdf"

# Beside the source model rather than a temp dir: conversion is slow and
# deterministic, so it should survive across runs. Gitignored build artifact.
USD_DIR = YAM_ULTRA_MODEL_DIR / "usd"
# The importer writes <usd_path>/<stem>/<stem>.usda, one level deeper than given.
YAM_ULTRA_USD = USD_DIR / "yam_ultra" / "yam_ultra.usda"
TWIN_URDF = USD_DIR / "yam_twin.urdf"
TWIN_USD = USD_DIR / "yam_twin.usd"

BEHAVIOR_1K = Path(os.environ.get("BEHAVIOR_1K", Path.home() / "BEHAVIOR-1K"))
OBJECT_ASSETS = BEHAVIOR_1K / "datasets" / "behavior-1k-assets" / "objects"
# Mass, volume and density per category; the object files carry a placeholder 1 kg.
OBJECT_SPECS = BEHAVIOR_1K / "datasets" / "behavior-1k-assets" / "metadata" / "avg_category_specs.json"
OMNIGIBSON_KEY = BEHAVIOR_1K / "datasets" / "omnigibson.key"
# The MDL shaders the objects use, which import each other by relative path.
OMNIGIBSON_MATERIALS = BEHAVIOR_1K / "OmniGibson" / "omnigibson" / "materials"
OBJECT_CACHE = Path.home() / ".cache" / "sparklab_sim" / "objects"


def require(path: Path) -> Path:
    """Return *path*, or raise FileNotFoundError naming the command that builds it."""
    if not path.exists():
        if path in (YAM_ULTRA_USD, TWIN_USD):
            flag = " --twin" if path == TWIN_USD else ""
            raise FileNotFoundError(
                f"{path} not found — run the URDF->USD conversion first:\n"
                f"    ./scripts/isaac_python.sh -m sparklab_sim.convert{flag}")
        raise FileNotFoundError(f"missing asset: {path}")
    return path
