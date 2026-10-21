"""BEHAVIOR-1K objects for the twin's table, loaded without OmniGibson.

The dataset's USDs are encrypted. Each is decrypted once into OBJECT_CACHE under
the same <category>/<model>/usd/ layout, with material/ and OmniGibson's MDL
shaders linked beside it so the relative texture and shader paths still resolve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths


@dataclass
class Prop:
    """An object on the table: what it is, and where it starts in left_base."""

    name: str
    category: str
    xy: tuple[float, float]
    yaw_deg: float = 0.0
    model: str | None = None  # None: the category's first model
    size: np.ndarray = field(default=None, repr=False)  # bounding box, filled in by add()


def _model(category: str, model: str | None) -> Path:
    folder = paths.OBJECT_ASSETS / category
    if not folder.is_dir():
        raise FileNotFoundError(f"no BEHAVIOR-1K category {category!r} under {paths.OBJECT_ASSETS}")
    return folder / (model or sorted(p.name for p in folder.iterdir())[0])


def usd(category: str, model: str | None = None) -> Path:
    """The decrypted USD of one object, decrypting it the first time."""
    from cryptography.fernet import Fernet

    source = _model(category, model)
    cached = paths.OBJECT_CACHE / category / source.name
    out = cached / "usd" / f"{source.name}.usd"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        key = paths.OMNIGIBSON_KEY.read_bytes()
        encrypted = (source / "usd" / f"{source.name}.encrypted.usd").read_bytes()
        out.write_bytes(Fernet(key).decrypt(encrypted))
        (cached / "material").symlink_to(source / "material")
    for shader in paths.OMNIGIBSON_MATERIALS.glob("*.mdl"):
        link = out.parent / shader.name
        if not link.exists():
            link.symlink_to(shader)
    return out


def size(category: str, model: str | None = None) -> np.ndarray:
    """The object's bounding box (x, y, z) in metres, from the dataset's metadata."""
    metadata = json.loads((_model(category, model) / "misc" / "metadata.json").read_text())
    return np.asarray(metadata["bbox_size"], dtype=float)


def set_pose(prim, T: np.ndarray) -> None:
    """Replace a prim's local transform with the 4x4 T, whatever ops it carried."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    # USD multiplies row vectors, so its matrices are the transpose of ours.
    xform.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))


def add(stage, prop: Prop, table_top_z: float, root: str = "/World/Objects") -> str:
    """Reference one object onto the table, resting on it at prop.xy. Returns its prim path."""
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    prop.size = size(prop.category, prop.model)
    path = f"{root}/{prop.name}"
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(usd(prop.category, prop.model)))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", prop.yaw_deg, degrees=True).as_matrix()
    # A millimetre of drop, so it starts clear of the table and settles onto it.
    T[:3, 3] = [*prop.xy, table_top_z + prop.size[2] / 2 + 0.001]
    set_pose(prim, T)
    UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{path}/base_link")).GetMassAttr().Set(mass(prop.category))
    return path


def mass(category: str) -> float:
    """The category's average mass in kg, from the dataset's metadata."""
    return float(json.loads(paths.OBJECT_SPECS.read_text())[category]["mass"])
