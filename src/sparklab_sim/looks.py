"""What the twin looks like: colours sampled from the real cameras, and the room's light.

Each colour is the sRGB the real top camera records (frame of 2026-10-14); its
albedo is that value made linear, and the lights are tuned until the twin's table
renders at the real table's brightness. Call these after SimulationApp exists.
"""

from __future__ import annotations

import numpy as np

from . import calibration

# ---- colours, as the real top camera records them (sRGB) ----------------------
ARM_BLACK = (14, 14, 12)
GRIPPER_GREY = (110, 112, 116)
TABLE_GREY = (176, 178, 178)
CARDBOARD = (176, 136, 96)
RAIL_SILVER = (168, 170, 174)
FLOOR = (58, 58, 62)
WALL = (205, 205, 200)

# ---- light, fitted to the real top camera's table (Nelder-Mead, 12/255 mean error) ----
# The real table is lit brighter on the right arm's side, as if by a window there.
AMBIENT_INTENSITY = 18.5
PANEL_INTENSITY = 367.0
PANEL_Z = 2.2
WINDOW_INTENSITY = 3970.0

# ---- fixed things, placed from the top camera's view ---------------------------
# The bases sit near the table's back edge.
TABLE_X = (-0.15, 0.90)
TABLE_Y = (-1.10, 0.55)
TABLE_THICKNESS = 0.04
# x range, y range and wall height of the open box left of the left arm.
CARDBOARD_BOX = ((0.49, 0.74), (0.09, 0.39), 0.15)
# The extrusion the bases are bolted to, along the table's back edge.
RAIL_X = -0.11


def material(stage, name: str, srgb, roughness: float = 0.6, metallic: float = 0.0):
    """A UsdPreviewSurface whose albedo is srgb (0-255) made linear. Made once per name."""
    from pxr import Gf, Sdf, UsdShade

    path = f"/World/Looks/{name}"
    if stage.GetPrimAtPath(path):
        return UsdShade.Material(stage.GetPrimAtPath(path))
    look = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    linear = (np.asarray(srgb, dtype=float) / 255.0) ** 2.2
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return look


def paint(stage, prim_path: str, look) -> None:
    """Bind a look to a prim and everything under it, over whatever it carried."""
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(look, UsdShade.Tokens.strongerThanDescendants)


def paint_arm(stage, arm_path: str) -> None:
    """Black links, a grey gripper body and grey fingers."""
    for link in ("base", "link1", "link2", "link3", "link4", "link5"):
        paint(stage, f"{arm_path}/{link}/visuals", material(stage, "ArmBlack", ARM_BLACK, 0.45))
    # The stock fingers are not sampled from the top camera yet; they take the body's grey.
    for part in ("gripper", "tip_left", "tip_right"):
        paint(stage, f"{arm_path}/{part}/visuals", material(stage, "GripperGrey", GRIPPER_GREY, 0.4, 0.5))


def room(stage) -> None:
    """Floor, walls, ceiling panels and ambient light; the table; the box and the rail on it."""
    import carb
    from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
    from pxr import Gf, UsdLux

    top = calibration.TABLE_TOP_Z
    floor_z = top - 0.75
    paint(stage, FixedCuboid("/World/Floor", position=np.array([0.4, -0.3, floor_z - 0.05]),
                             scale=np.array([10.0, 10.0, 0.1])).prim_path, material(stage, "Floor", FLOOR, 0.8))
    for name, centre, size in (("WallFront", (3.4, -0.3, 0.5), (0.1, 8, 3.5)), ("WallBack", (-2.6, -0.3, 0.5), (0.1, 8, 3.5)),
                               ("WallLeft", (0.4, 3.7, 0.5), (8, 0.1, 3.5)), ("WallRight", (0.4, -4.3, 0.5), (8, 0.1, 3.5))):
        paint(stage, VisualCuboid(f"/World/{name}", position=np.array(centre), scale=np.array(size)).prim_path,
              material(stage, "Wall", WALL, 0.9))

    # Fixed exposure, so the light levels below mean what they meant when fitted.
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    UsdLux.DomeLight.Define(stage, "/World/Ambient").CreateIntensityAttr(AMBIENT_INTENSITY)
    for i, x in enumerate((-0.2, 0.6, 1.4)):
        panel = UsdLux.RectLight.Define(stage, f"/World/Panel{i}")
        panel.CreateIntensityAttr(PANEL_INTENSITY)
        panel.CreateWidthAttr(0.6)
        panel.CreateHeightAttr(1.2)
        panel.AddTranslateOp().Set(Gf.Vec3d(x, -0.3, PANEL_Z))  # a RectLight shines down its -z
    window = UsdLux.RectLight.Define(stage, "/World/Window")
    window.CreateIntensityAttr(WINDOW_INTENSITY)
    window.CreateWidthAttr(2.5)
    window.CreateHeightAttr(1.5)
    window.AddTranslateOp().Set(Gf.Vec3d(0.4, -2.2, 0.9))
    window.AddRotateXOp().Set(90.0)  # turns its -z to +y, toward the table

    table = FixedCuboid("/World/Table", position=np.array([np.mean(TABLE_X), np.mean(TABLE_Y), top - TABLE_THICKNESS / 2]),
                        scale=np.array([np.ptp(TABLE_X), np.ptp(TABLE_Y), TABLE_THICKNESS]))
    paint(stage, table.prim_path, material(stage, "Table", TABLE_GREY, 0.75))
    paint(stage, FixedCuboid("/World/Rail", position=np.array([RAIL_X, np.mean(TABLE_Y), top + 0.02]),
                             scale=np.array([0.04, np.ptp(TABLE_Y), 0.04])).prim_path,
          material(stage, "Rail", RAIL_SILVER, 0.35, 0.8))
    _cardboard_box(stage)


def _cardboard_box(stage) -> None:
    """An open box: a floor and four 5 mm walls."""
    from isaacsim.core.api.objects import FixedCuboid

    (x0, x1), (y0, y1), height = CARDBOARD_BOX
    top, wall = calibration.TABLE_TOP_Z, 0.005
    look = material(stage, "Cardboard", CARDBOARD, 0.9)
    panels = {
        "Bottom": ((x0 + x1) / 2, (y0 + y1) / 2, wall / 2, x1 - x0, y1 - y0, wall),
        "Front": (x0, (y0 + y1) / 2, height / 2, wall, y1 - y0, height),
        "Back": (x1, (y0 + y1) / 2, height / 2, wall, y1 - y0, height),
        "Left": ((x0 + x1) / 2, y1, height / 2, x1 - x0, wall, height),
        "Right": ((x0 + x1) / 2, y0, height / 2, x1 - x0, wall, height),
    }
    for name, (x, y, z, sx, sy, sz) in panels.items():
        panel = FixedCuboid(f"/World/CardboardBox/{name}", position=np.array([x, y, top + z]), scale=np.array([sx, sy, sz]))
        paint(stage, panel.prim_path, look)
