"""Convert the YAM-Ultra URDF to USD. Run once; the result is cached.

    ./scripts/isaac_python.sh -m sparklab_sim.convert           # stock gripper, posed only
    ./scripts/isaac_python.sh -m sparklab_sim.convert --twin    # the twin's arm, with collisions

Boots a headless SimulationApp because the URDF importer is a Kit extension,
which is also why every Isaac import here sits inside ``main()``.

The stock source is the vendored URDF, not i2rt's installed copy: that clone
predates v1.2.4 and still has the swapped joint2/joint3 limits. The twin's
source is capx_sparklab's ``yam.urdf``, the one cuRobo plans with.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

from . import paths

# cuRobo's frames in yam.urdf; the twin publishes tool0 as the driver defines it instead.
PLANNING_FRAMES = {"world", "grasp_base", "tool0"}


def write_twin_urdf() -> None:
    """capx_sparklab's yam.urdf without the planning frames, with absolute mesh paths."""
    root = ET.parse(paths.require(paths.CAPX_URDF)).getroot()
    for link in [l for l in root.findall("link") if l.get("name") in PLANNING_FRAMES]:
        root.remove(link)
    for joint in root.findall("joint"):
        if {joint.find("parent").get("link"), joint.find("child").get("link")} & PLANNING_FRAMES:
            root.remove(joint)
    for mesh in root.iter("mesh"):
        mesh.set("filename", str(paths.CAPX_URDF.parent / mesh.get("filename")))
    paths.TWIN_URDF.write_text(ET.tostring(root, encoding="unicode"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--twin", action="store_true", help="the arm the twin simulates, with collisions")
    ap.add_argument("--force", action="store_true", help="reconvert even if the USD already exists")
    args = ap.parse_args()

    out = paths.TWIN_USD if args.twin else paths.YAM_ULTRA_USD
    if out.exists() and not args.force:
        print(f"already converted: {out}\n(use --force to redo)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.twin:
        write_twin_urdf()
    urdf = paths.TWIN_URDF if args.twin else paths.require(paths.YAM_ULTRA_URDF)

    # --- Isaac boots here; nothing omni/isaacsim may be imported above ------
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.kit.commands
        from isaacsim.asset.importer.urdf import _urdf

        _, config = omni.kit.commands.execute("URDFCreateImportConfig")
        # Bolted to the table; a floating base would fall the moment physics ran.
        config.set_fix_base(True)
        # Merging fixed joints erases link frames, which the cameras hang from.
        config.set_merge_fixed_joints(False)
        config.set_make_default_prim(True)
        config.set_create_physics_scene(False)
        config.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
        # The URDF has visual meshes only; the twin needs them to collide, for grasping.
        # Decomposed, not hulled: one hull per link would fill its concave parts.
        config.set_collision_from_visuals(args.twin)
        config.set_convex_decomp(args.twin)
        config.set_self_collision(False)
        print(f"converting {urdf.name} -> {out}", flush=True)
        omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf),
                                  import_config=config, dest_path=str(out))
        print(f"wrote {out}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
