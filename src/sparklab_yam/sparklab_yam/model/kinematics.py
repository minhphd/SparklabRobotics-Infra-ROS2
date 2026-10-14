"""MuJoCo model construction for the I2RT YAM-Ultra arm.

Combines the vendored arm and gripper MJCF (both copied byte-for-byte from
i2rt v1.2.4) and adds the two sites the IK pipeline needs: ``tool0`` at the
gripper's grasp centre, and ``j4_anchor`` on link3, the wrist-invariant
position-task anchor. The IK math lives in ``ik/decoupled_ik.py``.

JOINT LIMITS. Everything matches i2rt except ``INTENTIONAL_DEVIATIONS`` —
joint3's upper limit is 3.0 rad here against i2rt's nominal π. That is
measured on this rig, not drift; do not "fix" it back. The IK clamps into
these limits and derives its limit-pressure haptic from them, so both a too
tight and a too loose value fail quietly. Re-check after any i2rt upgrade::

    python -m sparklab_yam.model.kinematics
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent

ARM_XML = _HERE / "yam_ultra.xml"
GRIPPER_XML = _HERE / "linear_4310.xml"
# Gripper already assembled; for USD/sim importers only. Nothing in the
# control path loads it — the IK and viewer use the MJCF above.
ARM_URDF = _HERE / "yam_ultra.urdf"

# Wrist-mount transform, from i2rt's robots/config/linear_4310.yml. A no-op
# against v1.2.4+, which bakes it into link6; kept for older arm XMLs.
LINK6_MOUNT_POS = "0.0404996 2.39858e-07 -0.0419481"
LINK6_MOUNT_QUAT = "1 0 0 0"
JOINT6_AXIS = "0 0 -1"

# Grasp-centre offset in the gripper body frame, matching the gripper XML's
# grasp_site. The quat flips z to point out along the fingers.
TOOL0_OFFSET_XYZ = np.array([0.0, 0.0, -0.1347])
TOOL0_OFFSET_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

DEFAULT_Q_REST = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])

# joint name -> ((lower, upper), why). _check_against_i2rt() reports these as
# expected rather than drift, so the check stays meaningful.
INTENTIONAL_DEVIATIONS: dict[str, tuple[tuple[float, float], str]] = {
    "joint3": ((0.0, 3.0),
               "elbow stops near 3.0 rad on this rig; i2rt's nominal 3.14159 "
               "(pi, corrected in v1.2.4 — see module docstring) would let "
               "the IK drive it into the mechanical stop"),
    # Same physical stop in the URDF's naming; both vendored assets carry it.
    "dof_joint3": ((0.0, 3.0),
                   "URDF counterpart of the joint3 deviation above"),
}


def combine_arm_and_gripper_xml(
    arm_xml: str | Path = ARM_XML,
    gripper_xml: str | Path = GRIPPER_XML,
) -> str:
    """Combine the arm and gripper MJCF fragments into one XML string.

    Self-contained port of i2rt's ``combine_arm_and_gripper_xml``: sets link6's
    mount transform and joint6's axis, merges mesh assets, grafts the gripper
    subtree, and carries over its equality/contact sections.
    """
    arm_xml, gripper_xml = Path(arm_xml), Path(gripper_xml)
    arm_root = ET.parse(arm_xml).getroot()
    grip_root = ET.parse(gripper_xml).getroot()

    link6 = arm_root.find(".//body[@name='link6']")
    if link6 is None:
        raise RuntimeError("link6 body not found in arm XML")
    link6.set("pos", LINK6_MOUNT_POS)
    link6.set("quat", LINK6_MOUNT_QUAT)
    joint6 = link6.find("joint")
    if joint6 is None:
        raise RuntimeError("joint6 not found on link6")
    joint6.set("axis", JOINT6_AXIS)

    # Absolutize mesh paths, since the string is compiled from another cwd.
    for root, xml_path in ((arm_root, arm_xml), (grip_root, gripper_xml)):
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        asset = root.find("asset")
        if asset is not None:
            for child in asset:
                file = child.get("file")
                if file and not os.path.isabs(file):
                    child.set("file", str((xml_path.parent / meshdir / file).resolve()))
        if compiler is not None and compiler.get("meshdir"):
            del compiler.attrib["meshdir"]

    # Merge gripper assets into the arm's asset block (skip duplicates).
    arm_asset = arm_root.find("asset")
    grip_asset = grip_root.find("asset")
    if grip_asset is not None:
        if arm_asset is None:
            arm_asset = ET.Element("asset")
            arm_root.insert(0, arm_asset)
        existing = {(c.tag, c.get("name")) for c in arm_asset}
        for child in grip_asset:
            if (child.tag, child.get("name")) not in existing:
                arm_asset.append(deepcopy(child))

    grip_body = grip_root.find(".//body[@name='gripper']")
    if grip_body is None:
        raise RuntimeError("gripper body not found in gripper XML")
    link6.append(deepcopy(grip_body))

    for section_tag in ("equality", "contact"):
        grip_section = grip_root.find(section_tag)
        if grip_section is None:
            continue
        arm_section = arm_root.find(section_tag)
        if arm_section is None:
            arm_section = ET.SubElement(arm_root, section_tag)
        for child in grip_section:
            arm_section.append(deepcopy(child))

    return ET.tostring(arm_root, encoding="unicode")


def build_spec_with_tool0_site(
    arm_xml: str | Path | None = None,
    gripper_xml: str | Path | None = None,
) -> mujoco.MjSpec:
    """The combined arm+gripper spec with the two IK sites, not yet compiled.

    Split out so tooling can decorate the model before compiling without
    touching the actuator-free spec the IK solves against.
    """
    xml = combine_arm_and_gripper_xml(
        arm_xml or ARM_XML, gripper_xml or GRIPPER_XML
    )
    spec = mujoco.MjSpec.from_string(xml)

    gripper = spec.body("gripper")
    if gripper is None:
        raise RuntimeError("gripper body missing after combine")
    gripper.add_site(
        name="tool0",
        pos=TOOL0_OFFSET_XYZ.tolist(),
        quat=TOOL0_OFFSET_QUAT.tolist(),
    )
    # On link3, upstream of joint 4 and so wrist-invariant: 10 cm past joint 4's
    # pivot along link3→link4. Recomputed for i2rt v1.2.4's body frames.
    link3 = spec.body("link3")
    if link3 is None:
        raise RuntimeError("link3 body not found in arm XML")
    link3.add_site(
        name="j4_anchor",
        pos=[-0.0830, 0.0938, -0.3388],
        size=[0.015, 0.0, 0.0],     # 1.5 cm sphere
        rgba=[1.0, 0.5, 0.0, 1.0],  # orange
    )
    return spec


def build_model_with_tool0_site(
    arm_xml: str | Path | None = None,
    gripper_xml: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_spec_with_tool0_site(arm_xml, gripper_xml)
    model = spec.compile()
    return model, mujoco.MjData(model)


# ---------- self-test ----------

def _joint_ranges(xml_path: Path) -> dict[str, tuple[float, float]]:
    """joint name -> (lower, upper) as written in an MJCF file."""
    out: dict[str, tuple[float, float]] = {}
    for j in ET.parse(xml_path).getroot().iter("joint"):
        name, rng = j.get("name"), j.get("range")
        if name and rng:
            lo, hi = (float(v) for v in rng.split())
            out[name] = (lo, hi)
    return out


def _urdf_joint_ranges(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """joint name -> (lower, upper) as written in a URDF file."""
    out: dict[str, tuple[float, float]] = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        name, lim = j.get("name"), j.find("limit")
        if name and lim is not None:
            lo, hi = lim.get("lower"), lim.get("upper")
            if lo is not None and hi is not None:
                out[name] = (float(lo), float(hi))
    return out


# Fingerprint of i2rt's pre-v1.2.4 URDF, which had joint2/joint3's limits
# swapped. An older install differs on every joint, so detect it and say so.
_PRE_V124_JOINT3_UPPER = 3.66519


def _check_vendored_assets_agree() -> int:
    """Check the vendored MJCF and URDF describe the same arm.

    Independent of i2rt on purpose, so the invariant holds on a machine that
    never had it installed.

    Returns: number of disagreements found, 0 if they agree.
    """
    mjcf, urdf = _joint_ranges(ARM_XML), _urdf_joint_ranges(ARM_URDF)
    problems: list[str] = []
    for i in range(1, 7):
        mj, ur = f"joint{i}", f"dof_joint{i}"
        o, u = mjcf.get(mj), urdf.get(ur)
        if o is None or u is None:
            problems.append(f"{mj}/{ur}: missing from one vendored asset ({o=}, {u=})")
        elif not all(abs(a - b) < 1e-6 for a, b in zip(o, u)):
            problems.append(f"{mj} {o} (MJCF) disagrees with {ur} {u} (URDF)")

    if problems:
        print(f"WARNING: vendored MJCF and URDF disagree ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        print("  Both must carry the same limits — see INTENTIONAL_DEVIATIONS.")
    else:
        print("vendored MJCF and URDF agree on all 6 joint ranges")
    return len(problems)


def _check_against_i2rt() -> None:
    """Diff the vendored MJCF against i2rt's installed copy, if importable.

    Separates ``INTENTIONAL_DEVIATIONS`` from genuine drift. Skipped when i2rt
    isn't installed; never raises — it reports, you decide.
    """
    try:
        import i2rt
    except ImportError:
        print("i2rt not installed — skipping vendored-model drift check")
        return

    root = Path(i2rt.__file__).resolve().parent / "robot_models"
    their_arm = root / "arm" / "yam_ultra" / "yam_ultra.xml"

    # Against a pre-v1.2.4 install essentially everything differs, and a check
    # that always warns is a check nobody reads. Compare only what still can be.
    their_predates_v124 = False
    if their_arm.exists():
        j3 = _joint_ranges(their_arm).get("joint3")
        their_predates_v124 = bool(j3 and abs(j3[1] - _PRE_V124_JOINT3_UPPER) < 1e-6)

    # No intentional deviations in these, so byte-compare.
    pairs = [(GRIPPER_XML, root / "gripper" / "linear_4310" / "linear_4310.xml")]
    for name in ("base", "link1", "link2", "link3", "link4", "link5"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "arm" / "yam_ultra" / "assets" / f"{name}.stl"))
    for name in ("gripper", "tip_left", "tip_right"):
        pairs.append((_HERE / "assets" / f"{name}.stl",
                      root / "gripper" / "linear_4310" / "assets" / f"{name}.stl"))

    drift, expected = [], []
    if their_predates_v124:
        expected.append(
            "installed i2rt predates v1.2.4 — it regenerated the meshes and fixed "
            "the swapped joint2/joint3 limits, so the vendored v1.2.4 assets are "
            "EXPECTED to differ throughout. Skipping mesh and joint-range "
            "comparisons; re-run after upgrading i2rt to make this check useful "
            "again.")
    else:
        for ours, theirs in pairs:
            if not theirs.exists():
                drift.append(f"{ours.name}: no counterpart at {theirs}")
            elif ours.read_bytes() != theirs.read_bytes():
                drift.append(f"{ours.name}: differs from {theirs}")

    # Compared range by range, so an intentional limit change can't mask an
    # unrelated edit elsewhere in the file.
    if their_predates_v124:
        pass
    elif not their_arm.exists():
        drift.append(f"yam_ultra.xml: no counterpart at {their_arm}")
    else:
        ours_r, theirs_r = _joint_ranges(ARM_XML), _joint_ranges(their_arm)
        for name in sorted(set(ours_r) | set(theirs_r)):
            o, t = ours_r.get(name), theirs_r.get(name)
            if o is None or t is None:
                drift.append(f"joint {name}: present in only one model ({o=}, {t=})")
                continue
            if all(abs(a - b) < 1e-6 for a, b in zip(o, t)):
                continue
            want = INTENTIONAL_DEVIATIONS.get(name)
            if want and all(abs(a - b) < 1e-6 for a, b in zip(o, want[0])):
                expected.append(f"joint {name}: {o} vs i2rt {t} — {want[1]}")
            else:
                drift.append(f"joint {name}: {o} vs i2rt {t}"
                             + (f" (expected {want[0]})" if want else ""))
        if ARM_XML.read_bytes() != their_arm.read_bytes() and not (
                set(ours_r.items()) ^ set(theirs_r.items())):
            drift.append("yam_ultra.xml: differs from i2rt outside the joint ranges")

    # By joint range only: the URDF carries the assembled gripper and its own
    # dof_joint* naming, so a byte-compare would differ without being useful.
    their_urdf = root / "arm" / "yam_ultra" / "yam_ultra.urdf"
    if their_predates_v124:
        pass  # already reported once above, for both assets
    elif not their_urdf.exists():
        drift.append(f"yam_ultra.urdf: no counterpart at {their_urdf}")
    else:
        ours_u, theirs_u = _urdf_joint_ranges(ARM_URDF), _urdf_joint_ranges(their_urdf)
        for name in sorted(set(ours_u) | set(theirs_u)):
            o, t = ours_u.get(name), theirs_u.get(name)
            if o is None or t is None:
                drift.append(f"urdf joint {name}: present in only one model ({o=}, {t=})")
                continue
            if all(abs(a - b) < 1e-6 for a, b in zip(o, t)):
                continue
            want = INTENTIONAL_DEVIATIONS.get(name)
            if want and all(abs(a - b) < 1e-6 for a, b in zip(o, want[0])):
                expected.append(f"urdf joint {name}: {o} vs i2rt {t} — {want[1]}")
            else:
                drift.append(f"urdf joint {name}: {o} vs i2rt {t}"
                             + (f" (expected {want[0]})" if want else ""))

    for e in expected:
        print(f"expected deviation — {e}")
    if drift:
        print(f"WARNING: {len(drift)} unexpected difference(s) from i2rt:")
        for d in drift:
            print(f"  - {d}")
        print("  If unintended, re-copy from i2rt (see the module docstring).")
    elif their_predates_v124:
        print("vendored model NOT verified against i2rt — install predates v1.2.4 "
              "(see above). The MJCF/URDF agreement check above still applies.")
    else:
        print(f"vendored model matches i2rt "
              f"({len(pairs)} files + all joint ranges checked, "
              f"{len(expected)} intentional deviation(s))")


def _self_test() -> None:
    """Compile the model and verify the wrist geometry and 3+3 decoupling."""
    _check_vendored_assets_agree()
    _check_against_i2rt()
    model, data = build_model_with_tool0_site()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "j4_anchor")
    assert -1 not in (site_id, anchor_id), "required site missing"
    print(f"compiled: nq={model.nq} (6 arm + 2 coupled finger slides)")

    data.qpos[:6] = DEFAULT_Q_REST
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    print(f"tool0 at rest: {data.site_xpos[site_id]}")

    # Must be independent: without the mount transform joints 5/6 are coaxial.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    det_rot = float(np.linalg.det(jacr[:, 3:6]))
    mujoco.mj_jacSite(model, data, jacp, jacr, anchor_id)
    det_pos = float(np.linalg.det(jacp[:, :3]))
    print(f"det(J_rot wrist) = {det_rot:.4f}   det(J_pos joints1-3) = {det_pos:.4f}")
    assert abs(det_rot) > 0.5, "wrist axes not independent — mount transform broken?"
    assert abs(det_pos) > 1e-3, "rest pose is position-singular"

    # Joints 5 and 6 must intersect; joint 4 sits a known offset upstream.
    def joint_line(j: int) -> tuple[np.ndarray, np.ndarray]:
        bid = model.jnt_bodyid[j]
        R = data.xmat[bid].reshape(3, 3)
        return data.xpos[bid] + R @ model.jnt_pos[j], R @ model.jnt_axis[j]

    p5, a5 = joint_line(4)
    p6, a6 = joint_line(5)
    cross = np.cross(a5, a6)
    gap56 = abs(float((p6 - p5) @ cross / np.linalg.norm(cross)))
    # Wrist centre = the j5/j6 axes' near-intersection point.
    t5 = float(np.cross(p6 - p5, a6) @ cross / (cross @ cross))
    wrist_center = p5 + t5 * a5
    p4, a4 = joint_line(3)
    v = wrist_center - p4
    off4 = float(np.linalg.norm(v - (v @ a4) * a4))
    print(f"joint5/6 axis gap = {gap56 * 1000:.2f} mm   "
          f"joint4 offset from wrist center = {off4 * 1000:.1f} mm")
    assert gap56 < 1e-3, "joints 5/6 do not intersect"


if __name__ == "__main__":
    _self_test()
