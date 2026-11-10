"""Fast pick skill (few commands: each command() costs ~2 s of settling).

usage: fast_pick.py ARM X Y [--yaw DEG] [--max-width M] [--wide M] [--radius M]
                    [--lift M] [--place-back] [--home]
  X, Y: coarse object position in ARM's base frame; --yaw: coarse object axis yaw (deg).

1 look from above (vertical fingers, pre-rotated) -> localise (3 frames) -> [re-aim if needed]
-> 1-2 cmd lag-compensated descent [-> 1 correction if > 5 mm] -> close -> 1 cmd lift -> held check
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rig-toolkit" / "scripts"))
from skill_core import *
from grasp_lib import localise

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("x", type=float)
ap.add_argument("y", type=float)
ap.add_argument("--yaw", type=float, default=None)
ap.add_argument("--max-width", type=float, default=0.055)
ap.add_argument("--wide", type=float, default=0.05)
ap.add_argument("--radius", type=float, default=0.08)
ap.add_argument("--lift", type=float, default=0.10)
ap.add_argument("--place-back", action="store_true")
ap.add_argument("--home", action="store_true")
ap.add_argument("--tag", default="fp")
A = ap.parse_args()
arm = A.arm
TZ = -0.031
LAG_RADIAL, LAG_Z = 0.008, 0.008  # measured: arm stops ~1 cm short (high) and ~1 cm too far out


def close_yaw(axis_yaw):
    return wrap_close_yaw(axis_yaw + np.pi / 2)


# 1. look from above, already rotated to the prior yaw
cy = close_yaw(np.radians(A.yaw)) if A.yaw is not None else np.pi / 2
R = topdown_R(cy)
tilted = False
q = None
for h_look in (0.11, 0.09, 0.075, 0.06, 0.05):
    try:
        q = solve(tool0_for_tips([A.x, A.y, TZ + h_look], R), R, REST)
        break
    except ValueError:
        pass
if q is None:  # beyond vertical reach at look height: 30 deg tilted look, as in the first skill
    from grasp_lib import grasp_R
    tilted, h_look = True, 0.11
    R_look = grasp_R([A.x, A.y], np.arctan2(A.y, A.x), np.radians(30))
    q = solve(tool0_for_tips([A.x, A.y, TZ + h_look], R_look), R_look, REST)
fast_move(arm, q, grip=1.0, min_tip_z=TZ + min(h_look - 0.025, 0.04),
          label=f"look ({'tilted 30 deg, ' if tilted else ''}tips {h_look*100:.0f} cm up)")

# 2. localise
res = localise(arm, [A.x, A.y], radius=A.radius, tag=f"{A.tag}_look", hmax=h_look - 0.015, nframes=3)
if not res["found"]:
    raise SystemExit("object not found")
tz = res["table_z"]
g, ayaw, top = grasp_point(res, A.max_width, A.wide)
if np.hypot(*(g - [A.x, A.y])) > 0.03:  # object was off-centre in the view: look again over it
    Rl = R_look if tilted else R
    q = solve(tool0_for_tips([g[0], g[1], tz + h_look], Rl), Rl, get_joint_angles(arm)[:6])
    fast_move(arm, q, min_tip_z=tz + min(h_look - 0.025, 0.04), label="re-look")
    res = localise(arm, g, radius=A.radius, tag=f"{A.tag}_look2", hmax=h_look - 0.015, nframes=3)
    tz = res["table_z"]
    g, ayaw, top = grasp_point(res, A.max_width, A.wide)

def grasp_frame(g, ayaw, tilt):
    """Fingers tilted `tilt` from vertical, leaning outward ALONG the object axis, so the closing
    axis stays horizontal and perpendicular to the object (tilt 0 = straight down)."""
    a = np.array([np.cos(ayaw), np.sin(ayaw), 0.0])
    if a @ np.r_[g, 0] < 0:
        a = -a
    z = np.sin(tilt) * a + np.cos(tilt) * np.array([0, 0, -1.0])
    y = np.cross(z, a)
    y /= np.linalg.norm(y)
    yr = np.arctan2(g[1], g[0])
    if y @ np.array([-np.sin(yr), np.cos(yr), 0]) < 0:  # nearest the natural wrist roll
        y = -y
    return np.c_[np.cross(y, z), y, z]


h_grasp0 = float(np.clip(0.3 * top, 0.007, 0.02))
cur = tips_now(arm)
dyaw = np.degrees(np.arccos(np.clip(abs(np.cos(close_yaw(ayaw) - cy)), -1, 1)))
plan = None
for tilt_deg in (0, 20, 30, 40):  # smallest tilt for which pre-grasp and grasp are reachable
    t = np.radians(tilt_deg)
    R = grasp_frame(g, ayaw, t)
    edge = 0.02 * np.sin(t)          # a tilted 4 cm-wide finger plate dips this far below the tip
    h_grasp = max(h_grasp0, edge + 0.004)
    try:
        qg = solve(tool0_for_tips([g[0], g[1], tz + h_grasp], R), R, get_joint_angles(arm)[:6])
    except ValueError:
        continue
    for h in np.arange(max(top + 0.03, 0.05) + edge, top + 0.0149 + edge, -0.005):
        try:
            plan = (tilt_deg, R, h_grasp, h,
                    solve(tool0_for_tips([g[0], g[1], tz + h], R), R, get_joint_angles(arm)[:6]))
            break
        except ValueError:
            pass
    if plan:
        break
if plan is None:
    raise SystemExit("no reachable pre-grasp/grasp pair up to 40 deg tilt")
tilt_deg, R, h_grasp, h_pre, q = plan
stamp(f"grasp plan: tilt {tilt_deg} deg along the object axis, pre-grasp tips {h_pre*100:.1f} cm, "
      f"grasp tips {h_grasp*100:.1f} cm above table")

# 3. re-aim only if needed (tilted look/grasp, big rotation or offset), at a height clear of the object
if tilted or tilt_deg or dyaw > 10 or np.hypot(*(cur[:2] - g)) > 0.01:
    fast_move(arm, q, min_tip_z=tz + top + 0.01,
              label=f"re-aim (dyaw {dyaw:.0f} deg, tips {h_pre*100:.1f} cm up)")

# 4. descend with the measured lag pre-compensated, one correction if still > 5 mm off
target = np.array([g[0], g[1], tz + h_grasp])
radial = np.r_[g / np.linalg.norm(g), 0]
# lag grows with reach: ~0 at r=0.36 m, ~1-2 cm at r=0.43 m. Under-compensate (never overshoot),
# so any correction continues downward instead of fighting stiction upward.
lag = float(np.clip((np.linalg.norm(g) - 0.36) * 0.14, 0.0, 0.010))
comp = -lag * radial - np.array([0, 0, min(lag, h_grasp - 0.002)])
fast_line(arm, target + comp, R, tol=0.004, label=f"descend (lag pre-comp {lag*1000:.0f} mm)")
err = tips_now(arm) - target
if err[2] > 0.004 or np.hypot(*err[:2]) > 0.006:
    # stiction is unpredictable: the arm may fully reach the corrected target, so never command
    # the fingertips below 4 mm above the table
    corr = target + comp - err
    corr[2] = max(corr[2], tz + 0.004)
    q = solve(tool0_for_tips(corr, R), R, get_joint_angles(arm)[:6])
    command(arm, q)
    err = tips_now(arm) - target
stamp(f"at grasp: tips error vs target {np.round(err*1000, 1)} mm, {tips_now(arm)[2]-tz:.3f} above table")
T_grasp = T_from(*get_ee_pose(arm))

# 5. close, 6. lift, 7. verify
st = command(arm, np.r_[get_joint_angles(arm)[:6], 0.0])
stamp(f"closed, gripper {st[6]:.3f}")
for h_lift in (A.lift, 0.08, 0.06):
    try:
        solve(tool0_for_tips([g[0], g[1], tz + h_lift], R), R, get_joint_angles(arm)[:6])
        break
    except ValueError:
        pass
fast_line(arm, [g[0], g[1], tz + h_lift], R, tol=0.008, label=f"lift to {h_lift*100:.0f} cm")
held = held_zone_check(arm)
Image.fromarray(get_image("top")).save(f"{A.tag}_lifted_top.png")
Image.fromarray(get_image(f"{arm}_wrist")).save(f"{A.tag}_lifted_wrist.png")
stamp(f"PICK {'SUCCEEDED: holding' if held else 'FAILED: not holding'} object")

# optional: put it back where it was, and/or go home
if A.place_back and held:
    fast_line(arm, [g[0], g[1], tz + h_grasp + 0.01], R, tol=0.004, label="place")
    command(arm, np.r_[get_joint_angles(arm)[:6], 1.0])
    stamp("released")
    fast_line(arm, [g[0], g[1], tz + h_pre + 0.02], R, tol=0.008, label="retreat")
    held_zone_check(arm)  # baseline: empty gripper, object >=5 cm below the tips -> expect ~0
elif not held:  # never open in mid-air: lower back to the grasp height first
    fast_line(arm, [g[0], g[1], tz + h_grasp + 0.01], R, tol=0.004, label="lower before opening")
    command(arm, np.r_[get_joint_angles(arm)[:6], 1.0])
    fast_line(arm, [g[0], g[1], tz + h_pre + 0.02], R, tol=0.008, label="retreat")
if A.home or A.place_back:
    fast_move(arm, REST, min_tip_z=tz + 0.05, label="rest")
stamp("done")
