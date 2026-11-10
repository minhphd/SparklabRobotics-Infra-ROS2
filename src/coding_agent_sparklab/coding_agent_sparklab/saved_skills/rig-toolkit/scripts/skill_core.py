"""Fast motion primitives + grasp planning / verification shared by the pick skill."""
import time
import numpy as np
from lib import *

T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def tips_of(q):
    p, qq = fk(q)
    return p + FINGERTIPS * quat_to_R(qq)[:, 2]


def tips_now(arm):
    return tips_of(get_joint_angles(arm)[:6])


LIFT_CHUNK = 0.5    # max step for a shoulder/elbow joint that raises the arm (against gravity)
SAFE_CHUNK = 0.12   # proven step size after a lag trip


def _chunks(q0, q):
    """Every command costs ~2 s of settling, so use one command unless joint 2 or 3 moves a lot in
    the direction that raises tool0 (against gravity) -- the case that has tripped the lag check."""
    n = 1
    z0 = fk(q0)[0][2]
    for j in (1, 2):
        dq = q[j] - q0[j]
        if abs(dq) <= LIFT_CHUNK:
            continue
        qt = q0.copy()
        qt[j] += 0.02 * np.sign(dq)
        if fk(qt)[0][2] > z0:
            n = max(n, int(np.ceil(abs(dq) / LIFT_CHUNK)))
    return n


def _go(arm, q, label=""):
    """Command q with the fewest commands. On one lag trip (arm is held in place) finish the leg
    in SAFE_CHUNK steps; a second trip is re-raised (could be a real obstruction)."""
    q = np.asarray(q, float)
    q0 = get_joint_angles(arm)[:6]
    if np.abs(q0).max() < 0.1 and np.abs(q - q0).max() > SAFE_CHUNK:
        # leaving the folded park pose: joints have tripped breaking away here; one proven-safe
        # small step first, then the big move
        st = command(arm, q0 + np.clip(q - q0, -SAFE_CHUNK, SAFE_CHUNK))
        stamp(f"{label}: unpark step done")
        q0 = st[:6]
    n = _chunks(q0, q)
    tripped = False
    for s in np.linspace(0, 1, n + 1)[1:]:
        tgt = q0 + s * (q - q0)
        try:
            st = command(arm, tgt)
        except RuntimeError as e:
            if tripped:
                raise
            tripped = True
            qa = get_joint_angles(arm)[:6]
            m = int(np.ceil(np.abs(tgt - qa).max() / SAFE_CHUNK))
            stamp(f"{label}: {e} -> finishing leg in {m} x {SAFE_CHUNK} rad chunks")
            for u in np.linspace(0, 1, m + 1)[1:]:
                st = command(arm, qa + u * (tgt - qa))
    return st, n


def fast_move(arm, q, grip=None, min_tip_z=None, label=""):
    q = np.asarray(q, float)
    q0 = get_joint_angles(arm)[:6]
    tmin, _ = path_min_z(q0, q)
    if min_tip_z is not None and tmin < min_tip_z:
        raise RuntimeError(f"{label}: path dips fingertips to z={tmin:.3f} < {min_tip_z:.3f}; refusing")
    st, n = _go(arm, q, label)
    if grip is not None:
        st = command(arm, np.r_[q, grip])
    stamp(f"{label}: {n} cmd(s), tips {np.round(tips_of(st[:6]), 4)}, grip {st[6]:.3f}")
    return st


def _line_waypoints(q0, t0, t1, R, tol, depth=0):
    """Joint waypoints whose straight joint-space legs keep the tips within tol of the line t0->t1."""
    q1 = solve(tool0_for_tips(t1, R), R, q0)
    dev = max(np.linalg.norm(tips_of(q0 + s * (q1 - q0)) - (t0 + s * (t1 - t0)))
              for s in np.linspace(0.1, 0.9, 9))
    if dev <= tol or depth >= 4:
        return [q1]
    tm = 0.5 * (t0 + t1)
    left = _line_waypoints(q0, t0, tm, R, tol, depth + 1)
    return left + _line_waypoints(left[-1], tm, t1, R, tol, depth + 1)


def fast_line(arm, tips_to, R, tol=0.004, label="", correct=False):
    """Straight fingertip line with as few commands as the tolerance allows; optional one-shot
    correction of the steady-state lag at the end."""
    q0 = get_joint_angles(arm)[:6]
    t0 = tips_of(q0)
    tips_to = np.asarray(tips_to, float)
    wps = _line_waypoints(q0, t0, tips_to, R, tol)
    for q in wps:
        st, _ = _go(arm, q, label)
    err = tips_of(st[:6]) - tips_to
    msg = f"{label}: {len(wps)} waypoint(s), error {np.round(err*1000, 1)} mm"
    if correct and np.linalg.norm(err) > 0.004:
        st = command(arm, solve(tool0_for_tips(tips_to - err, R), R, st[:6]))
        err = tips_of(st[:6]) - tips_to
        msg += f" -> corrected {np.round(err*1000, 1)} mm"
    stamp(msg)
    return st


def grasp_point(res, maxw=0.055, wide=0.05):
    """Graspable cross-section: narrow enough, thick enough, >=3 cm from wide parts, nearest the
    height-weighted centroid. Axis from ridge points (robust to ragged edges)."""
    P, h = res["P"], res["h"]
    xy = P[:, :2]
    c = np.median(xy, 0)
    ridge = h > 0.5 * np.percentile(h, 95)
    rxy = xy[ridge]
    _, V = np.linalg.eigh(np.cov((rxy - rxy.mean(0)).T))
    a, n = V[:, 1], V[:, 0]
    s, t = (xy - c) @ a, (xy - c) @ n
    lo_s, hi_s = np.percentile(s, [1, 99])
    bins = []
    for lo in np.arange(lo_s, hi_s, 0.01):
        m = (s >= lo) & (s < lo + 0.01)
        bins.append(None if m.sum() < 15 else
                    dict(s=lo + 0.005, wid=np.percentile(t[m], 95) - np.percentile(t[m], 5),
                         top=np.percentile(h[m], 90), t=np.median(t[m])))
    s_com = np.sum(s * h) / np.sum(h)
    cands = []
    for i, b in enumerate(bins):
        if b is None or b["wid"] > maxw or b["top"] < 0.012:
            continue
        near = bins[max(0, i - 3):i + 4]
        if any(x is not None and x["wid"] > wide for x in near):
            continue
        cands.append((abs(b["s"] - s_com), i))
    if not cands:
        raise RuntimeError("no graspable cross-section found")
    b = bins[min(cands)[1]]
    g = c + b["s"] * a + b["t"] * n
    yaw = np.arctan2(a[1], a[0])
    half = max(0.03, 1.5 * b["wid"])
    m = ridge & (np.abs(s - b["s"]) < half)
    if m.sum() > 30:
        w2, V2 = np.linalg.eigh(np.cov((xy[m] - xy[m].mean(0)).T))
        if w2[1] / max(w2[0], 1e-9) >= 3:
            a2 = V2[:, 1] * np.sign(V2[:, 1] @ a)
            yaw = np.arctan2(a2[1], a2[0])
    stamp(f"grasp: xy ({g[0]:.3f},{g[1]:.3f}) axis {np.degrees(yaw):.0f} deg width {b['wid']:.3f} "
          f"top {b['top']:.3f}")
    return g, yaw, b["top"]


def held_zone_check(arm, nframes=3, min_pts=1000):  # held: 6318 / 11341 px; empty: 0 / 199 px
    """Shape-free held check: wrist-depth points expressed in the tool0 frame, counted in a zone
    only a held object can occupy -- near fingertip height, within +-6 cm of the closing axis, but
    3-15 cm to the side of the 4 cm-wide fingers. Empty gripper lifted 10 cm: nothing there."""
    img, pb, valid = wrist_cloud(arm, n=nframes)
    T = np.linalg.inv(T_from(*get_ee_pose(arm)))
    P = pb[valid]
    pt = (T[:3, :3] @ P.T + T[:3, 3:4]).T
    zt = FINGERTIPS
    m = ((np.abs(pt[:, 1]) < 0.06) & (np.abs(pt[:, 0]) > 0.03) & (np.abs(pt[:, 0]) < 0.15)
         & (pt[:, 2] > zt - 0.06) & (pt[:, 2] < zt + 0.03))
    g = get_joint_angles(arm)[6]
    n = int(m.sum())
    held = n >= min_pts or 0.1 < g < 0.9  # gripper stopped part-way = closed on something thick
    stamp(f"held check: {n} depth px in the held-object zone beside the fingertips (need {min_pts}), "
          f"gripper {g:.3f} -> {'HELD' if held else 'EMPTY'}")
    return held


def held_check(arm, P_obj, T_grasp, nframes=3):
    T_now = T_from(*get_ee_pose(arm))
    M = T_now @ np.linalg.inv(T_grasp)
    idx = np.random.default_rng(0).choice(len(P_obj), min(600, len(P_obj)), replace=False)
    Pn = (M[:3, :3] @ P_obj[idx].T + M[:3, 3:4]).T
    Tc = np.linalg.inv(wrist_T(arm))
    Pc = (Tc[:3, :3] @ Pn.T + Tc[:3, 3:4]).T
    px = project_to_wrist(arm, Pn)
    d = np.median(np.stack([get_depth(f"{arm}_wrist") for _ in range(nframes)]), axis=0)
    H, W = d.shape
    u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    dm, pred = d[v[ok], u[ok]], Pc[ok, 2]
    diff = dm[dm > 0] - pred[dm > 0]
    vis = diff > -0.015                     # points hidden behind the fingers carry no information
    match = (np.abs(diff[vis]) < 0.015).mean() if vis.sum() else 0.0
    behind = (diff[vis] > 0.04).mean() if vis.sum() else 0.0
    stamp(f"held check: {vis.sum()} visible of {len(diff)} predicted pts ({(~vis).mean():.0%} hidden by "
          f"fingers) | visible: match {match:.0%}, farther(table) {behind:.0%}")
    if vis.sum() < 20:
        stamp("held check: too few visible points -> assume held (never drop on an uncertain check)")
        return True
    return match > 0.5
