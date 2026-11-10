"""Grasp-frame construction and wrist-camera localisation of an object near a guessed point."""
import numpy as np
import cv2
from PIL import Image
from lib import *


def grasp_R(arm_xy, obj_axis_yaw, tilt):
    """Fingers tilted `tilt` from vertical toward the outward radial direction; closing axis
    perpendicular to the object's horizontal axis. Sign picked nearest the natural wrist."""
    yaw = np.arctan2(arm_xy[1], arm_xy[0])
    rad = np.array([np.cos(yaw), np.sin(yaw), 0])
    z = np.sin(tilt) * rad + np.array([0, 0, -np.cos(tilt)])
    a = np.array([np.cos(obj_axis_yaw), np.sin(obj_axis_yaw), 0])
    y = np.cross(a, z)
    y /= np.linalg.norm(y)
    natural_y = np.array([-np.sin(yaw), np.cos(yaw), 0])
    if y @ natural_y < 0:
        y = -y
    x = np.cross(y, z)
    return np.c_[x, y, z]


def localise(arm, guess_xy, radius=0.08, tag="look", hmax=0.06, nframes=7):
    """From the arm's wrist camera, fit the local table plane and segment the object nearest
    guess_xy (in the arm's base frame), using points 0.6 cm..hmax above the table (excludes the
    gripper's own fingers). Returns dict with table z, object centroid, axis yaw,
    length, width, top height, and saves annotated image."""
    img, pb, valid = wrist_cloud(arm, n=nframes)
    X, Y, Z = pb[..., 0], pb[..., 1], pb[..., 2]
    near = valid & (np.hypot(X - guess_xy[0], Y - guess_xy[1]) < radius)
    ring = valid & (np.hypot(X - guess_xy[0], Y - guess_xy[1]) < 0.18) & (Z < np.percentile(Z[valid], 70))
    # table plane: lowest dense layer in the ring (robust: fit to points below the 60th percentile)
    zr = Z[ring]
    z_tab = np.median(zr[zr < np.percentile(zr, 60)])
    sel = ring & (np.abs(Z - z_tab) < 0.006)
    A = np.c_[X[sel], Y[sel], np.ones(sel.sum())]
    coef = np.linalg.lstsq(A, Z[sel], rcond=None)[0]
    hgt = Z - (X * coef[0] + Y * coef[1] + coef[2])
    z_tab = guess_xy[0] * coef[0] + guess_xy[1] * coef[1] + coef[2]
    m = near & (hgt > 0.006) & (hgt < hmax)
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8)
    if n <= 1:
        print(f"[{tag}] no object found near {guess_xy}; table z {z_tab:.4f}")
        Image.fromarray(img).save(f"{tag}.png")
        return dict(table_z=z_tab, found=False)
    # choose the component with the best size x closeness score: transparent objects break into
    # fragments, and the nearest fragment can be a tiny patch of the object
    best, bestd = None, -1.0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 200:
            continue
        mi = (lab == i) & valid
        d = np.hypot(*(np.median(pb[mi][:, :2], 0) - np.asarray(guess_xy)))
        score = mi.sum() * np.exp(-d / 0.05)
        if score > bestd:
            best, bestd = i, score
    if best is None:
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mm = (lab == best) & valid
    P = pb[mm]
    xy = P[:, :2]
    c = np.median(xy, 0)
    w, V = np.linalg.eigh(np.cov((xy - c).T))
    ax = V[:, 1]
    proj = (xy - c) @ ax
    perp = (xy - c) @ V[:, 0]
    lo, hi = np.percentile(proj, [2, 98])
    plo, phi = np.percentile(perp, [2, 98])
    top = np.percentile(hgt[mm], 95)
    vis = img.copy()
    cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    for P_ in (np.r_[c, z_tab], np.r_[c + lo * ax, z_tab], np.r_[c + hi * ax, z_tab]):
        u, v = project_to_wrist(arm, P_[None])[0]
        cv2.circle(vis, (int(u), int(v)), 6, (255, 0, 255), 2)
    Image.fromarray(vis).save(f"{tag}.png")
    res = dict(found=True, table_z=float(z_tab), plane=coef.tolist(), c=c.tolist(),
               axis_yaw=float(np.arctan2(ax[1], ax[0])), len=float(hi - lo), wid=float(phi - plo),
               top=float(top), ends=[(c + lo * ax).tolist(), (c + hi * ax).tolist()],
               npts=int(mm.sum()), P=P, h=hgt[mm])
    print(f"[{tag}] table z {z_tab:.4f} (plane slope {coef[0]:.3f},{coef[1]:.3f}) | object centroid "
          f"({c[0]:.3f},{c[1]:.3f}) axis yaw {np.degrees(res['axis_yaw']):.0f} deg len {hi-lo:.3f} "
          f"wid {phi-plo:.3f} top {top:.3f} above table | ends {np.array(res['ends'])} | {mm.sum()} px")
    return res
