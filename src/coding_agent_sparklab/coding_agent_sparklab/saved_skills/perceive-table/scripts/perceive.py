"""Read-only perception: segment objects above the table from the top camera.

Deprojects top-camera depth into the left base frame, RANSAC-fits the table plane,
then segments pixels that are above the plane OR clearly non-grey (dark / blue), and
reports each blob's centroid, principal axis, extents and height (robust percentiles).
"""
import json
import numpy as np
import cv2
from PIL import Image
from coding_agent_sparklab.robot import get_depth, get_image, get_intrinsics, EXTRINSICS

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

K, _ = get_intrinsics("top")
img = get_image("top")
depth = np.median(np.stack([get_depth("top") for _ in range(9)]), axis=0)
H, W = depth.shape
T = EXTRINSICS[("left_base", "top")]

u, v = np.meshgrid(np.arange(W), np.arange(H))
rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(H * W)])
pb = (T[:3, :3] @ (rays * depth.ravel()) + T[:3, 3:4]).T.reshape(H, W, 3)
valid = depth > 0
X, Y, Z = pb[..., 0], pb[..., 1], pb[..., 2]

# Workspace box in left base frame (excludes arms near the bases: x > 0.20 except gripper region)
ws = valid & (X > 0.12) & (X < 0.75) & (Y > -0.80) & (Y < 0.25)

# RANSAC plane z = a x + b y + c
pts = pb[ws]
best, best_in = None, 0
for _ in range(300):
    s = pts[rng.choice(len(pts), 3, replace=False)]
    A = np.c_[s[:, :2], np.ones(3)]
    try:
        coef = np.linalg.solve(A, s[:, 2])
    except np.linalg.LinAlgError:
        continue
    r = np.abs(pts[:, 2] - (pts[:, :2] @ coef[:2] + coef[2]))
    ni = (r < 0.004).sum()
    if ni > best_in:
        best, best_in = coef, ni
inl = np.abs(pts[:, 2] - (pts[:, :2] @ best[:2] + best[2])) < 0.004
A = np.c_[pts[inl, :2], np.ones(inl.sum())]
coef = np.linalg.lstsq(A, pts[inl, 2], rcond=None)[0]
print(f"table plane z = {coef[0]:.4f} x + {coef[1]:.4f} y + {coef[2]:.4f}  "
      f"(inliers {inl.mean():.0%});  z at (0.35,0)= {0.35*coef[0]+coef[2]:.4f}, "
      f"at (0.35,-0.55)= {0.35*coef[0]-0.55*coef[1]+coef[2]:.4f}")
hgt = Z - (X * coef[0] + Y * coef[1] + coef[2])

# Colour cue: table is light grey (low saturation, mid-high value)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
sat, val = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
colour_obj = (sat > 70) | (val < 70)

mask = ws & (X > 0.30) & ((hgt > 0.006) | colour_obj)
mask = (mask | ((ws | ~valid) & colour_obj & (v < 360))) # thin dark parts may lack depth
mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)

vis = img.copy()
out = []
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 300:
        continue
    m = (lab == i) & valid
    if m.sum() < 50:
        continue
    P = pb[m]
    h = hgt[m]
    xy = P[:, :2]
    c = np.median(xy, axis=0)
    w_, V = np.linalg.eigh(np.cov((xy - c).T))
    ax = V[:, 1]
    proj = (xy - c) @ ax
    perp = (xy - c) @ V[:, 0]
    lo, hi = np.percentile(proj, [2, 98])
    plo, phi = np.percentile(perp, [2, 98])
    top = np.percentile(h, 95)
    yaw = np.degrees(np.arctan2(ax[1], ax[0]))
    cu, cv_ = cents[i]
    print(f"blob {i}: px ({cu:.0f},{cv_:.0f}) n {m.sum()}  centroid ({c[0]:.3f},{c[1]:.3f})  "
          f"len {hi-lo:.3f} wid {phi-plo:.3f}  top {top:.3f}  axis yaw {yaw:.0f}")
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    cv2.putText(vis, str(i), (int(cu) + 12, int(cv_)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    out.append(dict(id=i, px=[float(cu), float(cv_)], c=c.tolist(), axis=ax.tolist(),
                    len=float(hi - lo), wid=float(phi - plo), top=float(top)))

Image.fromarray(vis).save("top_seg.png")
Image.fromarray((np.clip(hgt, 0, 0.05) / 0.05 * 255).astype(np.uint8)).save("top_height.png")
json.dump(dict(plane=coef.tolist(), objs=out), open("objects.json", "w"), indent=1)
