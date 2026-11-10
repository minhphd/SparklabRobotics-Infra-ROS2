"""Helpers for picking: geometry, IK planning, path checks, guarded moves, wrist-camera perception."""
import numpy as np
import cv2
from coding_agent_sparklab.robot import (get_joint_angles, get_ee_pose, fk, ik, command, get_image,
                                         get_depth, get_intrinsics, EXTRINSICS, FINGERTIPS)

np.set_printoptions(precision=4, suppress=True)
REST = np.array([0, 0.5, 1.0, -1.0, 0, 0])
T_L_R = EXTRINSICS[("left_base", "right_base")]
TABLE_Z = -0.01  # operator-stated table height in each base frame (verify with wrist depth)


# ---------- rotations ----------
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = np.argmax(np.diag(R))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def topdown_R(close_yaw):
    """Fingers straight down; fingers close along base-frame direction at angle close_yaw (rad)."""
    y = np.array([np.cos(close_yaw), np.sin(close_yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.c_[x, y, z]


def wrap_close_yaw(close_yaw):
    """Closing is symmetric under +pi; pick the equivalent nearest the natural pi/2."""
    a = (close_yaw - np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2 + np.pi / 2
    return a


def tool0_for_tips(p_tips, R):
    return np.asarray(p_tips) - FINGERTIPS * R[:, 2]


# ---------- frames ----------
def left_to_right(p):
    return (np.linalg.inv(T_L_R) @ np.r_[p, 1])[:3]


def right_to_left(p):
    return (T_L_R @ np.r_[p, 1])[:3]


def T_from(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = p
    return T


# ---------- planning ----------
def solve(pos, R, seed):
    return ik(np.asarray(pos, float), R_to_quat(R), seed=np.asarray(seed, float))


def path_min_z(q0, q1, n=25):
    """Min fingertip z and min tool0 z along the straight joint-space path q0 -> q1."""
    tips, tool = [], []
    for s in np.linspace(0, 1, n):
        p, q = fk(q0 + s * (np.asarray(q1) - q0))
        tool.append(p[2])
        tips.append((p + FINGERTIPS * quat_to_R(q)[:, 2])[2])
    return min(tips), min(tool)


def move(arm, q, grip=None, min_tip_z=None, label=""):
    """Guarded joint move. Refuses if the path dips fingertips below min_tip_z."""
    q = np.asarray(q, float)
    q0 = get_joint_angles(arm)[:6]
    tmin, _ = path_min_z(q0, q)
    if min_tip_z is not None and tmin < min_tip_z:
        raise RuntimeError(f"{label}: path dips fingertips to z={tmin:.3f} < {min_tip_z:.3f}; refusing")
    # small awaited joint steps: joint 2 can't track one big ramp when the arm is extended & loaded
    nstep = max(1, int(np.ceil(np.abs(q - q0).max() / 0.08)))
    for s in np.linspace(0, 1, nstep + 1)[1:]:
        st = command(arm, q0 + s * (q - q0))
    if grip is not None:
        st = command(arm, np.r_[q, grip])
    p, qq = fk(st[:6])
    tips = p + FINGERTIPS * quat_to_R(qq)[:, 2]
    print(f"[{label}] {arm} joints {st[:6]} grip {st[6]:.3f} | tool0 {p} tips {tips} "
          f"| joint err max {np.abs(st[:6]-q).max():.4f} rad | path min tip z {tmin:.3f}")
    return st


def cart_line(arm, tips_to, R, step=0.01, label="", comp=False):
    """Straight-line fingertip motion (constant orientation R) in ~1 cm awaited IK waypoints.
    comp=True: measure the tracking lag after each waypoint and pre-offset the next one by it."""
    p, qq = get_ee_pose(arm)
    t0 = p + FINGERTIPS * quat_to_R(qq)[:, 2]
    tips_to = np.asarray(tips_to, float)
    n = max(1, int(np.ceil(np.linalg.norm(tips_to - t0) / step)))
    q = get_joint_angles(arm)[:6]
    lag = np.zeros(3)
    for s in np.linspace(0, 1, n + 1)[1:]:
        tgt = t0 + s * (tips_to - t0)
        cmd = tgt - lag
        q = solve(tool0_for_tips(cmd, R), R, q)
        st = command(arm, q)
        if comp:
            pm, qm = fk(st[:6])
            lag = (pm + FINGERTIPS * quat_to_R(qm)[:, 2]) - cmd
            lag = np.clip(lag, -0.02, 0.02)
    p, qq = fk(st[:6])
    tips = p + FINGERTIPS * quat_to_R(qq)[:, 2]
    print(f"[{label}] tips {tips} (target {tips_to}) joints {st[:6]} grip {st[6]:.3f}")
    return st


def grip(arm, g, label="grip"):
    q = get_joint_angles(arm)[:6]
    st = command(arm, np.r_[q, g])
    print(f"[{label}] {arm} gripper -> {st[6]:.3f} (cmd {g})")
    return st


# ---------- wrist camera ----------
def wrist_T(arm, q=None):
    q = get_joint_angles(arm)[:6] if q is None else q
    p, qu = fk(q)
    return T_from(p, qu) @ EXTRINSICS[(f"{arm}_tool0", f"{arm}_wrist")]


def wrist_cloud(arm, n=5):
    """Return image, per-pixel base-frame points (H,W,3), valid mask, from the arm's wrist camera."""
    cam = f"{arm}_wrist"
    K, D = get_intrinsics(cam)
    img = get_image(cam)
    d = np.median(np.stack([get_depth(cam) for _ in range(n)]), axis=0)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, D).reshape(-1, 2)
    rays = np.c_[und, np.ones(len(und))].T
    pc = rays * d.ravel()
    T = wrist_T(arm)
    pb = (T[:3, :3] @ pc + T[:3, 3:4]).T.reshape(H, W, 3)
    return img, pb, d > 0


def project_to_wrist(arm, P_base):
    """Project base-frame points into the wrist image (pixels)."""
    cam = f"{arm}_wrist"
    K, D = get_intrinsics(cam)
    T = np.linalg.inv(wrist_T(arm))
    Pc = (T[:3, :3] @ np.asarray(P_base).T + T[:3, 3:4]).T
    px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
    return px.reshape(-1, 2)
