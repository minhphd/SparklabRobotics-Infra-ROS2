---
name: rig-toolkit
description: Shared Python helpers the other skills import - rotations and frames, ik with a rotation matrix, guarded joint moves, straight fingertip lines, the gripper, wrist-camera point clouds, object localisation, grasp-point choice and held checks. Use when writing any script that moves an arm or reads a wrist camera, instead of rewriting these.
---

# rig-toolkit

Three modules in `scripts/`. Put that folder on `sys.path`, then import; from
another skill's script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rig-toolkit" / "scripts"))
from skill_core import *          # brings in all of lib too
from grasp_lib import localise, grasp_R
```

| module | provides |
|---|---|
| `lib.py` | `quat_to_R`, `R_to_quat`, `topdown_R(close_yaw)` (fingers down, closing along `close_yaw`), `wrap_close_yaw`, `tool0_for_tips(p_tips, R)`, `left_to_right` / `right_to_left` (points between base frames), `T_from(p, q)`, `solve(pos, R, seed)` (ik with a rotation matrix), `path_min_z(q0, q1)`, `move(arm, q, grip, min_tip_z)` (refuses a path that dips the fingertips too low), `cart_line(arm, tips_to, R)` (straight fingertip line in 1 cm steps), `grip(arm, g)`, `wrist_T(arm)`, `wrist_cloud(arm, n)` (image, per-pixel base-frame points, valid mask), `project_to_wrist(arm, P)`; constants `REST`, `T_L_R`, `TABLE_Z` |
| `skill_core.py` | `stamp(msg)` (timed print), `tips_of(q)`, `tips_now(arm)`, `fast_move(arm, q, grip, min_tip_z, label)`, `fast_line(arm, tips_to, R, tol, label, correct)`, `grasp_point(res, maxw, wide)`, `held_zone_check(arm)`, `held_check(arm, P_obj, T_grasp)` |
| `grasp_lib.py` | `grasp_R(arm_xy, obj_axis_yaw, tilt)`, `localise(arm, guess_xy, radius, tag, hmax, nframes)` (wrist-camera table plane and the object nearest `guess_xy`) |

Prefer `fast_move` and `fast_line` over `move` and `cart_line`: they reach the
same places in fewer commands.

## Measured on the rig

On 2026-11-10, with the long fin-ray fingers:

- Every `command()` costs about 2 s of settling, so `fast_move` and `fast_line`
  use as few commands as the path allows.
- Joints 2 and 3 can trip `command()`'s lag check when lifting an extended arm,
  and when breaking away from park. `fast_move` takes one 0.12 rad step out of
  park first, and finishes a tripped leg in 0.12 rad steps; a second trip is
  raised, since it may be a real obstruction.
- `held_zone_check` counted 6318 and 11341 depth pixels beside the fingertips
  while holding, 0 and 199 with the gripper empty; it calls 1000 or more held, or
  a gripper that stopped between 0.1 and 0.9.
