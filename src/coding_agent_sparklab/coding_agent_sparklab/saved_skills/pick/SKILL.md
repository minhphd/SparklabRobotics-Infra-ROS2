---
name: pick
description: Pick up one object from the table with one arm, given its rough position (e.g. from perceive-table) - looks from above with the wrist camera, finds a graspable cross-section, descends in a straight line, closes, lifts and checks the object is held. Use to pick something up; it can also put it back or go to the rest pose afterwards.
---

# pick

```bash
python <this folder>/scripts/fast_pick.py ARM X Y [--yaw DEG] [--max-width M] [--wide M]
                                          [--radius M] [--lift M] [--place-back] [--home] [--tag NAME]
```

| argument | default | |
|---|---|---|
| `ARM` | | `left` or `right` |
| `X Y` | | the object's rough position in that arm's base frame, m |
| `--yaw` | none | rough direction of the object's long axis, degrees, so the first look is already turned |
| `--max-width` | 0.055 | widest cross-section it will close on, m |
| `--wide` | 0.05 | stays 3 cm away from any part wider than this, m |
| `--radius` | 0.08 | how far from X, Y the object may be, m |
| `--lift` | 0.10 | lift height, m; it tries 0.08 and 0.06 if that is out of reach |
| `--place-back` | | set the object down where it was, release and retreat |
| `--home` | | go to the rest pose at the end |
| `--tag` | `fp` | prefix of the images it saves |

What it does:

1. Looks from 5–11 cm above X, Y, fingers down; if that is out of reach, from
   11 cm with the fingers tilted 30°.
2. Localises the object with the wrist camera and picks a grasp point; looks
   again over it if it is more than 3 cm off.
3. Chooses the smallest tilt (0, 20, 30 or 40°, leaning along the object) at
   which both the pre-grasp and the grasp are reachable, and re-aims if the
   tilt, the turn or the offset calls for it.
4. Descends in a straight line to a grasp 7 mm to 2 cm above the table,
   depending on the object's height; the lag pre-compensation may command as
   low as 2 mm, expecting the arm to stop short. If it ends more than 4 mm high
   or 6 mm to the side, it makes one correction, which never aims the
   fingertips lower than 4 mm above the table.
5. Closes, lifts, and checks the object is held. If it is not, it lowers back to
   the grasp height before opening: it never opens in mid-air.

It prints each step with elapsed time and ends with `PICK SUCCEEDED: holding
object` or `PICK FAILED: not holding object`. It saves `<tag>_look.png`,
`<tag>_lifted_top.png` and `<tag>_lifted_wrist.png` in the working directory.

## Measured on the rig

On 2026-11-10, with the long fin-ray fingers:

- Reaching far, the arm stops short and high: the pre-compensation grows from 0
  at 0.36 m from the base to 1 cm at 0.43 m.
- It assumes the table at z = −0.031 in the arm's base frame until the wrist
  camera measures it.
