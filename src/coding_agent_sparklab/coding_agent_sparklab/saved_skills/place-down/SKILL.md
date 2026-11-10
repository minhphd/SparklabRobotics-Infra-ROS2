---
name: place-down
description: Set down the object an arm is holding, straight below the gripper and keeping its orientation, then release, rise clear and go to the rest pose. Use after pick to put something on the table.
---

# place-down

```bash
python <this folder>/scripts/place_down.py ARM TABLE_Z [H_PLACE] [H_CLEAR]
```

| argument | default | |
|---|---|---|
| `ARM` | | `left` or `right` |
| `TABLE_Z` | | the table's height in that arm's base frame, m: the `table z` pick printed, about −0.03 |
| `H_PLACE` | 0.02 | fingertip height above the table to release at, m |
| `H_CLEAR` | 0.085 | fingertip height to rise to after releasing, m |

It lowers the fingertips in a straight line to `H_PLACE`, opens, rises to
`H_CLEAR`, goes to the rest pose, and saves `place_down_top.png` in the working
directory. To set the object somewhere else, move it there first, keeping the
fingertips well above the table.
