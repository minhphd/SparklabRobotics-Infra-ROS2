---
name: perceive-table
description: Find the objects on the table with the top camera - fits the table plane from depth and segments everything standing above it or clearly coloured, giving each object's centre, axis, size and height in the left arm's base frame. Use at the start of a task to learn what is on the table and roughly where. Moves nothing.
---

# perceive-table

```bash
python <this folder>/scripts/perceive.py
```

Run it from the session's working directory; it writes there:

- `objects.json`: `plane`, the table as `z = a x + b y + c` (`[a, b, c]`), and
  `objs`, one per object with `id`, `px` (its pixel), `c` (x, y centre, m),
  `axis` (unit xy direction of its long side), `len`, `wid`, and `top` (height of
  its top above the table, m).
- `top_seg.png`: the top image with every object outlined and numbered.
- `top_height.png`: height above the table, 0 to 5 cm as black to white.

It prints the plane and one line per object.

- Everything is in the left arm's base frame; `left_to_right` in rig-toolkit
  converts a point for the right arm.
- It searches x 0.12–0.75 m and y −0.80–0.25 m, and drops anything nearer than
  x = 0.30 m, where the arms are.
- Light grey objects must stand at least 6 mm above the table. Dark or coloured
  ones are found even where the depth has holes, in the upper half of the image.
- Positions are good enough to aim a look: refine them with the wrist camera
  (rig-toolkit's `localise`, which pick uses) before grasping.
