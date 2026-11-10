# archive

Retired code, kept as it was for reference. Nothing in the stack imports it.

```
decoupled_ik.py   DecoupledIKSolver: joints 1-3 for position, 4-6 for orientation; the YAM's IK until 2026-11-10
```

`decoupled_ik.py` was `sparklab_yam/ik/decoupled_ik.py`; `sparklab_yam/ik/solver.py`
replaced it. With the workspace sourced, its self-test still runs:
`python archive/decoupled_ik.py`. Why it was replaced:
[`DESIGN.md`](../DESIGN.md#inverse-kinematics).
