# ik

Inverse kinematics for the YAM-Ultra: given a target gripper pose (`tool0` in the
arm base), find joint angles.

```
solver.py   IKSolver
```

**How it works:** one damped least-squares step on the whole 6-D error, so every
joint works on position and orientation together. Orientation counts
`rot_weight` metres per radian against position. Each call takes one step from
the joint angles it is given: the teleop calls it every control tick, and
`coding_agent_sparklab.robot.ik` iterates it to convergence.

**It never fails.** `solve()` always returns 6 joint angles within the limits.
A joint the step would push past its limit is pinned there and the others
re-solve to make up for it. Damping grows near any singularity, shoulder or
wrist, so steps stay bounded. Each call also reports how close it is to trouble
(`last_limit_pressure`, `last_pos_err_norm`, `last_singularity_proximity`); the
teleop turns these into controller vibration.

**Tuning:** damping, `rot_weight` and the per-step joint cap are constructor
arguments, documented on the class. The teleop sets them from `QuestTeleopConfig`.
Joint limits come from the arm model in [`../model/`](../model/README.md).

The decoupled solver this replaced is in [`archive/`](../../../../archive/README.md).
