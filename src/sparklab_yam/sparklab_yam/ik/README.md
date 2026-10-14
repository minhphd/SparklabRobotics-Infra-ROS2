# ik

Inverse kinematics for the YAM-Ultra: given a target gripper pose, find joint
angles.

```
decoupled_ik.py   DecoupledIKSolver
```

**How it works:** joints 1–3 move the wrist to the right *position*, joints 4–6
turn the gripper to the right *orientation*. Each call takes one small, damped
step from the current joint angles — so it is meant to be called every control
tick, not once.

**It never fails.** `solve()` always returns 6 valid joint angles. Near a
singularity, a joint limit, or an impossible twist it slows down or holds that
part still instead of freezing or jumping. Each call also reports how close it
is to trouble (`last_limit_pressure`, `last_pos_err_norm`,
`last_singularity_proximity`, `last_wrist_gimbal_proximity`); the teleop turns
these into controller vibration.

**Tuning:** damping, rest-pose bias and the per-step joint cap are constructor
arguments, documented on the class. Joint limits come from the arm model in
[`../model/`](../model/README.md).

Self-test: `python -m sparklab_yam.ik.decoupled_ik`.
