"""Apply an MJCF's joint equalities to a pose that was written straight into qpos.

i2rt's ``SimRobot`` teleports ``qpos[:n_dofs]`` and only runs kinematics. Joint
equality constraints act through ``mj_step``, so a coupled joint outside those
``n_dofs`` (the LINEAR_4310's second finger, ``joint8 = joint7``) never moves.
This solves such joints from the model's own polycoef and reference positions.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

logger = logging.getLogger(__name__)
_warned: set[int] = set()


def apply_joint_couplings(model: mujoco.MjModel, data: mujoco.MjData, n_written: int) -> bool:
    """Set each joint coupled to a written one by an active linear joint equality.

    MuJoCo's joint equality is ``q1 - q1_0 = c0 + c1 (q2 - q2_0) + ...`` with
    ``q*_0`` from ``qpos0``. Only the side outside ``qpos[:n_written]`` is solved;
    nonlinear or non-invertible couplings are skipped with one warning.

    n_written: how many leading qpos entries the caller wrote.
    Returns: True if any qpos changed (kinematics are then recomputed).
    """
    changed = False
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not data.eq_active[e]:
            continue
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if j2 < 0:
            continue
        a1, a2 = int(model.jnt_qposadr[j1]), int(model.jnt_qposadr[j2])
        c = model.eq_data[e][:5]
        written1, written2 = a1 < n_written, a2 < n_written
        if written1 == written2:
            continue
        if np.any(c[2:]) or c[1] == 0.0:
            if e not in _warned:
                _warned.add(e)
                logger.warning("joint equality %d is nonlinear or not invertible; left to dynamics", e)
            continue
        y0, x0 = model.qpos0[a1], model.qpos0[a2]
        if written1:
            data.qpos[a2] = x0 + (data.qpos[a1] - y0 - c[0]) / c[1]
        else:
            data.qpos[a1] = y0 + c[0] + c[1] * (data.qpos[a2] - x0)
        changed = True
    if changed:
        mujoco.mj_forward(model, data)
    return changed
