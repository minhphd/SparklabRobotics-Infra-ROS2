# model

The YAM-Ultra arm model, copied from i2rt, plus the code that assembles it for
the IK.

```
kinematics.py     builds the MuJoCo model the IK uses, adds two reference points, self-checks
yam_ultra.xml     the arm (MuJoCo format)
linear_4310.xml   the gripper (MuJoCo format)
yam_ultra.urdf    arm + gripper (URDF format), used by the Isaac twin
assets/*.stl      meshes
```

The files come byte-for-byte from **i2rt v1.2.4**
(`i2rt/robot_models/arm/yam_ultra/` and `.../gripper/linear_4310/`). Take them
from that tag, not from an older local clone.

**One deliberate change:** joint 3's upper limit is **3.0 rad** here instead of
i2rt's π, because this rig's elbow physically stops at about 3.0. Don't change
it back. It's only applied to the MuJoCo file; the URDF still says π.

**Reference points** added by `kinematics.py`:

- `tool0` — the gripper's grasp centre; the point the IK steers.
- `j4_anchor` — a point on the forearm the position solve uses.

**Naming differences:** the URDF calls the joints `dof_joint1..6` (MuJoCo:
`joint1..6`) and models the gripper as two finger joints; LeRobot uses one
`gripper.pos` from 0 to 1.

**After updating i2rt, run:**

```bash
python -m sparklab_yam.model.kinematics
```

It checks the MuJoCo and URDF files agree, compares against i2rt's copy, and
re-checks the IK's assumptions.
