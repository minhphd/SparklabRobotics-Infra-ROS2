import unittest

import mujoco
import numpy as np

from sparklab_yam.model.kinematics import combine_arm_and_gripper_xml
from sparklab_yam.sim_coupling import apply_joint_couplings

AFFINE = """
<mujoco>
  <worldbody>
    <body name="a"><joint name="ja" type="slide" axis="1 0 0" ref="0.1"/><geom size="0.01"/></body>
    <body name="b" pos="0 1 0"><joint name="jb" type="slide" axis="0 1 0" ref="0.2"/><geom size="0.01"/></body>
  </worldbody>
  <equality><joint joint1="jb" joint2="ja" polycoef="0.01 2 0 0 0"/></equality>
</mujoco>
"""


class SimCouplingTest(unittest.TestCase):
    """Joints left out of a teleported qpos follow the model's joint equalities."""

    def test_linear4310_second_finger_follows_the_first(self) -> None:
        model = mujoco.MjModel.from_xml_string(combine_arm_and_gripper_xml())
        data = mujoco.MjData(model)
        for q7 in (0.0, 0.02, 0.0475):
            data.qpos[:] = 0.0
            data.qpos[6] = q7
            apply_joint_couplings(model, data, 7)
            self.assertAlmostEqual(data.qpos[7], q7, places=12)

    def test_affine_coupling_uses_polycoef_and_reference_positions(self) -> None:
        model = mujoco.MjModel.from_xml_string(AFFINE)
        data = mujoco.MjData(model)
        data.qpos[0] = 0.3
        self.assertTrue(apply_joint_couplings(model, data, 1))
        self.assertAlmostEqual(data.qpos[1] - 0.2, 0.01 + 2 * (0.3 - 0.1), places=12)
        mujoco.mj_forward(model, data)
        self.assertLess(abs(data.efc_pos[0]) if data.nefc else 0.0, 1e-9)

    def test_solves_the_other_side_when_only_joint1_was_written(self) -> None:
        model = mujoco.MjModel.from_xml_string(AFFINE.replace('joint1="jb" joint2="ja"', 'joint1="ja" joint2="jb"'))
        data = mujoco.MjData(model)
        data.qpos[0] = 0.3
        apply_joint_couplings(model, data, 1)
        self.assertAlmostEqual(0.3 - 0.1, 0.01 + 2 * (data.qpos[1] - 0.2), places=12)

    def test_untouched_when_both_or_neither_side_was_written(self) -> None:
        model = mujoco.MjModel.from_xml_string(AFFINE)
        data = mujoco.MjData(model)
        data.qpos[:] = [0.3, 0.5]
        self.assertFalse(apply_joint_couplings(model, data, 2))
        self.assertFalse(apply_joint_couplings(model, data, 0))
        np.testing.assert_array_equal(data.qpos, [0.3, 0.5])

    def test_inactive_equality_is_ignored(self) -> None:
        model = mujoco.MjModel.from_xml_string(AFFINE)
        data = mujoco.MjData(model)
        data.eq_active[0] = 0
        data.qpos[0] = 0.3
        self.assertFalse(apply_joint_couplings(model, data, 1))


if __name__ == "__main__":
    unittest.main()
