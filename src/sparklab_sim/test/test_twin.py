"""The twin's geometry agrees with the real rig's; no Isaac needed.

    PYTHONPATH=src python -m pytest src/sparklab_sim/test     # in sparklab-ros
"""

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from sparklab_sim import calibration, paths, twin_scene


def test_tool0_is_the_drivers():
    if not paths.TWIN_URDF.exists():
        pytest.skip("run ./scripts/isaac_python.sh -m sparklab_sim.convert --twin first")
    from sparklab_yam.ik.decoupled_ik import DecoupledIKSolver

    driver = DecoupledIKSolver()
    rng = np.random.default_rng(0)
    for q in rng.uniform([-1.5, 0.0, 0.0, -1.5, -1.5, -2.0], [1.5, 2.5, 2.5, 1.5, 1.5, 2.0], (20, 6)):
        position, quat_wxyz = driver.fk(q)
        T = twin_scene.tool0(q)
        assert np.allclose(T[:3, 3], position, atol=1e-4)
        angle = (Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_quat(np.roll(quat_wxyz, -1))).magnitude()
        assert angle < 1e-3


def test_cameras_are_the_rigs():
    rig = yaml.safe_load((paths.YAM_ULTRA_CONFIG_DIR / "cameras.yaml").read_text())["cameras"]
    specs = calibration.cameras()
    assert set(specs) == set(rig)
    for cam_id, spec in specs.items():
        assert (spec.width, spec.height) == (rig[cam_id]["width"], rig[cam_id]["height"])
        assert abs(spec.cx - spec.width / 2) < 20 and abs(spec.cy - spec.height / 2) < 20
    assert specs["top"].parent == "left_base"
    assert {specs["left_wrist"].parent, specs["right_wrist"].parent} == {"left_tool0", "right_tool0"}


def test_opening_moves_each_finger_out_by_its_travel():
    if not paths.TWIN_URDF.exists():
        pytest.skip("run ./scripts/isaac_python.sh -m sparklab_sim.convert --twin first")
    for joint, side in zip(twin_scene.FINGER_JOINTS, (1.0, -1.0)):  # tip_left is on +y
        closed_y = twin_scene._joints()[joint][0][1, 3]
        assert side * (twin_scene.open_finger(joint)[1, 3] - closed_y) == pytest.approx(abs(twin_scene.FINGER_OPEN))
