"""The point cloud built from depth aligned to colour."""

import numpy as np
from builtin_interfaces.msg import Time

from sparklab_cameras.node import CLOUD_DTYPE, CLOUD_STRIDE, _cloud, _rays

K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])


def test_a_wall_lands_at_its_depth_and_pinhole_position():
    mm = np.full((480, 640), 1000, np.uint16)
    mm[:, :320] = 0  # no reading on the left half
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[..., 0], rgb[..., 2] = 200, 7
    msg = _cloud(_rays(K, np.zeros(5), 640, 480), rgb, mm, Time(sec=5), "top")
    points = np.frombuffer(msg.data, CLOUD_DTYPE)
    assert msg.header.frame_id == "top" and msg.width == len(points)
    assert len(points) == (480 // CLOUD_STRIDE) * (320 // CLOUD_STRIDE)
    assert np.allclose(points["z"], 1.0)
    assert points["x"].min() == 0.0
    assert points["x"].max() == np.float32((640 - CLOUD_STRIDE - 320) / 400)
    assert points["y"].min() == np.float32(-240 / 400)
    assert np.all(points["rgb"] == (200 << 16) | 7)


def test_rays_undo_lens_distortion():
    D = np.array([-0.05, 0.06, 0.0, 0.0, -0.02])
    pinhole, undistorted = _rays(K, np.zeros(5), 640, 480), _rays(K, D, 640, 480)
    assert np.allclose(undistorted[60, 80], pinhole[60, 80])  # the principal point
    # Barrel distortion pulls the corner pixel inwards, so its ray points further out.
    assert np.linalg.norm(undistorted[0, 0]) > np.linalg.norm(pinhole[0, 0])
