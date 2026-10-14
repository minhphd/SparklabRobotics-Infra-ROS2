"""Publish every RealSense in a rig file: colour, depth aligned to it, and intrinsics.

    ros2 run sparklab_cameras cameras --ros-args -p rig:=src/sparklab_yam/config/cameras.yaml

Per camera id, all stamped together and with frame_id = id:

    /cameras/<id>/color/image_raw           sensor_msgs/Image  rgb8
    /cameras/<id>/depth/image_raw           sensor_msgs/Image  16UC1 millimetres, 0 = no reading,
                                            aligned to colour: same size, pixel for pixel
    /cameras/<id>/{color,depth}/camera_info the colour intrinsics, after any crop
    /cameras/<id>/points                    sensor_msgs/PointCloud2  xyz (m) + rgb in the optical frame,
                                            every 4th pixel, <= 10 Hz, only while subscribed

Rig file: ``cameras: {<id>: {serial, width, height, fps, crop_height, exposure,
white_balance, gain}}``; all but serial optional. ``height`` must be a profile
the sensor offers; ``crop_height`` then centre-crops rows. exposure,
white_balance and gain freeze those options. A camera that fails to open is
logged and skipped.
"""

from __future__ import annotations

import array
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

STREAMS = ("color", "depth")
# Every 4th pixel at <= 10 Hz keeps a 720p cloud near 1 MB and the stream light enough for Foxglove.
CLOUD_STRIDE, CLOUD_HZ = 4, 10.0
CLOUD_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(("x", "y", "z", "rgb"))]
CLOUD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])


def _image(frame: np.ndarray, encoding: str, stamp, frame_id: str) -> Image:
    frame = np.ascontiguousarray(frame)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = encoding
    msg.step = frame.strides[0]
    # An array.array, not bytes: rclpy converts bytes element by element (~90 ms at 720p).
    msg.data = array.array("B", frame.tobytes())
    return msg


def _camera_info(intr, top: int, height: int, frame_id: str) -> CameraInfo:
    """Pinhole intrinsics of the colour stream, shifted for a crop of `top` rows."""
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width, msg.height = intr.width, height
    msg.distortion_model = "plumb_bob"
    msg.d = [float(c) for c in intr.coeffs]
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy - top
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _rays(K: np.ndarray, D: np.ndarray, width: int, height: int) -> np.ndarray:
    """Undistorted normalised image coordinates of every CLOUD_STRIDE-th pixel, (h, w, 2)."""
    v, u = np.mgrid[0:height:CLOUD_STRIDE, 0:width:CLOUD_STRIDE]
    pixels = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    return cv2.undistortPoints(pixels, K, D).reshape(*u.shape, 2)


def _cloud(rays: np.ndarray, rgb: np.ndarray, mm: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Coloured points from depth aligned to colour; pixels with no reading are dropped.

    rgb is packed into a float32 field, PCL's convention, which Foxglove and RViz read.
    """
    z = mm[::CLOUD_STRIDE, ::CLOUD_STRIDE].astype(np.float32) * 1e-3
    keep = z > 0
    points = np.empty(int(keep.sum()), CLOUD_DTYPE)
    points["x"] = rays[..., 0][keep] * z[keep]
    points["y"] = rays[..., 1][keep] * z[keep]
    points["z"] = z[keep]
    c = rgb[::CLOUD_STRIDE, ::CLOUD_STRIDE][keep].astype(np.uint32)
    points["rgb"] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    msg = PointCloud2(height=1, width=len(points), fields=CLOUD_FIELDS, is_bigendian=False,
                      point_step=CLOUD_DTYPE.itemsize, row_step=CLOUD_DTYPE.itemsize * len(points), is_dense=True)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.data = array.array("B", points.tobytes())
    return msg


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("cameras")
        rig = Path(self.declare_parameter("rig", "").value)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pipes: list = []
        for cam_id, cfg in (yaml.safe_load(rig.read_text())["cameras"] or {}).items():
            try:
                pipe, profile = self._open(cfg)
            except RuntimeError as e:
                self.get_logger().error(f"camera {cam_id} ({cfg['serial']}) not opened: {e}")
                continue
            pubs = {s: (self.create_publisher(Image, f"/cameras/{cam_id}/{s}/image_raw", 1),
                        self.create_publisher(CameraInfo, f"/cameras/{cam_id}/{s}/camera_info", 1))
                    for s in STREAMS}
            pubs["points"] = self.create_publisher(PointCloud2, f"/cameras/{cam_id}/points", 1)
            thread = threading.Thread(target=self._loop, args=(cam_id, cfg, pipe, profile, pubs),
                                      name=f"cam-{cam_id}", daemon=True)
            thread.start()
            self._pipes.append(pipe)
            self._threads.append(thread)
            self.get_logger().info(
                f"camera {cam_id} ({cfg['serial']}, {cfg.get('width', 640)}x"
                f"{cfg.get('height', 480)}@{cfg.get('fps', 30)}) -> /cameras/{cam_id}/{{color,depth}}")

    @staticmethod
    def _open(cfg: dict):
        pipe = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(str(cfg["serial"]))
        size = (cfg.get("width", 640), cfg.get("height", 480))
        rs_cfg.enable_stream(rs.stream.color, *size, rs.format.rgb8, cfg.get("fps", 30))
        rs_cfg.enable_stream(rs.stream.depth, *size, rs.format.z16, cfg.get("fps", 30))
        profile = pipe.start(rs_cfg)
        # D405/D415: colour comes off the one stereo module, not a separate sensor.
        sensor = profile.get_device().query_sensors()[0]
        for key, option, auto in (("exposure", rs.option.exposure, rs.option.enable_auto_exposure),
                                  ("white_balance", rs.option.white_balance,
                                   rs.option.enable_auto_white_balance),
                                  ("gain", rs.option.gain, None)):
            if key in cfg:
                if auto is not None:
                    sensor.set_option(auto, 0)
                sensor.set_option(option, cfg[key])
        return pipe, profile

    def _loop(self, cam_id: str, cfg: dict, pipe, profile, pubs: dict) -> None:
        # Depth units differ by model: 1 mm on the D435, 0.1 mm on the D405.
        to_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        height = min(cfg.get("crop_height") or intr.height, intr.height)
        top = (intr.height - height) // 2
        info = _camera_info(intr, top, height, cam_id)
        rays = _rays(np.array(info.k).reshape(3, 3), np.array(info.d), intr.width, height)
        align = rs.align(rs.stream.color)
        last_frame_t = last_warn_t = time.monotonic()
        last_cloud_t = 0.0
        while not self._stop.is_set():
            try:
                frames = align.process(pipe.wait_for_frames(timeout_ms=1000))
            except RuntimeError:
                now = time.monotonic()
                if now - last_frame_t > 3.0 and now - last_warn_t > 5.0:
                    last_warn_t = now
                    self.get_logger().warning(
                        f"camera {cam_id}: no frame in {now - last_frame_t:.1f}s — check USB "
                        "bandwidth/power, or another process holding the device")
                continue
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            last_frame_t = time.monotonic()
            rgb = np.asanyarray(color.get_data())[top:top + height]
            mm = np.asanyarray(depth.get_data())[top:top + height]
            if not np.isclose(to_mm, 1.0):
                mm = np.round(mm * to_mm).astype(np.uint16)
            info.header.stamp = self.get_clock().now().to_msg()
            for stream, (frame, encoding) in zip(STREAMS, ((rgb, "rgb8"), (mm, "16UC1"))):
                image_pub, info_pub = pubs[stream]
                image_pub.publish(_image(frame, encoding, info.header.stamp, cam_id))
                info_pub.publish(info)
            if pubs["points"].get_subscription_count() and last_frame_t - last_cloud_t >= 1.0 / CLOUD_HZ:
                last_cloud_t = last_frame_t
                pubs["points"].publish(_cloud(rays, rgb, mm, info.header.stamp, cam_id))

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        for pipe in self._pipes:
            pipe.stop()


def main() -> None:
    # Own the signals so the capture threads stop before the context does.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = CameraNode()
    try:
        # A timeout so the KeyboardInterrupt is raised promptly: nothing else wakes this node.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
