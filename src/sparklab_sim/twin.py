"""The digital twin: the rig in Isaac Sim, behind the real rig's ROS 2 interface.

    ./scripts/isaac_python.sh -m sparklab_sim.twin              # domain 42
    ROS_DOMAIN_ID=42 ./scripts/capx_python.sh -m capx_sparklab.agent

Publishes and serves what the two drivers, the camera node and the launch's
static TF do, with the same names, frames and units:

    /<arm>/joint_states, /<arm>/ee_pose, /tf <arm>_base -> <arm>_tool0
    /<arm>/arm/command, /<arm>/gripper/command (1 open .. 0 closed)
    /<arm>/park, /<arm>/zero_gravity (refused, as a SimRobot does)
    /cameras/<id>/{color,depth}/image_raw, /cameras/<id>/{color,depth}/camera_info
    /tf_static from extrinsics.yaml

and, for scoring practice runs only, each prop's true pose as /tf left_base -> twin/<name>.

Keep it on its own domain: the real rig runs on the default one.
"""

from __future__ import annotations

import argparse
import array
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from . import calibration, twin_scene

PHYSICS_HZ = 120
CAMERA_HZ = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", type=int, default=42, help="ROS_DOMAIN_ID to serve on")
    ap.add_argument("--gui", action="store_true", help="open Isaac's window")
    args = ap.parse_args()

    # --- Isaac boots here; nothing omni/isaacsim may be imported above ------
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui})
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / CAMERA_HZ)
    twin = twin_scene.build(world)
    world.reset()
    rclpy.init(domain_id=args.domain)
    node = TwinNode(twin)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    print(f"twin up on ROS domain {args.domain}", flush=True)

    start, steps, next_frame = time.monotonic(), 0, 0.0
    try:
        while app.is_running():
            for _ in range(4):
                executor.spin_once(timeout_sec=0)
            node.drive()
            world.step(render=False)
            steps += 1
            node.publish_state()
            if time.monotonic() >= next_frame:
                world.render()
                node.publish_cameras()
                node.publish_props()
                next_frame = time.monotonic() + 1 / CAMERA_HZ
            # Real time: commands arrive on the wall clock, so the physics must keep it.
            ahead = start + steps / PHYSICS_HZ - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        app.close()
    return 0


def _image(stamp, frame_id: str, pixels: np.ndarray, encoding: str) -> Image:
    height, width = pixels.shape[:2]
    msg = Image(height=height, width=width, encoding=encoding, step=pixels.strides[0])
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    msg.data = array.array("B", np.ascontiguousarray(pixels).tobytes())
    return msg


def _transform(stamp, parent: str, child: str, T: np.ndarray) -> TransformStamped:
    msg = TransformStamped(child_frame_id=child)
    msg.header.stamp, msg.header.frame_id = stamp, parent
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = T[:3, 3]
    w, x, y, z = calibration.quaternion_wxyz(T)
    msg.transform.rotation.w, msg.transform.rotation.x = w, x
    msg.transform.rotation.y, msg.transform.rotation.z = y, z
    return msg


class TwinNode(Node):
    """The rig's ROS 2 interface, served from the twin. Everything runs on the sim's thread.

    twin: what twin_scene.build() returned, after world.reset().
    """

    def __init__(self, twin: twin_scene.Twin) -> None:
        super().__init__("sparklab_twin")
        self._twin = twin
        self._max_step = np.asarray(calibration.driver_parameter("max_step"), dtype=float)
        self._park_s = float(calibration.driver_parameter("park_duration_s"))

        self._dof = {}
        self._target, self._gripper, self._park = {}, {}, {}
        self._state_pub, self._pose_pub = {}, {}
        for arm, articulation in twin.arms.items():
            articulation.initialize()
            names = articulation.dof_names
            self._dof[arm] = [names.index(j) for j in twin_scene.ARM_JOINTS + twin_scene.FINGER_JOINTS]
            kps, kds, efforts = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
            kps[self._dof[arm]], kds[self._dof[arm]] = twin_scene.STIFFNESS, twin_scene.DAMPING
            efforts[self._dof[arm]] = twin_scene.MAX_EFFORT
            controller = articulation.get_articulation_controller()
            controller.set_gains(kps=kps, kds=kds)
            controller.set_max_efforts(efforts)
            self._target[arm], self._gripper[arm] = self._arm_joints(arm), 1.0
            self._state_pub[arm] = self.create_publisher(JointState, f"/{arm}/joint_states", 10)
            self._pose_pub[arm] = self.create_publisher(PoseStamped, f"/{arm}/ee_pose", 10)
            self.create_subscription(Float64MultiArray, f"/{arm}/arm/command",
                                     lambda m, a=arm: self._on_arm(a, m), 10)
            self.create_subscription(Float64, f"/{arm}/gripper/command",
                                     lambda m, a=arm: self._gripper.__setitem__(a, float(np.clip(m.data, 0, 1))), 10)
            self.create_service(Trigger, f"/{arm}/park", lambda q, r, a=arm: self._on_park(a, r))
            self.create_service(SetBool, f"/{arm}/zero_gravity", self._on_zero_gravity)

        self._color_pub, self._depth_pub, self._info_pub = {}, {}, {}
        for cam_id, camera in twin.cameras.items():
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
            self._color_pub[cam_id] = self.create_publisher(Image, f"/cameras/{cam_id}/color/image_raw", 1)
            self._depth_pub[cam_id] = self.create_publisher(Image, f"/cameras/{cam_id}/depth/image_raw", 1)
            self._info_pub[cam_id] = [self.create_publisher(CameraInfo, f"/cameras/{cam_id}/{s}/camera_info", 1)
                                      for s in ("color", "depth")]

        for body in twin.props.values():
            body.initialize()
        self._tf = TransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        StaticTransformBroadcaster(self).sendTransform(
            [_transform(stamp, parent, child, T) for child, (parent, T) in calibration.transforms().items()])

    # ---- the arms --------------------------------------------------------------------
    def _arm_joints(self, arm: str) -> np.ndarray:
        return np.asarray(self._twin.arms[arm].get_joint_positions(), dtype=float)[self._dof[arm][:6]]

    def _opening(self, arm: str) -> float:
        """The gripper as the driver reports it: 1 open .. 0 closed."""
        fingers = np.asarray(self._twin.arms[arm].get_joint_positions(), dtype=float)[self._dof[arm][6:]]
        return float(np.clip(fingers.mean() / twin_scene.FINGER_OPEN, 0.0, 1.0))

    def _on_arm(self, arm: str, msg) -> None:
        target = np.asarray(msg.data, dtype=float)
        if target.shape != (6,) or arm in self._park:
            return
        # The driver's clamp, so the twin cuts the same jumps hardware would.
        present = self._arm_joints(arm)
        self._target[arm] = present + np.clip(target - present, -self._max_step, self._max_step)

    def _on_park(self, arm: str, response):
        self._park[arm] = (self._arm_joints(arm), time.monotonic())
        response.success, response.message = True, f"parking over {self._park_s:.1f}s"
        return response

    @staticmethod
    def _on_zero_gravity(request, response):
        response.success, response.message = False, "zero gravity refused: the twin has no zero-gravity mode"
        return response

    def drive(self) -> None:
        """Send every arm its current target; a park ramps to zero first, as the driver's does."""
        from isaacsim.core.utils.types import ArticulationAction

        for arm, articulation in self._twin.arms.items():
            if arm in self._park:
                start, began = self._park[arm]
                progress = min(1.0, (time.monotonic() - began) / self._park_s)
                self._target[arm], self._gripper[arm] = (1.0 - progress) * start, 0.0
                if progress >= 1.0:
                    del self._park[arm]
            finger = twin_scene.FINGER_OPEN * self._gripper[arm]
            targets = np.asarray(articulation.get_joint_positions(), dtype=float)
            targets[self._dof[arm]] = [*self._target[arm], finger, finger]
            articulation.apply_action(ArticulationAction(joint_positions=targets))

    def publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for arm, articulation in self._twin.arms.items():
            q = self._arm_joints(arm)
            velocity = np.asarray(articulation.get_joint_velocities(), dtype=float)[self._dof[arm][:6]]
            state = JointState(name=[f"joint_{i}" for i in range(1, 7)] + ["gripper"],
                               position=[*q, self._opening(arm)], velocity=[*velocity, 0.0], effort=[0.0] * 7)
            state.header.stamp = stamp
            self._state_pub[arm].publish(state)

            tool0 = twin_scene.tool0(q)
            transforms.append(_transform(stamp, f"{arm}_base", f"{arm}_tool0", tool0))
            pose = PoseStamped()
            pose.header = transforms[-1].header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tool0[:3, 3]
            w, x, y, z = calibration.quaternion_wxyz(tool0)
            pose.pose.orientation.w, pose.pose.orientation.x = w, x
            pose.pose.orientation.y, pose.pose.orientation.z = y, z
            self._pose_pub[arm].publish(pose)
        self._tf.sendTransform(transforms)

    def publish_props(self) -> None:
        """Each prop's true pose, as TF left_base -> twin/<name>. Ground truth: never for the robot."""
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for name, body in self._twin.props.items():
            position, wxyz = body.get_world_pose()
            T = calibration.matrix(position, np.roll(np.asarray(wxyz, dtype=float), -1))
            transforms.append(_transform(stamp, "left_base", f"twin/{name}", T))
        self._tf.sendTransform(transforms)

    # ---- the cameras -------------------------------------------------------------------
    def publish_cameras(self) -> None:
        """Colour, depth aligned to it in millimetres (0 = no reading), and the lens: one stamp each."""
        stamp = self.get_clock().now().to_msg()
        for cam_id, camera in self._twin.cameras.items():
            rgb, depth = camera.get_rgb(), camera.get_depth()
            if rgb is None or depth is None or rgb.size == 0 or depth.size == 0:
                continue
            spec = self._twin.specs[cam_id]
            millimetres = np.where(np.isfinite(depth) & (depth < 65.0), np.round(depth * 1000.0), 0).astype(np.uint16)
            self._color_pub[cam_id].publish(_image(stamp, cam_id, np.asarray(rgb, dtype=np.uint8)[..., :3], "rgb8"))
            self._depth_pub[cam_id].publish(_image(stamp, cam_id, millimetres, "16UC1"))
            info = CameraInfo(width=spec.width, height=spec.height, distortion_model="plumb_bob",
                              d=[0.0] * 5, k=spec.k, r=[1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                              p=[spec.fx, 0.0, spec.cx, 0.0, 0.0, spec.fy, spec.cy, 0.0, 0.0, 0.0, 1.0, 0.0])
            info.header.stamp, info.header.frame_id = stamp, cam_id
            for pub in self._info_pub[cam_id]:
                pub.publish(info)


if __name__ == "__main__":
    raise SystemExit(main())
