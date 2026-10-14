"""Bring up the YAM-Ultra rig: a driver per arm, the cameras, the Quest bridge and the camera TF.

    ros2 launch sparklab_yam yam.launch.py
    ros2 launch sparklab_yam yam.launch.py sim:=true viewer:=true cameras:=false
    ros2 launch sparklab_yam yam.launch.py teleop:=true

Leave it running; recording, rollouts and the harness are clients of it.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _static_tf(t: dict) -> Node:
    x, y, z = map(str, t["translation"])
    qx, qy, qz, qw = map(str, t["rotation"])
    return Node(package="tf2_ros", executable="static_transform_publisher", name=f"tf_{t['child']}",
                arguments=["--x", x, "--y", y, "--z", z, "--qx", qx, "--qy", qy, "--qz", qz, "--qw", qw,
                           "--frame-id", t["parent"], "--child-frame-id", t["child"]])


def generate_launch_description() -> LaunchDescription:
    config = Path(get_package_share_directory("sparklab_yam")) / "config"
    params = str(config / "yam.yaml")
    arg = LaunchConfiguration
    args = [
        DeclareLaunchArgument("sim", default_value="false",
                              description="i2rt SimRobots instead of CAN hardware"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="MuJoCo window per simulated arm (sim only)"),
        DeclareLaunchArgument("cameras", default_value="true",
                              description="publish the RealSense cameras"),
        DeclareLaunchArgument("teleop", default_value="false",
                              description="Quest teleop drives the arms directly"),
        DeclareLaunchArgument("certfile", default_value=str(Path.cwd() / "cert.pem")),
        DeclareLaunchArgument("keyfile", default_value=str(Path.cwd() / "key.pem")),
    ]
    # Parking runs after SIGINT and takes park_duration_s; launch's default 5 s
    # before escalating to SIGTERM would cut it short.
    drivers = [
        Node(package="sparklab_yam", executable="driver", name="yam_driver", namespace=side,
             parameters=[params, {"sim": arg("sim"), "viewer": arg("viewer")}],
             sigterm_timeout="20")
        for side in ("left", "right")
    ]
    static_tf = [_static_tf(t) for t in yaml.safe_load((config / "extrinsics.yaml").read_text())["transforms"]]
    return LaunchDescription([
        *args,
        *drivers,
        *static_tf,
        Node(package="sparklab_cameras", executable="cameras", name="cameras",
             parameters=[{"rig": str(config / "cameras.yaml")}],
             condition=IfCondition(arg("cameras"))),
        Node(package="sparklab_quest", executable="bridge", name="quest_bridge",
             parameters=[{"rig": str(config / "cameras.yaml"),
                          "certfile": arg("certfile"), "keyfile": arg("keyfile")}]),
        Node(package="sparklab_yam", executable="teleop", name="yam_teleop",
             parameters=[params], sigterm_timeout="20",
             condition=IfCondition(arg("teleop"))),
    ])
