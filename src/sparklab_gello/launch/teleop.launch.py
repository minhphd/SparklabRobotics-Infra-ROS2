"""Both YAM arms and both GELLO leaders: a driver and a GELLO node per arm.

    ros2 launch sparklab_gello teleop.launch.py
    ros2 launch sparklab_gello teleop.launch.py sim:=true viewer:=true

Each leader homes to its arm on start-up. `/gello/arm` then arms both arms, or neither,
and `/gello/disarm` stops both; one arm alone is `/gello/left/arm` or `/gello/right/arm`.
Ctrl-C stops everything; the drivers park the arms to zero, and the leaders fold there too.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

# USB serial of each leader's servo adapter; the files are per unit (see config/).
LEADERS = {
    "left": ("5B8E115998", "yam_active_leader_eeprom.json", "gello_to_yam.yaml"),
    "right": ("5B79018636", "yam_active_leader_right_eeprom.json", "gello_to_yam_right.yaml"),
}


def generate_launch_description() -> LaunchDescription:
    params = str(Path(get_package_share_directory("sparklab_yam")) / "config" / "yam.yaml")
    config = Path(get_package_share_directory("sparklab_gello")) / "config"
    sim, viewer = LaunchConfiguration("sim"), LaunchConfiguration("viewer")
    # Sim drivers stall past 0.1 s when the shared PC is loaded; hardware keeps the default.
    state_timeout = PythonExpression(["0.5 if '", sim, "' == 'true' else 0.1"])
    # i2rt's kp with j5/j6 2.5x stiffer, plus velocity feed-forward, for this launch only.
    # See DESIGN.md#wrist-gains-and-velocity-feed-forward-in-teleoplaunchpy.
    kp = [80.0, 80.0, 80.0, 40.0, 25.0, 25.0]
    nodes = [Node(package="sparklab_gello", executable="pair", name="gello")]
    for side, (serial, calibration, mapping) in LEADERS.items():
        # Parking runs after SIGINT and takes park_duration_s, twice that if the arm lags;
        # launch's default 5 s before escalating to SIGTERM would cut it short.
        nodes.append(Node(package="sparklab_yam", executable="driver", name="yam_driver",
                          namespace=side,
                          parameters=[params, {"sim": sim, "viewer": viewer, "kp": kp,
                                               "velocity_feedforward": True}],
                          sigterm_timeout="30"))
        nodes.append(Node(package="sparklab_gello", executable="gello", name=side, namespace="gello", parameters=[{
            "arm": side,
            "port": f"/dev/serial/by-id/usb-1a86_USB_Single_Serial_{serial}-if00",
            "calibration_path": str(config / calibration),
            "mapping_path": str(config / mapping),
            "state_timeout_s": state_timeout,
        }]))
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="false",
                              description="i2rt SimRobots instead of CAN hardware"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="MuJoCo window per simulated arm (sim only)"),
        *nodes,
    ])
