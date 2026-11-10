import os
import subprocess
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_prefix

from coding_agent_sparklab import robot


@pytest.fixture(scope="session")
def drivers():
    """Two simulated drivers, parking in 1 s, shared by every test that needs arms."""
    # Domain 94: nothing else of ours lives there. Set before robot first starts rclpy.
    os.environ.update(ROS_DOMAIN_ID="94", ROS_LOCALHOST_ONLY="1")
    driver = Path(get_package_prefix("sparklab_yam")) / "lib" / "sparklab_yam" / "driver"
    procs = [subprocess.Popen([str(driver), "--ros-args", "-r", f"__ns:=/{arm}", "-r", "__node:=yam_driver",
                               "-p", "sim:=true", "-p", "park_on_exit:=false",
                               "-p", "rest_on_start:=false", "-p", "park_duration_s:=1.0"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for arm in robot.ARMS]
    yield
    for p in procs:
        p.terminate()
        p.wait(timeout=10)
