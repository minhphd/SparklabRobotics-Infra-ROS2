from glob import glob

from setuptools import find_packages, setup

package_name = "sparklab_yam"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["model/*.xml", "model/*.urdf", "model/assets/*"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.json")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "driver = sparklab_yam.driver:main",
            "teleop = sparklab_yam.teleop_node:main",
            "keyboard = sparklab_yam.keyboard:main",
            "calibrate = sparklab_yam.calibration:main",
            "calibration_board = sparklab_yam.calibration:make_board_image",
        ],
    },
)
