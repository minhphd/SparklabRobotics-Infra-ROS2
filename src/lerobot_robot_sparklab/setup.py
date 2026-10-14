from setuptools import find_packages, setup

# The `lerobot_robot_` prefix is load-bearing: every lerobot-* CLI imports the
# installed distributions that carry it. See DESIGN.md.
package_name = "lerobot_robot_sparklab"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["harness/web/*"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "record_manual = lerobot_robot_sparklab.record_manual:main",
            "rollout = lerobot_robot_sparklab.rollout.live:main",
            "harness = lerobot_robot_sparklab.harness.server:main",
        ],
    },
)
