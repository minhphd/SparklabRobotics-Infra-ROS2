from glob import glob

from setuptools import find_packages, setup

package_name = "capx_sparklab"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["robot/*.urdf", "robot/*.yml", "robot/assets/*",
                                 "robot/graspgenx/*/*", "configs/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Foxglove fetches the arm meshes as package://capx_sparklab/robot/assets/...
        (f"share/{package_name}/robot/assets", glob("capx_sparklab/robot/assets/*.stl")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    license="Apache-2.0",
)
