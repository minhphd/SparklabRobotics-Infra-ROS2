from setuptools import find_packages, setup

package_name = "sparklab_cameras"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cameras = sparklab_cameras.node:main",
            "apply_presets = sparklab_cameras.apply_presets:main",
            "calibrate = sparklab_cameras.calibrate:main",
            "export_intrinsics = sparklab_cameras.export_intrinsics:main",
        ],
    },
)
