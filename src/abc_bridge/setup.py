from setuptools import find_packages, setup

package_name = "abc_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge = abc_bridge.node:main",
        ],
    },
)
