from glob import glob

from setuptools import find_packages, setup

package_name = "sparklab_gello"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.json")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gello = sparklab_gello.node:main",
        ],
    },
)
