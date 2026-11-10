from setuptools import find_packages, setup

package_name = "coding_agent_sparklab"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["page.html", "prompt.md"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    license="Apache-2.0",
    entry_points={"console_scripts": ["ui = coding_agent_sparklab.ui:main"]},
)
