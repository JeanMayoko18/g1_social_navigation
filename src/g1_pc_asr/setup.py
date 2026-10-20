from setuptools import find_packages, setup

package_name = "g1_pc_asr"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jean",
    maintainer_email="jean@todo.todo",
    description="PC microphone ASR bridge for ROS4HRI using Faster-Whisper.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pc_asr = g1_pc_asr.pc_asr_node:main",
        ],
    },
)
