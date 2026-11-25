from setuptools import find_packages, setup
import os
from glob import glob


package_name = "g1_robot_collaboration"


setup(
    name=package_name,
    version="0.0.0",

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
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],

    install_requires=["setuptools"],
    zip_safe=True,

    maintainer="jean",
    maintainer_email="icmayoko18@gmail.com",

    description=(
        "Social collaboration and HRI decision management "
        "for the Unitree G1."
    ),

    license="Apache-2.0",

    entry_points={
        "console_scripts": [
            "collaboration_manager_node = "
            "g1_robot_collaboration.collaboration_manager_node:main",

            "trajectory_manager = "
            "g1_robot_collaboration.trajectory_manager:main",

            "pre_goal_announcement = "
            "g1_robot_collaboration.pre_goal_announcement_node:main",

            "trajectory_social_validator = "
            "g1_robot_collaboration.trajectory_social_validator:main",

            'directional_keepout_mask_node = '
            'g1_robot_collaboration.directional_keepout_mask_node:main',

            "robot_motion_state_monitor = "
            "g1_robot_collaboration.robot_motion_state_monitor_node:main",

        ],
    },

)