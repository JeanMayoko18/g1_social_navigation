"""
HRI manager launch.

Flow
----

    CollaborationManager
            |
            | DialogueRequest
            v
       HRIManager
            |
            | HumanIntent
            v
    CollaborationManager

Responsibilities
----------------

This launch owns only the project-level HRI manager.

It does NOT:
- launch ROS4HRI;
- launch dialogue_manager;
- own speech synthesis;
- publish navigation commands.

ROS4HRI integration is owned by
g1_hri/launch/ros4hri_adapter.launch.py.

The system orchestrator decides whether the HRI manager and the
ROS4HRI compatibility layer are launched for interaction=dialogue.
"""

"""
Launch the project-level HRI manager.

Load shared configuration before applying explicit launch overrides.
ROS4HRI integration and speech synthesis are launched separately.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    hri_enabled = LaunchConfiguration("hri_enabled")
    params_file = LaunchConfiguration("params_file")

    default_params_file = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )

    hri_manager = Node(
        package="g1_hri",
        executable="hri_manager_node",
        name="hri_manager",
        output="screen",
        parameters=[
            params_file,
            {
                "enabled": ParameterValue(
                    hri_enabled,
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "hri_enabled",
            default_value="true",
            choices=["true", "false"],
            description="Enable project-level HRI processing.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to the shared ROS parameter YAML file.",
        ),
        hri_manager,
    ])