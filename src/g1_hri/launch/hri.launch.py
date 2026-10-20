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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hri_enabled = LaunchConfiguration("hri_enabled")

    hri_manager = Node(
        package="g1_hri",
        executable="hri_manager_node",
        name="hri_manager",
        output="screen",
        parameters=[
            {
                "enabled": hri_enabled,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hri_enabled",
                default_value="true",
                choices=["true", "false"],
                description="Enable project-level HRI processing.",
            ),
            hri_manager,
        ]
    )