"""
Launch the social-navigation decision layer.

                 social_system.yaml
                       │
              ┌────────┴─────────┐
              ▼                  ▼
SocialNavigationDecision   collaboration.launch.py
                                  │
                         ┌────────┼────────┐
                         ▼        ▼        ▼
                  Collaboration  Traj.   Validator
                     Manager     Manager
Responsibility:
    Convert social perception into deterministic social-navigation decisions
    and produce one final CollaborativeDecision.

Architecture:
    - Perception is owned by social_perception.launch.py.
    - SocialNavigationDecision remains deterministic.
    - CollaborationManager is the unique final semantic decision producer.
    - Navigation execution belongs exclusively to navigation.launch.py.
    - ROS interface configuration is loaded from the central social_system.yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    interaction = LaunchConfiguration("interaction")
    host = LaunchConfiguration("host")
    language = LaunchConfiguration("language")

    # -------------------------------------------------------------------------
    # CENTRAL CONFIGURATION
    # -------------------------------------------------------------------------

    config_file = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )

    # -------------------------------------------------------------------------
    # COLLABORATION LAUNCH
    # -------------------------------------------------------------------------

    collaboration_launch = os.path.join(
        get_package_share_directory("g1_robot_collaboration"),
        "launch",
        "collaboration.launch.py",
    )

    # -------------------------------------------------------------------------
    # SOCIAL NAVIGATION DECISION
    # -------------------------------------------------------------------------

    social_navigation_decision = Node(
        package="g1_social_perception",
        executable="social_navigation_decision",
        name="social_navigation_decision",
        output="screen",
        parameters=[
            config_file,
        ],
    )

    # -------------------------------------------------------------------------
    # COLLABORATION
    # -------------------------------------------------------------------------

    collaboration_manager = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            collaboration_launch
        ),
        launch_arguments={
            "interaction": interaction,
            "host": host,
            "language": language,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "interaction",
            default_value="none",
            choices=["none", "dialogue"],
            description=(
                "Interaction mode: deterministic social resolution "
                "or dialogue-based social resolution."
            ),
        ),

        DeclareLaunchArgument(
            "host",
            choices=["pc", "robot"],
            description="Host running this social-navigation layer.",
        ),

        DeclareLaunchArgument(
            "language",
            default_value="en",
            choices=["en", "fr", "es", "auto"],
            description="Human interaction language.",
        ),

        social_navigation_decision,
        collaboration_manager,
    ])