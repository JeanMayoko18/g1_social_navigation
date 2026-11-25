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
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    interaction = LaunchConfiguration("interaction")
    manager_type = LaunchConfiguration("collaboration_manager_type")
    host = LaunchConfiguration("host")
    language = LaunchConfiguration("language")
    demo_robot_only_single_yes = LaunchConfiguration(
        "demo_robot_only_single_yes"
    )

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
            "anticipatory_planning_enabled": LaunchConfiguration(
                "anticipatory_planning_enabled"
            ),
            "collaboration_manager_type": manager_type,
            "language": language,
            "demo_robot_only_single_yes": demo_robot_only_single_yes,
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
            "demo_robot_only_single_yes",
            default_value="false",
            choices=["true", "false"],
            description="Enable shortened robot-initiated demo dialogue.",
        ),

        DeclareLaunchArgument(
            "collaboration_manager_type",
            default_value="original",
            choices=["original", "simple_cd"],
            description="Select the collaboration manager implementation",
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
        DeclareLaunchArgument(
            "anticipatory_planning_enabled",
            default_value="false",
            choices=["true", "false"],
        ),

        social_navigation_decision,
        collaboration_manager,
    ])