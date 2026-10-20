"""
Launch the navigation execution subsystem.

Flow:

    Public NavigateToPose
            |
            v
    NavigationGoalGate
            |
            | /navigate_to_pose_internal
            v
           Nav2
            |
            v
       /cmd_vel pipeline

Social execution input:

    /social_navigation/collaborative_decision
            |
            v
    NavigationGoalGate

Responsibilities:
    - launch the existing g1_nav2 navigation stack;
    - activate exactly one NavigationGoalGate for social profiles;
    - keep the public NavigateToPose API stable;
    - keep the Nav2 action internal when the gate is active.
    - publish deterministic navigation speech requests to the common
      speech API;

This launch does not start:
    - hardware;
    - perception;
    - social decision;
    - collaboration;
    - explanation;
    - the final speech engine;
    - ROS4HRI;
    - visualization.

Hardware remains externally owned by rl_sar.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    nav2_share = get_package_share_directory("g1_nav2")

    nav2_launch = os.path.join(
        nav2_share,
        "launch",
        "g1_bringup.launch.py",
    )

    config_file = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )

    navigation_goal_gate = Node(
        package="g1_nav2",
        executable="navigation_goal_gate",
        name="navigation_goal_gate",
        output="screen",
        parameters=[
            config_file,
        ],
        condition=IfCondition(
            LaunchConfiguration("enable_navigation_goal_gate")
        ),
    )


    navigation_interaction_speech = Node(
        package="g1_nav2",
        executable="navigation_interaction_speech",
        name="navigation_interaction_speech",
        output="screen",
        parameters=[
            config_file,
            {
                "enabled": LaunchConfiguration(
                    "enable_navigation_announcements"
                ),
                "language": LaunchConfiguration("language"),
                "enable_goal_speech": LaunchConfiguration(
                    "enable_direct_goal_speech"
                ),
            },
        ],
        condition=IfCondition(
            LaunchConfiguration(
                "enable_navigation_announcements"
            )
        ),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "profile",
            default_value="A",
            choices=["A", "B", "C", "D"],
            description="Navigation/perception profile.",
        ),

        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(
                os.path.expanduser("~"),
                "g1_ws",
                "maps",
                "mic.yaml",
            ),
            description="Nav2 map YAML.",
        ),

        DeclareLaunchArgument(
            "controller",
            default_value="dwb",
            description="Nav2 controller selection.",
        ),

        DeclareLaunchArgument(
            "navigation_speed",
            default_value="0.20",
            description="Maximum navigation speed.",
        ),

        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "ros_distro",
            default_value="auto",
            choices=["auto", "foxy", "humble"],
        ),

                DeclareLaunchArgument(
            "enable_navigation_announcements",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Publish deterministic navigation announcements "
                "to the common speech API."
            ),
        ),

        DeclareLaunchArgument(
            "enable_direct_goal_speech",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Let navigation_interaction_speech announce goal "
                "reception. Keep false when NavigationGoalGate "
                "owns that announcement."
            ),
        ),

        DeclareLaunchArgument(
            "language",
            default_value="es",
            choices=["es", "en"],
            description="Navigation announcement language.",
        ),

        DeclareLaunchArgument(
            "enable_navigation_goal_gate",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Start the social navigation gate. "
                "System orchestration enables it for profiles C and D."
            ),
        ),

        DeclareLaunchArgument(
            "navigation_mode",
            default_value="nav2",
            choices=["nav2", "slam"],
            description="Navigation mode forwarded to g1_nav2.",
        ),

        DeclareLaunchArgument(
            "start_rviz",
            default_value="false",
            choices=["true", "false"],
            description="Start the standard Nav2 RViz instance.",
        ),

        DeclareLaunchArgument(
            "start_visual_odometry",
            default_value="true",
            choices=["true", "false"],
            description="Start the visual odometry pipeline.",
        ),

        DeclareLaunchArgument(
            "enable_locomotion_calibration",
            default_value="false",
            choices=["true", "false"],
            description="Enable command-level locomotion calibration.",
        ),

        DeclareLaunchArgument(
            "locomotion_calibration_profile",
            default_value="previous_reference_scaled",
            description="Locomotion calibration profile.",
        ),

        DeclareLaunchArgument(
            "forward_lateral_compensation_profile",
            default_value="previous_reference_scaled",
            description="Forward lateral compensation profile.",
        ),

        DeclareLaunchArgument(
            "forward_heading_compensation_profile",
            default_value="previous_reference_scaled",
            description="Forward heading compensation profile.",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                "navigation_mode": LaunchConfiguration(
                    "navigation_mode"
                ),
                "perception_profile": LaunchConfiguration(
                    "profile"
                ),
                "map": LaunchConfiguration(
                    "map"
                ),
                "controller": LaunchConfiguration(
                    "controller"
                ),
                "navigation_speed": LaunchConfiguration(
                    "navigation_speed"
                ),
                "use_sim_time": LaunchConfiguration(
                    "use_sim_time"
                ),
                "ros_distro": LaunchConfiguration(
                    "ros_distro"
                ),
                "start_rviz": LaunchConfiguration(
                    "start_rviz"
                ),
                "start_visual_odometry": LaunchConfiguration(
                    "start_visual_odometry"
                ),
                "enable_locomotion_calibration": LaunchConfiguration(
                    "enable_locomotion_calibration"
                ),
                "locomotion_calibration_profile": LaunchConfiguration(
                    "locomotion_calibration_profile"
                ),
                "forward_lateral_compensation_profile": LaunchConfiguration(
                    "forward_lateral_compensation_profile"
                ),
                "forward_heading_compensation_profile": LaunchConfiguration(
                    "forward_heading_compensation_profile"
                ),
                # The gate is required only for social profiles.
                # This flag will be resolved by system.launch.py in the
                # next integration patch.
                "enable_navigation_goal_gate": LaunchConfiguration(
                    "enable_navigation_goal_gate"
                ),
            }.items(),
        ),

        navigation_goal_gate,
        navigation_interaction_speech,
    ])