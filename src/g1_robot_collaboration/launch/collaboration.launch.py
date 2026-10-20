"""
Launch the robot collaboration layer.

Responsibility:
    Start the collaboration FSM and the trajectory-generation /
    social-validation pipeline.

Interaction:
    interaction=none
        -> deterministic social resolution
        -> no human dialogue

    interaction=dialogue
        -> dialogue-based social resolution
        -> HRI dialogue enabled

Flow:
    SocialNavigationDecision
              |
              v
      CollaborationManager
              |
              +-----------------------------+
              |                             |
              v                             v
    TrajectoryManager          TrajectorySocialValidator
              |                             |
              v                             v
     CandidateTrajectory       TrajectorySocialValidation
              \                           /
               +-------------+-------------+
                             |
                             v
                  CollaborationManager
                             |
                             v
                  CollaborativeDecision

Notes:
    - This launch does not start perception.
    - This launch does not start Nav2.
    - This launch does not start the navigation gate.
    - This launch does not start ROS4HRI.
    - This launch does not start TTS.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    interaction = LaunchConfiguration("interaction")

    config_file = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )
    host = LaunchConfiguration("host")
    language = LaunchConfiguration("language")
    # -------------------------------------------------------------------------
    # ROBOT MOTION STATE 
    # -------------------------------------------------------------------------
    robot_motion_state_monitor = Node(
        package="g1_robot_collaboration",
        executable="robot_motion_state_monitor",
        name="robot_motion_state_monitor",
        output="screen",
        parameters=[
                config_file,
            ],
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
            description="Host on which this collaboration launch is running.",
        ),
        DeclareLaunchArgument(
            "language",
            default_value="en",
            choices=["en", "fr", "es", "auto"],
            description="Human speech recognition language.",
        ),

        # =====================================================================
        # ROBOT MOTION STATE
        # =====================================================================
        robot_motion_state_monitor,

        # =====================================================================
        # COLLABORATION FSM
        # =====================================================================

        Node(
            package="g1_robot_collaboration",
            executable="collaboration_manager_node",
            name="collaboration_manager",
            output="screen",
            parameters=[
                config_file,
                {
                    "interaction": interaction,
                },
            ],
        ),



        # =====================================================================
        # TRAJECTORY GENERATION
        # =====================================================================

        Node(
            package="g1_robot_collaboration",
            executable="trajectory_manager",
            name="trajectory_manager",
            output="screen",
            parameters=[
                config_file,
            ],
        ),

        # =====================================================================
        # TRAJECTORY SOCIAL VALIDATION
        # =====================================================================

        Node(
            package="g1_robot_collaboration",
            executable="trajectory_social_validator",
            name="trajectory_social_validator",
            output="screen",
            parameters=[
                config_file,
            ],
        ),

    ])