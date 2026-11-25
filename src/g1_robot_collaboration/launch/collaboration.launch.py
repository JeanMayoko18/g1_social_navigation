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
from launch.conditions import UnlessCondition
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    interaction = LaunchConfiguration("interaction")
    manager_type = LaunchConfiguration("collaboration_manager_type")

    config_file = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )

    host = LaunchConfiguration("host")
    language = LaunchConfiguration("language")
    demo_robot_only_single_yes = LaunchConfiguration(
        "demo_robot_only_single_yes"
    )
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

    pre_goal_announcement = Node(
        package="g1_robot_collaboration",
        executable="pre_goal_announcement",
        name="pre_goal_announcement",
        output="screen",
        parameters=[{
            "language": "en",
            "cooldown_sec": 35.0,
            "max_event_age_sec": 2.0,
        }],
    )
    directional_keepout_mask = Node(
        package="g1_robot_collaboration",
        executable="directional_keepout_mask_node",
        name="directional_keepout_mask_node",
        output="screen",
        parameters=[
            config_file,
        ],
        condition=IfCondition(
            PythonExpression([
                "'", manager_type, "' == 'simple_cd' and '",
                interaction, "' == 'dialogue'",
            ])
        ),
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
            "collaboration_manager_type",
            default_value="original",
            choices=["original", "simple_cd"],
            description="Select the collaboration manager implementation",
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
        DeclareLaunchArgument(
            "anticipatory_planning_enabled",
            default_value="false",
            choices=["true", "false"],
            description="Enable anticipatory trajectory planning in dialogue mode.",
        ),

        DeclareLaunchArgument(
            "demo_robot_only_single_yes",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Demo HRI: robot-initiated question only; "
                "accept one explicit YES or unambiguous "
                "LEFT/RIGHT instruction without final "
                "confirmation question."
            ),
        ),

        # =====================================================================
        # ROBOT MOTION STATE
        # =====================================================================
        robot_motion_state_monitor,
        directional_keepout_mask,
        pre_goal_announcement,
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
                    "collaboration_manager_type": manager_type,
                    "anticipatory_planning_enabled": ParameterValue(
                        LaunchConfiguration("anticipatory_planning_enabled"),
                        value_type=bool,
                    ),
                    "demo_robot_only_single_yes": ParameterValue(
                        demo_robot_only_single_yes,
                        value_type=bool,
                    ),
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
            condition=UnlessCondition(
                PythonExpression([
                    "'", manager_type, "' == 'simple_cd'"
                ])
            ),
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
            condition=UnlessCondition(
                PythonExpression([
                    "'", manager_type, "' == 'simple_cd'"
                ])
            ),
            parameters=[
                config_file,
            ],
        ),

    ])