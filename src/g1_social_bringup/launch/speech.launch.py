"""
G1 common speech subsystem.

FLOW
====
Text producers
    |
    +-- social explanation
    +-- navigation announcements
    +-- ROS4HRI compatibility adapter
    |
    v
/explainability/text
    |
    v
PiperTTSNode
    |
    +-- single speech queue
    +-- Piper synthesis
    +-- selected audio backend
    |
    v
/explainability/speech_status

RESPONSIBILITIES
================
This launch owns the final speech engine.

It starts exactly one PiperTTSNode on the host that owns the
speech role.

It does not generate explanation text.
It does not generate navigation announcements.
It does not launch ROS4HRI.
It does not launch navigation.
It does not launch hardware.

ExplanationText is currently the common speech request message.
SpeechStatus reports real synthesis/playback completion or failure.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)

from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def _launch_speech(context):
    runtime_host = (
        LaunchConfiguration("runtime_host")
        .perform(context)
        .strip()
        .lower()
    )

    ros_distro = (
        LaunchConfiguration("ros_distro")
        .perform(context)
        .strip()
        .lower()
    )

    audio_backend = (
        LaunchConfiguration("audio_backend")
        .perform(context)
        .strip()
        .lower()
    )

    if runtime_host not in ("pc", "robot"):
        raise RuntimeError(
            "runtime_host must be 'pc' or 'robot'."
        )

    if ros_distro not in ("foxy", "humble"):
        raise RuntimeError(
            "ros_distro must be 'foxy' or 'humble'."
        )

    if audio_backend not in ("auto", "aplay", "unitree"):
        raise RuntimeError(
            "audio_backend must be "
            "'auto', 'aplay', or 'unitree'."
        )

    social_bringup_share = get_package_share_directory(
        "g1_social_bringup"
    )

    config_file = os.path.join(
        social_bringup_share,
        "config",
        "social_system.yaml",
    )

    runtime_environment = {
        "ROS_DISTRO": ros_distro,
    }

    tts_node = Node(
        package="g1_robot_explainability",
        executable="piper_tts",
        name="piper_tts",
        output="screen",
        additional_env=runtime_environment,
        parameters=[
            config_file,
            {
                "enabled": True,
                "audio_backend": audio_backend,
                "audio_player": LaunchConfiguration(
                    "audio_player"
                ),
                "unitree_network_interface": (
                    LaunchConfiguration(
                        "unitree_network_interface"
                    )
                ),
                "unitree_volume": LaunchConfiguration(
                    "unitree_volume"
                ),
            },
        ],
    )

    return [tts_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "runtime_host",
                description="Speech host: pc | robot",
            ),

            DeclareLaunchArgument(
                "ros_distro",
                description="ROS distribution: foxy | humble",
            ),

            DeclareLaunchArgument(
                "audio_backend",
                default_value="auto",
                description=(
                    "Audio backend: auto | aplay | unitree"
                ),
            ),

            DeclareLaunchArgument(
                "audio_player",
                default_value="aplay",
                description="PC audio playback executable.",
            ),

            DeclareLaunchArgument(
                "unitree_network_interface",
                default_value="eth0",
                description=(
                    "Network interface used by Unitree audio."
                ),
            ),

            DeclareLaunchArgument(
                "unitree_volume",
                default_value="85",
                description="Unitree speaker volume: 0-100.",
            ),

            OpaqueFunction(function=_launch_speech),
        ]
    )