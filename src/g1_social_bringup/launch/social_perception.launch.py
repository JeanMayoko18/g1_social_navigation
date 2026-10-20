#!/usr/bin/env python3

"""
Launch the downstream social perception role.

FLOW
====
YOLO detections -----------+
                           |
Aligned depth image -------+--> person_depth_fusion
Camera info ---------------+          |
                                      v
                               frame transform
                                      |
                                      v
                               spatial tracking
                                      |
                                      v
                                social zones

The pipeline also contains the existing 2D person tracking stage.

RESPONSIBILITIES
================
This launch owns the downstream social perception processing chain.

It owns:
    - person tracking
    - person/depth fusion
    - frame transformation
    - spatial tracking
    - social zone generation

It MUST NOT own:
    - RealSense acquisition
    - YOLO inference
    - image transport / ingress
    - deployment policy
    - social navigation decision
    - navigation
    - explanation / LLM
    - speech / TTS
    - visualization

Input topics are resolved by system.launch.py according to deployment
and host placement. This launch does not decide whether an input is
local or remote.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    social_perception_share = get_package_share_directory(
        "g1_social_perception"
    )

    default_config = os.path.join(
        social_perception_share,
        "config",
        "person_depth_fusion.yaml",
    )

    perception_config = LaunchConfiguration(
        "perception_config"
    )

    depth_topic = LaunchConfiguration(
        "depth_topic"
    )

    camera_info_topic = LaunchConfiguration(
        "camera_info_topic"
    )

    language = LaunchConfiguration(
        "language"
    )

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                social_perception_share,
                "launch",
                "person_depth_fusion.launch.py",
            )
        ),
        launch_arguments={
            "config_file": perception_config,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "language": language,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "perception_config",
                default_value=default_config,
                description=(
                    "Configuration for the downstream "
                    "social perception pipeline."
                ),
            ),

            DeclareLaunchArgument(
                "depth_topic",
                default_value=(
                    "/camera/camera/"
                    "aligned_depth_to_color/image_raw"
                ),
                description=(
                    "Aligned depth input selected by the "
                    "system orchestrator."
                ),
            ),

            DeclareLaunchArgument(
                "camera_info_topic",
                default_value=(
                    "/camera/camera/"
                    "aligned_depth_to_color/camera_info"
                ),
                description=(
                    "Aligned depth CameraInfo input selected "
                    "by the system orchestrator."
                ),
            ),
            
            DeclareLaunchArgument(
                "language",
                default_value="en",
                choices=["en", "es"],
                description=(
                    "System-wide language propagated to "
                    "ExplanationContext."
                ),
            ),

            pipeline,
        ]
    )