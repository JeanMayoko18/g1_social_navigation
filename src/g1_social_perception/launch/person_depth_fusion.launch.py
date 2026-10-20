#!/usr/bin/env python3

"""
G1 social perception pipeline.

FLOW
====
External hardware stack (rl_sar)
    |
    +-- RGB image
    +-- aligned depth image
    +-- camera info
    |
    v
YOLO detections
    |
    v
person_tracking
    |
    v
person_depth_fusion
    |
    v
person_frame_transform
    |
    v
person_spatial_tracking
    |
    v
social_zone_generator
    |
    v
/social_perception/social_zones

RESPONSIBILITIES
================
This launch owns only the social perception processing chain.

External inputs:
    - RealSense topics are provided by rl_sar.
    - YOLO detections are provided by yolo_bringup.

This launch MUST NOT own:
    - RealSense acquisition
    - YOLO
    - Nav2
    - social navigation decision
    - collaboration
    - explanation / LLM
    - speech / TTS
    - RViz

The social decision layer consumes the output of this pipeline
but is launched separately.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    perception_share = get_package_share_directory(
        "g1_social_perception"
    )

    default_config = os.path.join(
        perception_share,
        "config",
        "person_depth_fusion.yaml",
    )

    config_file = LaunchConfiguration("config_file")
    

    return LaunchDescription(
        [
            # ---------------------------------------------------------
            # Arguments
            # ---------------------------------------------------------

            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description=(
                    "Shared configuration for the social "
                    "perception pipeline."
                ),
            ),
            
            DeclareLaunchArgument(
                "depth_topic",
                default_value=(
                    "/camera/camera/"
                    "aligned_depth_to_color/image_raw"
                ),
                description="Aligned depth input.",
            ),
            
            DeclareLaunchArgument(
                "language",
                default_value="en",
                choices=["en", "es"],
                description=(
                    "Language propagated to ExplanationContext."
                ),
            ),

            DeclareLaunchArgument(
                "camera_info_topic",
                default_value=(
                    "/camera/camera/"
                    "aligned_depth_to_color/camera_info"
                ),
                description="Aligned depth CameraInfo input.",
            ),

            # ---------------------------------------------------------
            # 2D person tracking
            # ---------------------------------------------------------

            Node(
                package="g1_social_perception",
                executable="person_tracking",
                name="person_tracking",
                output="screen",
                parameters=[
                    config_file,
                ],
            ),

            # ---------------------------------------------------------
            # 2D + depth -> 3D person detections
            # ---------------------------------------------------------

            Node(
                package="g1_social_perception",
                executable="person_depth_fusion",
                name="person_depth_fusion",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "camera_info_topic": LaunchConfiguration(
                            "camera_info_topic"
                        ),
                    },
                ],
            ),

            # ---------------------------------------------------------
            # Camera frame -> robot base frame
            # ---------------------------------------------------------

            Node(
                package="g1_social_perception",
                executable="person_frame_transform",
                name="person_frame_transform",
                output="screen",
                parameters=[
                    config_file,
                ],
            ),

            # ---------------------------------------------------------
            # Temporal/spatial person tracking
            # ---------------------------------------------------------

            Node(
                package="g1_social_perception",
                executable="person_spatial_tracking",
                name="person_spatial_tracking",
                output="screen",
                parameters=[
                    config_file,
                ],
            ),

            # ---------------------------------------------------------
            # Social zones
            # ---------------------------------------------------------

            Node(
                package="g1_social_perception",
                executable="social_zone_generator",
                name="social_zone_generator",
                output="screen",
                parameters=[
                    config_file,
                ],
            ),
            
        ]
    )