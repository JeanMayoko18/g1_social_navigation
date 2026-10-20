#!/usr/bin/env python3

"""
Launch G1 system visualization.

FLOW
====

            rl_sar / hardware
                |
                | raw camera topics
                v
            +--------------------------+
            | system.launch.py         |
            |                          |
            | deployment resolution    |
            | role placement           |
            | ingress decision         |
            +------------+-------------+
                        |
        +-----+-----------------------+
        |                             |
        v                             v
    image_ingress                local hardware
    when required                when already robot-local
        |
        v
    /perception/ingress/*
        |
        +----------+-------------+
        |          |             |
        v          v             v
        YOLO    depth fusion   optional RViz


RESPONSIBILITIES
================
This launch owns visualization only.

It owns:
    - visualization-only marker generation
    - social perception RViz
    - optional raw RGB/depth visualization

It MUST NOT own:
    - RealSense acquisition
    - image ingress
    - YOLO inference
    - social perception processing
    - social decision
    - navigation
    - explanation / LLM
    - speech / TTS

Visualization marker nodes may consume functional perception outputs,
but they must never feed data back into perception or decision making.

Starting or stopping visualization must never start, stop, or otherwise
control functional perception.

Raw RGB/depth visualization is opt-in.

The system orchestrator resolves which image topics are safe for the
current deployment.
"""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node



def generate_launch_description():

    social_perception_share = get_package_share_directory(
        "g1_social_perception"
    )
    default_social_perception_config = os.path.join(
        social_perception_share,
        "config",
        "person_depth_fusion.yaml",
    )

    social_perception_config = LaunchConfiguration(
        "social_perception_config"
    )

    enable_visualization = LaunchConfiguration(
        "enable_visualization"
    )
    enable_ros4hri_visualization = LaunchConfiguration(
        "enable_ros4hri_visualization"
    )

    rgb_topic = LaunchConfiguration(
        "rgb_topic"
    )

    aligned_depth_topic = LaunchConfiguration(
        "aligned_depth_topic"
    )

    person_markers_topic = LaunchConfiguration(
        "person_markers_topic"
    )

    yolo_image_topic = LaunchConfiguration(
        "yolo_image_topic"
    )

    rviz_fixed_frame = LaunchConfiguration(
        "rviz_fixed_frame"
    )

    rviz_qos_timeout = LaunchConfiguration(
        "rviz_qos_timeout"
    )

    person_3d_marker = Node(
        package="g1_social_perception",
        executable="person_3d_marker",
        name="person_3d_marker",
        output="screen",
        parameters=[
            social_perception_config,
            {
                "output_topic": person_markers_topic,
            },
        ],
        condition=IfCondition(enable_visualization),
        #condition=UnlessCondition(enable_ros4hri_visualization),
    )


    social_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                social_perception_share,
                "launch",
                "social_rviz.launch.py",
            )
        ),
        launch_arguments={
            "rgb_topic": rgb_topic,
            "aligned_depth_topic": aligned_depth_topic,
            "person_markers_topic": person_markers_topic,
            "yolo_image_topic": yolo_image_topic,
            "rviz_fixed_frame": rviz_fixed_frame,
            "rviz_qos_timeout": rviz_qos_timeout,
            "enable_ros4hri_visualization": (
                enable_ros4hri_visualization
            ),
        }.items(),
        condition=IfCondition(enable_visualization),
    )

    ros4hri_visualization_adapter = Node(
        package="g1_social_perception",
        executable="ros4hri_visualization_adapter",
        name="ros4hri_visualization_adapter",
        output="screen",
        condition=IfCondition(enable_ros4hri_visualization),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_visualization",
                default_value="true",
            ),

            DeclareLaunchArgument(
                "rgb_topic",
                default_value=(
                    "/visualization/raw_images_disabled/rgb"
                ),
                description=(
                    "Resolved RGB topic. The default intentionally "
                    "does not subscribe to a camera stream."
                ),
            ),

            DeclareLaunchArgument(
                "aligned_depth_topic",
                default_value=(
                    "/visualization/raw_images_disabled/depth"
                ),
                description=(
                    "Resolved depth topic. The default intentionally "
                    "does not subscribe to a camera stream."
                ),
            ),

            DeclareLaunchArgument(
                "enable_ros4hri_visualization",
                default_value="false",
                choices=["true", "false"],
                description=(
                    "Enable optional ROS4HRI human visualization. "
                    "The native G1 person markers remain available independently."
                ),
            ),
            DeclareLaunchArgument(
                "person_markers_topic",
                default_value="/person_3d_markers",
            ),
            
            DeclareLaunchArgument(
                "social_perception_config",
                default_value=default_social_perception_config,
                description=(
                    "Configuration used by visualization-only social "
                    "perception marker nodes."
                ),
            ),

            DeclareLaunchArgument(
                "yolo_image_topic",
                default_value="/yolo/dbg_image",
            ),

            DeclareLaunchArgument(
                "rviz_fixed_frame",
                default_value="base_footprint",
            ),

            DeclareLaunchArgument(
                "rviz_qos_timeout",
                default_value="5.0",
            ),

            person_3d_marker,
            ros4hri_visualization_adapter,
            social_rviz,
        ]
    )