#!/usr/bin/env python3

"""
Launch social-perception visualization.

FLOW
====
Resolved visualization topics
    |
    v
QoS-aware RViz config generator
    |
    v
RViz

RESPONSIBILITIES
================
This launch owns visualization only.

It MUST NOT own:
    - camera hardware
    - image ingress
    - YOLO inference
    - social perception processing
    - navigation
    - explanation
    - speech

RGB and depth displays are disabled by default at the system level.
When enabled, system.launch.py must provide the resolved topics so RViz
does not create an unintended second remote image subscription.
"""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    perception_share = get_package_share_directory(
        "g1_social_perception"
    )
    
    enable_ros4hri_visualization = LaunchConfiguration(
        "enable_ros4hri_visualization"
    )

    default_template = os.path.join(
        perception_share,
        "rviz",
        "person_depth_fusion.template.rviz",
    )

    rviz_template = LaunchConfiguration(
        "rviz_template"
    )

    rviz_runtime_config = LaunchConfiguration(
        "rviz_runtime_config"
    )

    rviz_qos_timeout = LaunchConfiguration(
        "rviz_qos_timeout"
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

    qos_generator = Node(
        package="g1_qos_manager",
        executable="generate_rviz_qos_config",
        name="social_rviz_qos_generator",
        output="screen",
        arguments=[
            "--template",
            rviz_template,
            "--output",
            rviz_runtime_config,
            "--rgb-topic",
            rgb_topic,
            "--depth-topic",
            aligned_depth_topic,
            "--yolo-topic",
            yolo_image_topic,
            "--markers-topic",
            person_markers_topic,
            "--fixed-frame",
            rviz_fixed_frame,
            "--timeout",
            rviz_qos_timeout,
            "--enable-ros4hri",
            enable_ros4hri_visualization,
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="social_perception_rviz",
        output="screen",
        arguments=[
            "-d",
            rviz_runtime_config,
        ],
    )
    

    start_rviz = RegisterEventHandler(
        OnProcessExit(
            target_action=qos_generator,
            on_exit=[rviz],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_template",
                default_value=default_template,
            ),

            DeclareLaunchArgument(
                "rviz_runtime_config",
                default_value=(
                    "/tmp/g1_social_perception_runtime.rviz"
                ),
            ),

            DeclareLaunchArgument(
                "rviz_qos_timeout",
                default_value="5.0",
            ),

            DeclareLaunchArgument(
                "rgb_topic",
                default_value=(
                    "/perception/ingress/color/image_raw"
                ),
                description=(
                    "Resolved RGB visualization topic."
                ),
            ),

            DeclareLaunchArgument(
                "aligned_depth_topic",
                default_value=(
                    "/perception/ingress/"
                    "aligned_depth/image_raw"
                ),
                description=(
                    "Resolved aligned-depth visualization topic."
                ),
            ),

            DeclareLaunchArgument(
                "person_markers_topic",
                default_value="/person_3d_markers",
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
                "enable_ros4hri_visualization",
                default_value="false",
                choices=["true", "false"],
                description=(
                    "Add ROS4HRI RViz displays to the generated "
                    "runtime configuration."
                ),
            ),

            qos_generator,
            start_rviz,
        ]
    )