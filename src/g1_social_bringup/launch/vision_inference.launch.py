#!/usr/bin/env python3

"""
Launch the vision inference role.

FLOW
====
External RGB image
    |
    v
YOLO
    |
    v
/yolo/detections

RESPONSIBILITIES
================
This launch owns only vision inference.

It owns:
    - YOLO inference

It MUST NOT own:
    - RealSense acquisition
    - image transport / ingress
    - depth fusion
    - person tracking
    - social zones
    - social decision
    - navigation
    - explanation / LLM
    - speech / TTS
    - visualization

The RGB input topic is configurable so deployment orchestration can
select either the direct hardware topic or a local ImageIngress topic.

RealSense acquisition remains externally owned by rl_sar.
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
from launch.conditions import IfCondition


def generate_launch_description():

    yolo_bringup_share = get_package_share_directory(
        "yolo_bringup"
    )

    input_image_topic = LaunchConfiguration(
        "input_image_topic"
    )

    image_reliability = LaunchConfiguration(
        "image_reliability"
    )
    enable_yolo = LaunchConfiguration(
        "enable_yolo"
    )

    yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                yolo_bringup_share,
                "launch",
                "yolov11.launch.py",
            )
        ),
        launch_arguments={
            "input_image_topic": input_image_topic,
            "image_reliability": image_reliability,
        }.items(),
        condition=IfCondition(enable_yolo),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_image_topic",
                default_value=(
                    "/camera/camera/color/image_raw"
                ),
                description="RGB input topic for YOLO.",
            ),

            DeclareLaunchArgument(
                "image_reliability",
                default_value="2",
                description=(
                    "YOLO image subscription reliability."
                ),
            ),

            DeclareLaunchArgument(
                "enable_yolo",
                default_value="true",
                description="Enable YOLO vision inference.",
            ),

            yolo,
        ]
    )