#!/usr/bin/env python3

"""
Launch the network image ingress boundary.

FLOW
====
rl_sar camera topics
    |
    | remote DDS boundary
    v
ImageIngress
    |
    +-- /perception/ingress/color/image_raw
    +-- /perception/ingress/aligned_depth/image_raw
    `-- /perception/ingress/aligned_depth/camera_info

RESPONSIBILITIES
================
This launch owns the camera network ingress on the destination host.

RGB and depth transport are independently selectable.

Downstream perception nodes must not subscribe directly to remote
hardware image topics while this ingress is active.

Hardware remains externally owned by rl_sar.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    ingress = Node(
        package="g1_social_perception",
        executable="image_ingress",
        name="image_ingress",
        output="screen",
        parameters=[
            {
                "enable_rgb": LaunchConfiguration("enable_rgb"),
                "enable_depth": LaunchConfiguration("enable_depth"),
                "enable_camera_info": LaunchConfiguration(
                    "enable_camera_info"
                ),
                "rgb_input_topic": LaunchConfiguration("rgb_input_topic"),
                "depth_input_topic": (
                    "/camera/camera/"
                    "aligned_depth_to_color/image_raw"
                ),
                "camera_info_input_topic": (
                    "/camera/camera/"
                    "aligned_depth_to_color/camera_info"
                ),
                "rgb_output_topic": (
                    "/perception/ingress/color/image_raw"
                ),
                "depth_output_topic": (
                    "/perception/ingress/"
                    "aligned_depth/image_raw"
                ),
                "camera_info_output_topic": (
                    "/perception/ingress/"
                    "aligned_depth/camera_info"
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_rgb",
                default_value="true",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "enable_depth",
                default_value="true",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "enable_camera_info",
                default_value="true",
                choices=["true", "false"],
            ),

            DeclareLaunchArgument(
                "rgb_input_topic",
                default_value="/camera/camera/color/image_raw",
            ),

            ingress,
        ]
    )