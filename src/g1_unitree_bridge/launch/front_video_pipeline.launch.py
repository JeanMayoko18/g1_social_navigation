#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction

from launch_ros.actions import Node


def generate_launch_description():
    """Launch the Unitree front-camera ROS 2 bridge pipeline."""

    package_share = get_package_share_directory(
        'g1_unitree_bridge'
    )

    bridge_config = os.path.join(
        package_share,
        'config',
        'front_video_bridge.yaml'
    )

    enable_rate_probe = LaunchConfiguration(
        'enable_rate_probe'
    )

    # ---------------------------------------------------------------------
    # UNITREE FRONT VIDEO BRIDGE
    # ---------------------------------------------------------------------
    bridge_node = Node(
        package='g1_unitree_bridge',
        executable='front_video_bridge',
        name='front_video_bridge',
        output='screen',
        parameters=[bridge_config],
    )


    # ---------------------------------------------------------------------
    # IMAGE RATE DIAGNOSTIC PROBE
    # ---------------------------------------------------------------------
    rate_probe_node = Node(
        package='g1_unitree_bridge',
        executable='image_rate_probe',
        name='image_rate_probe',
        output='screen',
        condition=IfCondition(enable_rate_probe),
    )

    delayed_rate_probe = TimerAction(
        period=2.0,
        actions=[
            rate_probe_node,
        ],
    )

    return LaunchDescription([
        # ---------------------------------------------------------------------
        # LAUNCH ARGUMENTS
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'enable_rate_probe',
            default_value='true',
            description=(
                'Enable the native C++ image-rate diagnostic subscriber.'
            ),
        ),

        # ---------------------------------------------------------------------
        # UNITREE FRONT VIDEO BRIDGE
        # ---------------------------------------------------------------------
        bridge_node,

        # ---------------------------------------------------------------------
        # IMAGE RATE DIAGNOSTIC PROBE
        # ---------------------------------------------------------------------
        delayed_rate_probe,
    ])