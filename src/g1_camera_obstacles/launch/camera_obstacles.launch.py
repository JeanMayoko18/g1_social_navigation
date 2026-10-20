#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():

    package_share = get_package_share_directory(
        'g1_camera_obstacles'
    )

    params_file = os.path.join(
        package_share,
        'config',
        'camera_obstacles.yaml',
    )

    return LaunchDescription(
        [
            Node(
                package='g1_camera_obstacles',
                executable='camera_obstacle_node',
                name='camera_obstacle_node',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
