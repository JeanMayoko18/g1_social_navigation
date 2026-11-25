"""Launch candidate preparation, depth, odom TF and identity association."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("g1_social_perception"),
        "config",
        "person_depth_fusion.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/perception/transport/color",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/aligned_depth_to_color/camera_info",
        ),

        Node(
            package="g1_social_perception",
            executable="person_detection_source",
            name="person_detection_source",
            output="screen",
        ),

        Node(
            package="g1_social_perception",
            executable="person_depth_fusion",
            name="person_depth_fusion",
            output="screen",
            parameters=[
                config,
                {
                    "detections_topic": "/social_perception/detections_2d",
                    "output_topic": "/social_perception/detections_3d",
                    "depth_topic": LaunchConfiguration("depth_topic"),
                    "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                },
            ],
        ),

        Node(
            package="g1_social_perception",
            executable="person_frame_transform",
            name="person_frame_transform",
            output="screen",
            parameters=[{
                "input_topic": "/social_perception/detections_3d",
                "output_topic": "/social_perception/detections_3d_odom",
                "target_frame": "odom",
                "tf_timestamp_policy": "detection",
            }],
        ),

        Node(
            package="g1_social_perception",
            executable="person_tracking",
            name="person_tracking",
            output="screen",
            parameters=[{
                "input_topic": "/social_perception/detections_3d_odom",
                "output_topic": "/social_perception/detections_identified_odom",
                "expected_frame": "odom",
                "rgb_topic": LaunchConfiguration("rgb_topic"),
            }],
        ),
    ])