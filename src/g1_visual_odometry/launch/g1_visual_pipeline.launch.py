#!/usr/bin/env python3
"""
File: g1_visual_pipeline.launch.py

Role:
-----
Launch the initial G1 visual odometry pipeline.

How it works:
-------------
1. Starts the C++ H264 decoder node that converts /frontvideostream
   into /camera/image_raw and /camera/camera_info.

2. Starts a visual odometry stub node that subscribes to:
   - /camera/image_raw
   - /camera/camera_info
   - /dog_imu_raw

3. Publishes a placeholder odometry topic:
   - /odom/visual

4. Optionally publishes TF:
   - odom -> base_footprint

This launch file is intended as the first integration step before replacing
the stub node with a real VO/VIO backend.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


ekf_config = os.path.join(
        get_package_share_directory("g1_visual_odometry"),
        "params",
        "ekf_visual.yaml"
)

def generate_launch_description():

    # ---------------------------------------------------------
    # 1) Video bridge: H264 -> /camera/image_raw + /camera/camera_info
    # ---------------------------------------------------------
    video_bridge = Node(
        package="g1_video_bridge_cpp",
        executable="frontvideo_to_image_cpp",
        name="frontvideo_to_image_cpp",
        output="screen",
        parameters=[
            {
                "out_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "frame_id": "camera_link",
                "publish_rate": 10.0,
                "fx": 900.0,
                "fy": 900.0,
                "cx": 640.0,
                "cy": 360.0,
            }
        ],
    )


    visual_stub = Node(
        package="g1_visual_odometry",
        executable="visual_odom_stub_node",
        name="visual_odom_stub_node",
        output="screen",
        parameters=[
            {
                "image_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "imu_topic": "/dog_imu_raw",
                "odom_topic": "/odom/visual",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
                "publish_tf": True,
                "publish_rate_hz": 20.0,
            }
        ],
    )

    # ---------------------------------------------------------
    # 2) Bootstrap visual odometry node
    # ---------------------------------------------------------

    visual_odometry_bootstrap_node = Node(
        package="g1_visual_odometry",
        executable="visual_odometry_bootstrap_node",
        name="visual_odometry_bootstrap_node",
        output="screen",
        parameters=[{
            "image_topic": "/camera/image_raw",
            "camera_info_topic": "/camera/camera_info",
            "imu_topic": "/dog_imu_raw",
            "odom_topic": "/odom/visual",
            "odom_frame": "odom",
            "base_frame": "base_footprint",
            "publish_tf": False, #True, #False,
            "meters_per_pixel": 0.0025,
            "max_features": 100,
            "min_tracked_points": 10,
            "max_dt_s": 0.8
        }]
    )

    # ---------------------------------------------------------
    # 3) Visual pipeline monitor
    # ---------------------------------------------------------

    visual_monitor = Node(
        package="g1_visual_odometry",
        executable="visual_odom_monitor_node",
        name="visual_odom_monitor_node",
        output="screen",
        parameters=[
            {
                "image_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "imu_topic": "/dog_imu_raw",
                "report_period_s": 2.0,
                "stale_timeout_s": 1.0,
                "max_samples": 100,
            }
        ],
    )

    # ---------------------------------------------------------
    # 4) Visual-inertial buffer node
    # ---------------------------------------------------------

    visual_inertial_buffer = Node(
        package="g1_visual_odometry",
        executable="visual_inertial_buffer_node",
        name="visual_inertial_buffer_node",
        output="screen",
        parameters=[
            {
                "image_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "imu_topic": "/dog_imu_raw",
                "imu_buffer_duration_s": 5.0,
                "imu_pre_window_s": 0.10,
                "imu_post_window_s": 0.02,
                "min_imu_samples_per_image": 3,
            }
        ],
    )

    # ---------------------------------------------------------
    # 5) EKF node: fuse visual odometry + IMU
    # ---------------------------------------------------------
    
    ekf_visual = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_visual_filter_node",
        output="screen",
        parameters=[ekf_config],
        remappings=[
            ("/odometry/filtered", "/odom"),
        ],
    )

    return LaunchDescription([
        video_bridge,
        visual_odometry_bootstrap_node,
        visual_monitor,
        visual_inertial_buffer,
        #ekf_visual,
    ])