from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    enable_realsense = LaunchConfiguration(
        "enable_realsense"
    )

    enable_realsense_tf = LaunchConfiguration(
        "enable_realsense_tf"
    )

    # ========================================================
    # REALSENSE DRIVER
    # ========================================================

    realsense_launch = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "realsense2_camera",
            "rs_launch.py",

            "camera_namespace:=camera",
            "camera_name:=camera",

            "enable_depth:=true",
            "enable_color:=true",

            "depth_module.depth_profile:=640x480x15",
            "rgb_camera.color_profile:=640x480x15",

            "enable_sync:=true",
            "align_depth.enable:=true",

            "pointcloud.enable:=false",
            "publish_tf:=true",

            "log_level:=info",
        ],

        additional_env={
            "ROS_DOMAIN_ID": "0",
            "ROS_LOCALHOST_ONLY": "0",
            "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
            "CYCLONEDDS_URI": (
                '<CycloneDDS><Domain id="any">'
                '<General>'
                '<NetworkInterfaceAddress>eth0</NetworkInterfaceAddress>'
                '<AllowMulticast>spdp</AllowMulticast>'
                '<MaxMessageSize>1400B</MaxMessageSize>'
                '<FragmentSize>1280B</FragmentSize>'
                '</General>'
                '<Internal>'
                '</Internal>'
                '</Domain></CycloneDDS>'
            ),
        },

        output="screen",

        condition=IfCondition(
            enable_realsense
        ),
    )

    # ========================================================
    # G1 CAMERA FRAME -> REALSENSE ROOT
    # ========================================================

    d435_to_camera_link_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d435_to_camera_link_tf",
        output="screen",

        condition=IfCondition(
            enable_realsense_tf
        ),

        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "d435_link",
            "camera_link",
        ],
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            "enable_realsense",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "enable_realsense_tf",
            default_value="true",
        ),

        realsense_launch,
        d435_to_camera_link_tf,
    ])