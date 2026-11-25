
#!/usr/bin/env python3
"""Optional identity benchmark: M1-M4, BoT-SORT, ReID, recorder."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    share = get_package_share_directory("g1_social_perception")

    baseline_config = os.path.join(
        share, "config", "benchmark", "botsort_baseline.yaml"
    )
    reid_config = os.path.join(
        share, "config", "benchmark", "botsort_reid.yaml"
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            "enable_ablation",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "enable_botsort",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "enable_benchmark_recorder",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "benchmark_input_topic",
            default_value="/social_perception/detections_enriched",
        ),

        DeclareLaunchArgument(
            "benchmark_rgb_topic",
            default_value="/camera/color/image_raw",
        ),

        DeclareLaunchArgument(
            "benchmark_model_path",
            default_value="/home/jean/g1_ws/models/yolo11n.pt",
        ),

        DeclareLaunchArgument(
            "benchmark_conda_python",
            default_value=(
                "/home/jean/miniconda3/envs/yolos/bin/python"
            ),
        ),

        DeclareLaunchArgument(
            "benchmark_output_directory",
            default_value="/home/jean/g1_ws/benchmark_results",
        ),

        # -------------------------------------------------
        # M1-M4 independent ablation
        # -------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_ablation",
            name="person_ablation",
            output="screen",
            condition=IfCondition(
                LaunchConfiguration("enable_ablation")
            ),
            parameters=[{
                "input_topic": LaunchConfiguration(
                    "benchmark_input_topic"
                ),
                "persistent_memory_enabled": False,
            }],
        ),

        # -------------------------------------------------
        # BoT-SORT / BoT-SORT + ReID
        #
        # ROS 2: system Python
        # GPU worker: Conda yolos
        # -------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_botsort_bridge",
            name="person_botsort_bridge",
            output="screen",
            condition=IfCondition(
                LaunchConfiguration("enable_botsort")
            ),

            parameters=[{
                "image_topic": LaunchConfiguration(
                    "benchmark_rgb_topic"
                ),
                "conda_python": LaunchConfiguration(
                    "benchmark_conda_python"
                ),
                "model_path": LaunchConfiguration(
                    "benchmark_model_path"
                ),
                "baseline_config": baseline_config,
                "reid_config": reid_config,
                "confidence": 0.25,
                "image_size": 640,
                "jpeg_quality": 90,
                "maximum_processing_hz": 0.0,
                "worker_timeout_sec": 30.0,
                "diagnostics_period_sec": 10.0,
            }],

        ),

        # -------------------------------------------------
        # Six-method recorder
        # -------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="identity_benchmark_recorder",
            name="identity_benchmark_recorder",
            output="screen",
            condition=IfCondition(
                LaunchConfiguration("enable_benchmark_recorder")
            ),
            parameters=[{
                "output_directory": LaunchConfiguration(
                    "benchmark_output_directory"
                ),
                "log_period_sec": 10.0,
                "maximum_file_size_mb": 512,
            }],
        ),
    ])
