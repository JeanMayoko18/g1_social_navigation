
#!/usr/bin/env python3
"""
G1 social perception pipeline.

External:
    - rl_sar: camera RGB, aligned depth, camera info
    - yolo_bringup: YOLO detections

Production:
    detection source -> depth fusion -> odom transform
    -> appearance extraction -> identity tracking
    -> spatial tracking -> social zones

Optional:
    identity benchmark (M1-M4, BoT-SORT, ReID, recorder)

No acquisition, YOLO bringup, Nav2 or social decision
process is started here.
"""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch.conditions import IfCondition
from launch.substitutions import PythonExpression

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    share = get_package_share_directory(
        "g1_social_perception"
    )

    config = os.path.join(
        share,
        "config",
        "person_depth_fusion.yaml",
    )

    benchmark_launch = os.path.join(
        share,
        "launch",
        "person_identity_benchmark.launch.py",
    )


    identity_backend = LaunchConfiguration("identity_backend")

    use_yolo_identity = IfCondition(
        PythonExpression([
            "'", identity_backend, "' == 'yolo_track'"
        ])
    )

    use_multimodal_identity = IfCondition(
        PythonExpression([
            "'", identity_backend, "' == 'multimodal'"
        ])
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            "config_file",
            default_value=config,
        ),

        DeclareLaunchArgument(
            "identity_backend",
            default_value="yolo_track",
            choices=["yolo_track", "multimodal"],
            description="Identity authority: YOLO tracking or legacy multimodal.",
        ),

        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/perception/transport/color",
        ),

        DeclareLaunchArgument(
            "depth_topic",
            default_value=(
                "/camera/camera/"
                "aligned_depth_to_color/image_raw"
            ),
        ),

        DeclareLaunchArgument(
            "camera_info_topic",
            default_value=(
                "/camera/camera/"
                "aligned_depth_to_color/camera_info"
            ),
        ),

        DeclareLaunchArgument(
            "language",
            default_value="en",
            choices=["en", "es"],
        ),

        DeclareLaunchArgument(
            "enable_benchmark",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "benchmark_ablation",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "benchmark_botsort",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "benchmark_recorder",
            default_value="true",
        ),

        # --------------------------------------------------
        # 1. External YOLO detections -> candidate preparation
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_detection_source",
            name="person_detection_source",
            output="screen",
        ),

        # --------------------------------------------------
        # 2. Depth fusion
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_depth_fusion",
            name="person_depth_fusion",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file"),
                {
                    "detections_topic": 
                        "/social_perception/detections_source",

                    "output_topic":
                        "/social_perception/detections_3d",

                    "depth_topic":
                        LaunchConfiguration("depth_topic"),

                    "camera_info_topic":
                        LaunchConfiguration("camera_info_topic"),
                },
            ],
        ),

        # --------------------------------------------------
        # 3. Camera -> odom
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_frame_transform",
            name="person_frame_transform",
            output="screen",
            parameters=[{
                "input_topic":
                    "/social_perception/detections_3d",

                "output_topic":
                    "/social_perception/detections_3d_odom",

                "target_frame": "odom",
                "tf_timestamp_policy": "detection",
            }],
        ),

        # --------------------------------------------------
        # 4. Upper / lower appearance extraction
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_appearance_extraction",
            name="person_appearance_extraction",
            output="screen",
            condition=use_multimodal_identity,
            parameters=[{
                "input_topic":
                    "/social_perception/detections_3d_odom",

                "output_topic":
                    "/social_perception/detections_enriched",

                "rgb_topic":
                    LaunchConfiguration("rgb_topic"),
            }],
        ),


        # --------------------------------------------------
        # 5. Multimodal identity tracking: production
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_tracking",
            name="person_tracking",
            condition=use_multimodal_identity,
            output="screen",
            parameters=[
                LaunchConfiguration("config_file"),
                {
                    "input_topic":
                        "/social_perception/detections_enriched",

                    "output_topic":
                        "/social_perception/detections_identified_odom",

                    "expected_frame": "odom",

                    "rgb_topic":
                        LaunchConfiguration("rgb_topic"),
                },
            ],
        ),

        # --------------------------------------------------
        # 5B. YOLO identity adapter (no multimodal scoring)
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_yolo_identity",
            name="person_yolo_identity",
            output="screen",
            condition=use_yolo_identity,
            parameters=[{
                "input_topic":
                    "/social_perception/detections_3d_odom",

                "output_topic":
                    "/social_perception/detections_identified_odom",

                "expected_frame": "odom",

                "track_retention_sec": 3.0,
            }],
        ),

        # --------------------------------------------------
        # 6. Spatial tracking
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="person_spatial_tracking",
            name="person_spatial_tracking",
            output="screen",
            parameters=[{
                "input_topic":
                    "/social_perception/detections_identified_odom",

                "odom_output_topic":
                    "/social_perception/tracked_persons_odom",

                "output_topic":
                    "/social_perception/tracked_persons",

                "expected_frame": "odom",
                "robot_frame": "base_footprint",

                "velocity_filter_alpha": 0.35,
                "minimum_velocity_dt_sec": 0.02,
                "maximum_velocity_dt_sec": 1.0,
                "maximum_speed_mps": 4.0,

                "minimum_tracking_observations": 3,
                "observation_freshness_timeout_sec": 1.0,
                "delete_timeout_sec": 5.0,
                "transform_timeout_sec": 0.05,
            }],
        ),

        # --------------------------------------------------
        # 7. Social zones
        # --------------------------------------------------

        Node(
            package="g1_social_perception",
            executable="social_zone_generator",
            name="social_zone_generator",
            output="screen",
            parameters=[
                LaunchConfiguration("config_file"),
            ],
        ),

        # --------------------------------------------------
        # 8. Optional identity benchmark
        # --------------------------------------------------

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                benchmark_launch
            ),
            condition=IfCondition(
                LaunchConfiguration("enable_benchmark")
            ),
            launch_arguments={
                "enable_ablation":
                    LaunchConfiguration(
                        "benchmark_ablation"
                    ),

                "enable_botsort":
                    LaunchConfiguration(
                        "benchmark_botsort"
                    ),

                "enable_benchmark_recorder":
                    LaunchConfiguration(
                        "benchmark_recorder"
                    ),

                "benchmark_input_topic":
                    "/social_perception/detections_enriched",

                "benchmark_rgb_topic":
                    LaunchConfiguration("rgb_topic"),
            }.items(),
        ),
    ])
