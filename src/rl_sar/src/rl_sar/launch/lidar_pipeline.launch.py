from launch import LaunchDescription
import os

from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# ============================================================
# NAVIGATION TIME SYNCHRONIZATION
# ============================================================

def create_navigation_time_sync(context):

    selected_lidar = LaunchConfiguration(
        "lidar_source"
    ).perform(context)

    if selected_lidar == "livox":
        lidar_topic = "/livox/lidar"
        lidar_timestamp_mode = "header_stamp"

    elif selected_lidar == "unitree":
        lidar_topic = "/utlidar/cloud_livox_mid360"
        lidar_timestamp_mode = "point_time"

    else:
        raise RuntimeError(
            f"Invalid lidar_source '{selected_lidar}'. "
            "Supported values: unitree, livox"
        )

    print("=" * 70)
    print("LIDAR TIME SOURCE SELECTION")
    print("=" * 70)
    print(f"LiDAR source   : {selected_lidar}")
    print(f"LiDAR topic    : {lidar_topic}")
    print(f"Timestamp mode : {lidar_timestamp_mode}")
    print("=" * 70)

    navigation_time_sync_node = Node(
        package="rl_sar",
        executable="navigation_time_sync_node",
        name="navigation_time_sync_node",
        output="screen",

        # IMPORTANT:
        # Do not force ETH0_ENV here.
        # This launch must run unchanged on PC or robot.

        parameters=[{
            "lidar_topic": lidar_topic,
            "lidar_timestamp_mode": lidar_timestamp_mode,

            "dog_odom_topic": "/dog_odom",

            "lidar_to_nav_offset_topic":
                "/navigation_time/lidar_to_nav_offset",

            "dog_to_lidar_offset_topic":
                "/navigation_time/dog_to_lidar_offset",

            "dog_to_nav_offset_topic":
                "/navigation_time/dog_to_nav_offset",

            "sync_status_topic":
                "/navigation_time/sync_status",

            "warmup_samples": 100,
            "history_size": 500,
            "stable_mad_threshold_sec": 0.005,
            "drift_detection_threshold_sec": 0.01,
            "min_drift_samples": 100,
            "diagnostics_frequency_hz": 1.0,
        }],
    )

    return [navigation_time_sync_node]

def create_mola_lidar_odometry(context):
    selected_lidar = LaunchConfiguration(
        "lidar_source"
    ).perform(context)

    mola_config_file = LaunchConfiguration(
        "mola_config_file"
    ).perform(context)

    if selected_lidar == "unitree":
        mola_lidar_topic = "/utlidar/cloud_livox_mid360"
        mola_imu_topic = "/utlidar/imu_livox_mid360"

    elif selected_lidar == "livox":
        mola_lidar_topic = "/livox/lidar"
        mola_imu_topic = "/livox/imu"

    else:
        raise RuntimeError(
            f"Invalid lidar_source '{selected_lidar}'. "
            "Supported values: unitree, livox"
        )

    print("=" * 70)
    print("MOLA LIDAR ODOMETRY")
    print("=" * 70)
    print(f"LiDAR source : {selected_lidar}")
    print(f"MOLA LiDAR   : {mola_lidar_topic}")
    print(f"MOLA IMU     : {mola_imu_topic}")
    print(f"MOLA config  : {mola_config_file}")
    print("=" * 70)

    mola_env = {
        **os.environ,

        "MOLA_LIDAR_TOPIC":
            mola_lidar_topic,

        "MOLA_IMU_TOPIC":
            mola_imu_topic,

        "MOLA_USE_FIXED_LIDAR_POSE":
            "true",

        "MOLA_USE_FIXED_IMU_POSE":
            "true",

        "LIDAR_POSE_X":
            "0.08",

        "LIDAR_POSE_Y":
            "0.0",

        "LIDAR_POSE_Z":
            "1.20",

        "LIDAR_POSE_YAW":
            "0.0",

        "LIDAR_POSE_PITCH":
            "0.0",

        "LIDAR_POSE_ROLL":
            "0.0",

        "IMU_POSE_X":
            "0.08",

        "IMU_POSE_Y":
            "0.0",

        "IMU_POSE_Z":
            "1.20",

        "IMU_POSE_YAW":
            "0.0",

        "IMU_POSE_PITCH":
            "0.0",

        "IMU_POSE_ROLL":
            "0.0",

        # MOLA must never become the owner of the canonical
        # navigation TF in this configuration.
        "MOLA_LOCALIZATION_PUBLISH_TF":
            "false",

        "MOLA_TF_FOOTPRINT_LINK":
            "",
    }

    mola_process = ExecuteProcess(
        cmd=[
            "mola-cli",
            mola_config_file,
        ],

        additional_env=mola_env,

        output="screen",
    )

    return [mola_process]

# ============================================================
# MAIN
# ============================================================

def generate_launch_description():

    lidar_source = LaunchConfiguration("lidar_source")

    enable_livox_driver = LaunchConfiguration(
        "enable_livox_driver"
    )

    run_livox_driver = IfCondition(
        enable_livox_driver
    )

    common_time_offset_sec = LaunchConfiguration(
        "common_time_offset_sec"
    )

    tf_time_offset_sec = LaunchConfiguration(
        "tf_time_offset_sec"
    )

    yaw_correction_rad = LaunchConfiguration(
        "yaw_correction_rad"
    )

    run_mola_lidar_odometry = LaunchConfiguration(
        "run_mola_lidar_odometry"
    )

    mola_config_file = LaunchConfiguration(
        "mola_config_file"
    )

    # ========================================================
    # SOURCE CONDITIONS
    # ========================================================

    is_unitree_lidar = IfCondition(
        PythonExpression([
            "'", lidar_source, "' == 'unitree'"
        ])
    )

    is_livox_lidar = IfCondition(
        PythonExpression([
            "'", lidar_source, "' == 'livox'"
        ])
    )

    lidar_topic = PythonExpression([
        "'/utlidar/cloud_livox_mid360' "
        "if '", lidar_source, "' == 'unitree' "
        "else '/livox/lidar'"
    ])

    # ========================================================
    # LASERSCAN FRAME CONFIGURATION
    # ========================================================

    scan_target_frame = LaunchConfiguration(
        "scan_target_frame"
    )

    scan_min_height = LaunchConfiguration(
        "scan_min_height"
    )

    scan_max_height = LaunchConfiguration(
        "scan_max_height"
    )
    # ========================================================
    # UNITREE CLOUD STABILIZATION
    # ========================================================

    cloud_nav_stabilizer = Node(
        package="rl_sar",
        executable="cloud_nav_stabilizer_node",
        name="cloud_nav_stabilizer_node",
        output="screen",

        condition=is_unitree_lidar,

        parameters=[{
            "input_cloud_topic": lidar_topic,
            "output_cloud_topic": "/cloud_nav_deskewed",

            "source_frame": "livox_frame",
            "target_frame": "base_footprint",

            "point_time_mode": "auto",
            "point_time_field": "time",
            "point_time_unit": "nanoseconds",
            "time_scale": 1.0e-9,

            "deskew_bins": 40,

            "common_time_offset_sec":
                common_time_offset_sec,

            "time_origin_mode": "auto_offset",
            "offset_filter_alpha": 0.02,
            "cloud_time_offset_sec": 0.0,

            "tf_time_offset_sec":
                tf_time_offset_sec,

            "output_stamp_policy":
                "current_ros_time",
        }],
    )

    # ========================================================
    # UNITREE -> LASERSCAN
    # ========================================================

    cloud_to_scan_unitree = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_unitree",
        output="screen",

        condition=is_unitree_lidar,

        remappings=[
            (
                "cloud_in",
                "/cloud_nav_deskewed",
            ),
            (
                "scan",
                "/scan",
            ),
        ],

        parameters=[{
            "target_frame": "base_footprint",

            "transform_tolerance": 0.05,

            "min_height": -0.53,
            "max_height": 0.72,

            "angle_min": -3.141592653589793,
            "angle_max": 3.141592653589793,
            "angle_increment": 0.0087,

            "scan_time": 0.10,

            "range_min": 0.30,
            "range_max": 30.0,

            "use_inf": True,
            "inf_epsilon": 1.0,

            "queue_size": 5,
        }],
    )

    # ========================================================
    # LIVOX MID360 DRIVER
    #
    # Optional ownership of /livox/lidar and /livox/imu.
    #
    # lidar_source:=livox
    # enable_livox_driver:=true
    #     -> this launch starts livox_ros_driver2
    #
    # lidar_source:=livox
    # enable_livox_driver:=false
    #     -> this launch expects /livox/lidar to already exist
    # ========================================================

    livox_driver = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "livox_ros_driver2",
            "msg_MID360_launch.py",
        ],
        output="screen",
        condition=IfCondition(
            PythonExpression([
                "'", lidar_source, "' == 'livox' and '",
                enable_livox_driver, "' == 'true'"
            ])
        ),
    )

    # ========================================================
    # LIVOX TIME OFFSET
    # ========================================================

    lidar_time_offset_node = Node(
        package="rl_sar",
        executable="lidar_time_offset_node",
        name="lidar_time_offset_node",
        output="screen",

        condition=is_livox_lidar,

        parameters=[{
            "input_topic":
                "/livox/lidar",

            "output_topic":
                "/navigation_time/lidar_cloud",

            "offset_topic":
                "/navigation_time/lidar_to_nav_offset",
        }],
    )

    # ========================================================
    # LIVOX FRESHNESS GUARD
    # ========================================================

    cloud_freshness_guard = Node(
        package="rl_sar",
        executable="cloud_freshness_guard_node",
        name="cloud_freshness_guard_node",
        output="screen",

        condition=is_livox_lidar,

        parameters=[{
            "input_topic":
                "/navigation_time/lidar_cloud",

            "output_topic":
                "/navigation_time/lidar_cloud_fresh",

            "warning_age_sec": 0.50,
            "drop_age_sec": 1.00,
            "future_tolerance_sec": 0.10,
        }],
    )

    # ========================================================
    # LIVOX -> LASERSCAN
    # ========================================================

    cloud_to_scan_livox = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_livox",
        output="screen",

        condition=is_livox_lidar,

        remappings=[
            (
                "cloud_in",
                "/livox/lidar",
                #"/navigation_time/lidar_cloud_fresh",
            ),
            (
                "scan",
                "/scan",
            ),
        ],

        parameters=[{
            # ====================================================
            # FRAME CONFIGURATION
            # ====================================================

            "target_frame": scan_target_frame,

            "transform_tolerance": 0.05,

            # ====================================================
            # HEIGHT FILTER IN SELECTED TARGET FRAME
            # ====================================================

            "min_height": ParameterValue(
                scan_min_height,
                value_type=float,
            ),

            "max_height": ParameterValue(
                scan_max_height,
                value_type=float,
            ),

            # ====================================================
            # ANGULAR FILTER
            # ====================================================

            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0087,

            # ====================================================
            # SCAN CONFIGURATION
            # ====================================================

            "scan_time": 0.1,

            "range_min": 0.3,
            "range_max": 30.0,

            "use_inf": True,

            # No queue_size override:
            # use pointcloud_to_laserscan defaults.
        }],
    )

    # ========================================================
    # MOLA
    #
    # IMPORTANT:
    # The current MOLA launcher in hardware.launch.py is
    # robot/Foxy-specific. Do not silently move that shell command
    # to the Humble PC yet.
    #
    # For now this switch is declared but MOLA extraction should
    # be done separately after /scan deployment is validated.
    # ========================================================

    return LaunchDescription([

        # ====================================================
        # ARGUMENTS
        # ====================================================

        DeclareLaunchArgument(
            "lidar_source",
            default_value="unitree",
            description="unitree | livox",
        ),

        DeclareLaunchArgument(
            "common_time_offset_sec",
            default_value="0.0",
            description=(
                "Common LiDAR/navigation time offset."
            ),
        ),

        DeclareLaunchArgument(
            "tf_time_offset_sec",
            default_value="0.0",
            description=(
                "TF lookup time offset."
            ),
        ),

        DeclareLaunchArgument(
            "yaw_correction_rad",
            default_value="0.4",
            description=(
                "Yaw correction for cloud stabilization."
            ),
        ),

        DeclareLaunchArgument(
            "run_mola_lidar_odometry",
            default_value="false",
            description=(
                "Start MOLA LiDAR odometry on the same host "
                "as the raw LiDAR pipeline."
            ),
        ),

        DeclareLaunchArgument(
            "mola_config_file",
            default_value=os.path.expanduser(
                "~/g1_ws/mola_g1_ros2.yaml"
            ),
            description=(
                "MOLA configuration file used by mola-cli."
            ),
        ),

        DeclareLaunchArgument(
            "enable_livox_driver",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Start livox_ros_driver2 when lidar_source=livox. "
                "Set false when /livox/lidar is provided externally."
            ),
        ),

        DeclareLaunchArgument(
            "scan_target_frame",
            default_value="torso_link",
            choices=[
                "torso_link",
                "pelvis",
                "base_footprint",
            ],
            description=(
                "Target frame used for Livox pointcloud "
                "to LaserScan conversion."
            ),
        ),

        DeclareLaunchArgument(
            "scan_min_height",
            default_value="-0.754",
            description=(
                "Minimum height in the selected target frame, in meters."
            ),
        ),

        DeclareLaunchArgument(
            "scan_max_height",
            default_value="0.666",
            description=(
                "Maximum height in the selected target frame, in meters."
            ),
        ),


        livox_driver,

        # ====================================================
        # TIME SYNCHRONIZATION
        # ====================================================

        TimerAction(
            period=0.2,
            actions=[
                OpaqueFunction(
                    function=create_navigation_time_sync
                )
            ],
        ),

        # ====================================================
        # MOLA LIDAR ODOMETRY
        #
        # Runs on the same host as the raw LiDAR.
        # ====================================================

        TimerAction(
            period=0.4,
            condition=IfCondition(
                run_mola_lidar_odometry
            ),
            actions=[
                OpaqueFunction(
                    function=create_mola_lidar_odometry
                )
            ],
        ),

        # ====================================================
        # UNITREE BRANCH
        # ====================================================

        TimerAction(
            period=0.5,
            actions=[
                cloud_nav_stabilizer
            ],
        ),

        TimerAction(
            period=0.8,
            actions=[
                cloud_to_scan_unitree
            ],
        ),

        # ====================================================
        # LIVOX BRANCH
        # ====================================================

        TimerAction(
            period=0.5,
            actions=[
                lidar_time_offset_node
            ],
        ),

        TimerAction(
            period=0.7,
            actions=[
                cloud_freshness_guard
            ],
        ),

        TimerAction(
            period=0.9,
            actions=[
                cloud_to_scan_livox
            ],
        ),
    ])