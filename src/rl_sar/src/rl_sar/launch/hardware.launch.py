from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
import os
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # ========================================================
    # DEPLOYMENT SWITCHES
    # ========================================================

    run_robot_core = LaunchConfiguration(
        "run_robot_core"
    )

    run_lidar_pipeline = LaunchConfiguration(
        "run_lidar_pipeline"
    )

    run_realsense_pipeline = LaunchConfiguration(
        "run_realsense_pipeline"
    )

    run_mola_lidar_odometry = LaunchConfiguration(
        "run_mola_lidar_odometry"
    )

    # ========================================================
    # ROBOT CORE ARGUMENTS
    # ========================================================

    network_interface = LaunchConfiguration(
        "network_interface"
    )

    cmd_vel_topic = LaunchConfiguration(
        "cmd_vel_topic"
    )

    cmd_vel_timeout_sec = LaunchConfiguration(
        "cmd_vel_timeout_sec"
    )

    hw_mode = LaunchConfiguration(
        "hw_mode"
    )

    g1_mode = LaunchConfiguration(
        "g1_mode"
    )

    joint_states_topic = LaunchConfiguration(
        "joint_states_topic"
    )

    odom_source = LaunchConfiguration(
        "odom_source"
    )

    mola_config_file = LaunchConfiguration(
        "mola_config_file"
    )

    enable_dog_odom_calibration = LaunchConfiguration(
        "enable_dog_odom_calibration"
    )

    imu_input_topic = LaunchConfiguration(
        "imu_input_topic"
    )

    imu_input_frame = LaunchConfiguration(
        "imu_input_frame"
    )

    imu_adapter_output_topic = LaunchConfiguration(
        "imu_adapter_output_topic"
    )

    # ========================================================
    # LIDAR ARGUMENTS
    # ========================================================

    lidar_source = LaunchConfiguration(
        "lidar_source"
    )
    
    enable_livox_driver = LaunchConfiguration(
        "enable_livox_driver"
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
    # REALSENSE ARGUMENTS
    # ========================================================

    enable_realsense = LaunchConfiguration(
        "enable_realsense"
    )

    enable_realsense_tf = LaunchConfiguration(
        "enable_realsense_tf"
    )

    # ========================================================
    # ROBOT CORE
    # ========================================================

    robot_core_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("rl_sar"),
                "launch",
                "robot_core.launch.py",
            ])
        ),

        condition=IfCondition(
            run_robot_core
        ),

        launch_arguments={
            "network_interface":
                network_interface,

            "cmd_vel_topic":
                cmd_vel_topic,

            "cmd_vel_timeout_sec":
                cmd_vel_timeout_sec,

            "hw_mode":
                hw_mode,

            "g1_mode":
                g1_mode,

            "joint_states_topic":
                joint_states_topic,

            "odom_source":
                odom_source,

            "enable_dog_odom_calibration":
                enable_dog_odom_calibration,

            "imu_input_topic":
                imu_input_topic,

            "imu_input_frame":
                imu_input_frame,

            "imu_adapter_output_topic":
                imu_adapter_output_topic,

            "common_time_offset_sec":
                common_time_offset_sec,

            "lidar_source":
                lidar_source,
        }.items(),
    )

    # ========================================================
    # LIDAR PIPELINE
    # ========================================================

    lidar_pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("rl_sar"),
                "launch",
                "lidar_pipeline.launch.py",
            ])
        ),

        condition=IfCondition(
            run_lidar_pipeline
        ),

        launch_arguments={
            "lidar_source":
                lidar_source,

            "enable_livox_driver":
                enable_livox_driver,

            "mola_config_file":
                mola_config_file,

            "common_time_offset_sec":
                common_time_offset_sec,

            "tf_time_offset_sec":
                tf_time_offset_sec,

            "yaw_correction_rad":
                yaw_correction_rad,

            "run_mola_lidar_odometry":
                run_mola_lidar_odometry,
            "scan_target_frame": 
                scan_target_frame,

            "scan_min_height": scan_min_height,
            "scan_max_height": scan_max_height,
        }.items(),
    )

    # ========================================================
    # REALSENSE PIPELINE
    # ========================================================

    realsense_pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("rl_sar"),
                "launch",
                "realsense_pipeline.launch.py",
            ])
        ),

        condition=IfCondition(
            run_realsense_pipeline
        ),

        launch_arguments={
            "enable_realsense":
                enable_realsense,

            "enable_realsense_tf":
                enable_realsense_tf,
        }.items(),
    )

    # ========================================================
    # PUBLIC HARDWARE LAUNCH INTERFACE
    # ========================================================

    return LaunchDescription([

        # ====================================================
        # DEPLOYMENT
        # ====================================================

        DeclareLaunchArgument(
            "run_robot_core",
            default_value="true",
            description=(
                "Run G1 low-level hardware, odometry, TF, "
                "IMU and command interfaces on this host."
            ),
        ),

        DeclareLaunchArgument(
            "run_lidar_pipeline",
            default_value="true",
            description=(
                "Run raw LiDAR processing and publish /scan "
                "on this host."
            ),
        ),

        DeclareLaunchArgument(
            "mola_config_file",
            default_value=os.path.expanduser(
                "~/g1_ws/mola_g1_ros2.yaml"
            ),
            description=(
                "MOLA configuration file used by the LiDAR pipeline."
            ),
        ),

        DeclareLaunchArgument(
            "run_realsense_pipeline",
            default_value="true",
            description=(
                "Run RealSense processing on this host."
            ),
        ),

        DeclareLaunchArgument(
            "run_mola_lidar_odometry",
            default_value="false",
            description=(
                "Run MOLA with the raw LiDAR pipeline."
            ),
        ),

        # ====================================================
        # ROBOT CORE
        # ====================================================

        DeclareLaunchArgument(
            "network_interface",
            default_value="eth0",
        ),

        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="/cmd_vel_rlsar",
        ),

        DeclareLaunchArgument(
            "cmd_vel_timeout_sec",
            default_value="0.35",
        ),

        DeclareLaunchArgument(
            "hw_mode",
            default_value="0",
        ),

        DeclareLaunchArgument(
            "g1_mode",
            default_value="mujoco",
        ),

        DeclareLaunchArgument(
            "joint_states_topic",
            default_value="/joint_states_source",
        ),

        DeclareLaunchArgument(
            "odom_source",
            default_value="dog",
            description=(
                "unitree | dog | dogc | dogekf | aocf | mola"
            ),
        ),

        DeclareLaunchArgument(
            "enable_dog_odom_calibration",
            default_value="true",
        ),

        # ====================================================
        # IMU
        # ====================================================

        DeclareLaunchArgument(
            "imu_input_topic",
            default_value="/livox/imu",
        ),

        DeclareLaunchArgument(
            "imu_input_frame",
            default_value="livox_frame",
        ),

        DeclareLaunchArgument(
            "imu_adapter_output_topic",
            default_value="/imu_ekf_input",
        ),

        # ====================================================
        # LIDAR
        # ====================================================

        DeclareLaunchArgument(
            "lidar_source",
            default_value="unitree",
            description="unitree | livox",
        ),

        DeclareLaunchArgument(
            "enable_livox_driver",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Start livox_ros_driver2 locally when "
                "lidar_source=livox. "
                "Set false when /livox/lidar and /livox/imu "
                "are already provided externally."
            ),
        ),

        DeclareLaunchArgument(
            "common_time_offset_sec",
            default_value="0.0",
        ),

        DeclareLaunchArgument(
            "tf_time_offset_sec",
            default_value="0.0",
        ),

        DeclareLaunchArgument(
            "yaw_correction_rad",
            default_value="0.4",
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
                """
                Minimum height in the selected scan target frame, in meters.
                
                -0.584	27 cm
                -0.704	15 cm
                -0.754	10 cm
                -0.804	5 cm
                """
            ),
        ),

        DeclareLaunchArgument(
            "scan_max_height",
            default_value="0.666",
            description=(
                "Maximum height in the selected scan target frame, in meters."
            ),
        ),

        # ====================================================
        # REALSENSE
        # ====================================================

        DeclareLaunchArgument(
            "enable_realsense",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "enable_realsense_tf",
            default_value="true",
        ),

        # ====================================================
        # LEGACY COMPATIBILITY
        #
        # Keep these arguments temporarily so existing shell
        # commands do not fail even though the old UDP bridge
        # is currently disabled.
        # ====================================================

        DeclareLaunchArgument(
            "bridge_imu",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "bridge_scan",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_odom",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_dog_odom",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "bridge_mola_odom",
            default_value="false",
        ),

        DeclareLaunchArgument(
            "bridge_tf",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_joint_states",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_map",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_goal",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_initialpose",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_global_costmap",
            default_value="true",
        ),

        DeclareLaunchArgument(
            "bridge_local_costmap",
            default_value="true",
        ),

        # ====================================================
        # CHILD LAUNCHES
        # ====================================================

        robot_core_launch,
        lidar_pipeline_launch,
        realsense_pipeline_launch,
    ])