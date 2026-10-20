"""
real_robot_bridge.launch.py

==============================================================
REAL ROBOT NAVIGATION BRIDGE (Production Version)
==============================================================

This launch file provides the minimal and clean bridge layer
between:

    • Real robot low-level drivers
    • SLAM Toolbox
    • Nav2 navigation stack

It performs 4 essential tasks:

1) /dog_odom  --->  /odom  + TF(odom -> base_frame)
   Standardizes odometry for Nav2 compatibility.

2) Static TF  base_frame -> lidar_frame
   Completes TF tree required by costmaps and SLAM.

3) PointCloud2 ---> LaserScan (/scan_raw)
   Converts 3D LiDAR to 2D scan for SLAM/Nav2.

4) Restamp /scan_raw ---> /scan
   Prevents TF message_filter timestamp errors
   caused by driver time inconsistencies.

--------------------------------------------------------------
Validated defaults (from real robot probes):

    /dog_odom.header.frame_id            = "odom"
    /dog_odom.child_frame_id             = "robot_center"
    /utlidar/cloud_livox_mid360.frame_id = "livox_frame"

So default TF chain becomes:

    map
      ↓
    odom
      ↓
    robot_center (base_footprint)
      ↓
    livox_frame

--------------------------------------------------------------
IMPORTANT:
This launch DOES NOT start debug/probe nodes.
It is strictly for production runtime.
--------------------------------------------------------------
"""

from launch.launch_description_sources import python_launch_description_source
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction,IncludeLaunchDescription, GroupAction, ExecuteProcess
from launch_ros.actions import SetRemap
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import yaml
import os

#For MOLA integration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

ETH0_ENV = {
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    "CYCLONEDDS_URI": "/home/unitree/cyclonedds_ws/cyclonedds.xml",
}

WIFI_ENV = {
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    "CYCLONEDDS_URI": "file:///home/unitree/cyclonedds_ws/cyclonedds_wifi.xml",
}

def generate_launch_description():

    # ==========================================================
    # 1️⃣ Clock configuration
    # ==========================================================
    # Real robot does NOT use /clock unless explicitly provided.
    # Must remain false to avoid TF time drift.
    use_sim_time = LaunchConfiguration("use_sim_time")


    # ==========================================================
    # 2️⃣ Input topics from hardware
    # ==========================================================
    # Raw odometry published by robot driver
    input_odom_topic = LaunchConfiguration("input_odom_topic")
    output_odom_topic = LaunchConfiguration("output_odom_topic") 

    # Raw 3D LiDAR cloud
    cloud_topic = LaunchConfiguration("cloud_topic")


    # ==========================================================
    # 3️⃣ TF Frame configuration
    # ==========================================================
    # These MUST match real robot frames
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    lidar_frame = LaunchConfiguration("lidar_frame")
    lidar_frame2d = LaunchConfiguration("lidar_frame2d")
    body_frame = LaunchConfiguration("body_frame")
    odommodestate_type = LaunchConfiguration("odommodestate_type")
    aocf_config = LaunchConfiguration("aocf_config")


    # ==========================================================
    # 4️⃣ Static transform parameters (extrinsics)
    # ==========================================================
    # Used to define LiDAR pose relative to base frame.
    # Default = zero (must be calibrated later).
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw = LaunchConfiguration("yaw")


    # ==========================================================
    # 5️⃣ Cloud -> Scan conversion parameters
    # ==========================================================
    min_height = LaunchConfiguration("min_height")
    max_height = LaunchConfiguration("max_height")
    range_min = LaunchConfiguration("range_min")
    range_max = LaunchConfiguration("range_max")
    transform_tolerance = LaunchConfiguration("transform_tolerance")


    pc2scan_params_file = LaunchConfiguration("pc2scan_params_file")
    #MOLA
    use_mola_odom = LaunchConfiguration("use_mola_odom")
    mola_lidar_topic = LaunchConfiguration("mola_lidar_topic")
    mola_imu_topic = LaunchConfiguration("mola_imu_topic")
    mola_robot_base = LaunchConfiguration("mola_robot_base")
    mola_base_link = LaunchConfiguration("mola_base_link")

    publish_fastlio_tf = LaunchConfiguration("publish_fastlio_tf")

    # -------------------------
    # Paths
    # -------------------------
    humanoid_nav_bridge = get_package_share_directory("humanoid_nav_bridge")
    odom_params__path = os.path.join(humanoid_nav_bridge, "params", "real_odom_tf_bridge_cmd_gated.yaml") 

    ekf_params_default = os.path.join(
        get_package_share_directory("humanoid_nav_bridge"),
        "params",
        "ekf_g1.yaml",
    )

    ekf_params_odoms = os.path.join(
        get_package_share_directory("humanoid_nav_bridge"),
        "params",
        "ekf_fastlio_dog.yaml",
    )

    dog_odom_nav_filter_params = os.path.join(
        humanoid_nav_bridge,
        "params",
        "dog_odom_nav_filter.yaml",
    )
    pc2scan_params_path = os.path.join(
        humanoid_nav_bridge, 
        "params", 
        "pc2scan.yaml"
    )

    scan_output_frame_id = LaunchConfiguration("scan_output_frame_id")
    scan_override_stamp = LaunchConfiguration("scan_override_stamp")
    scan_startup_drop_sec = LaunchConfiguration("scan_startup_drop_sec")

    dog_startup_tf_freeze_sec = LaunchConfiguration("dog_startup_tf_freeze_sec")
    dog_startup_tf_ramp_sec = LaunchConfiguration("dog_startup_tf_ramp_sec")

    fastlio_startup_tf_freeze_sec = LaunchConfiguration("fastlio_startup_tf_freeze_sec")
    fastlio_startup_tf_ramp_sec = LaunchConfiguration("fastlio_startup_tf_ramp_sec")

    start_scan_amcl_gate = LaunchConfiguration("start_scan_amcl_gate")
    slam_start_delay = LaunchConfiguration("slam_start_delay")

    g1_nav2_share = get_package_share_directory("g1_nav2")

    # -------------------------
    # G1 real robot low-level launch args
    # -------------------------
    network_interface = LaunchConfiguration("network_interface")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    cmd_vel_timeout_sec = LaunchConfiguration("cmd_vel_timeout_sec")
    hw_mode = LaunchConfiguration("hw_mode")
    g1_mode = LaunchConfiguration("g1_mode")

    # ==========================================================
    # (1) ODOM BRIDGE NODE
    # ==========================================================
    # Converts robot-native odometry into Nav2-compatible format.
    #
    # Responsibilities:
    #   - Subscribes to /dog_odom
    #   - Republishes standardized /odom
    #   - Publishes TF: odom -> base_frame
    #
    # Without this node:
    #   Nav2 cannot compute local costmap transforms.
    #
    odom_tf_bridge = Node(
        package="humanoid_nav_bridge",
        executable="odom_tf_bridge",  
        name="real_odom_tf_bridge",
        output="screen",

        parameters=[
            odom_params__path,
            {
                "use_sim_time": use_sim_time,
                "use_node_time": False,
                "input_odom_topic": input_odom_topic,
                "odom_frame": odom_frame,
                "base_frame": base_frame,        # base_footprint
                "body_frame": body_frame,        # robot_center
            }
        ]
    )


    # -------------------------
    # (1-bis) G1 real robot low-level bringup with TF
    # -------------------------
    g1_with_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                g1_nav2_share,
                "launch",
                "g1_with_tf.launch.py",
            )
        ),
        launch_arguments={
            "mode": g1_mode,
            "network_interface": network_interface,
            "cmd_vel_topic": cmd_vel_topic,
            "cmd_vel_timeout_sec": cmd_vel_timeout_sec,
            "hw_mode": hw_mode,
        }.items(),
    )
    # -------------------------
    # (A) ODOMMODESTATE -> /odommodestate/odom
    # -------------------------

    odommodestate_to_odom = Node(
        package="humanoid_nav_bridge",
        executable="odommodestate_to_odom",
        name="odommodestate_to_odom",
        output="screen",
        parameters=[{
            "input_topic": "/odommodestate",
            "in_type": odommodestate_type,       
            "output_topic": "/odommodestate/odom2d",
            "odom_frame": odom_frame,
            "child_frame": base_frame,
            "sub_reliability": "best_effort",
            "pub_reliability": "reliable",
            "sub_depth": 5,
            "pub_depth": 10,
        }],
    )

    static_tf_robot_center_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_robot_center_to_base",
        output="screen",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", body_frame,   # robot_center
            "--child-frame-id", "base",
        ],
    )

    static_tf_basefootprint_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_base_to_pelvis",
        output="screen",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", base_frame,
            "--child-frame-id", "base",
        ],
    )

    static_tf_livox_to_nav_laser = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_livox_to_nav_laser",
        output="screen",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "-0.292",   # correction -16.7°
            "--frame-id", "livox_frame",
            "--child-frame-id", "nav_laser_frame",
        ],
    )
    # -------------------------
    # (B) SMART ODOM BRIDGE -> /odom/raw + TF(base_footprint->robot_center)
    # -------------------------
    smart_bridge = Node(
        package="humanoid_nav_bridge",
        executable="smart_odom_bridge_cmd_gated",
        name="smart_odom_bridge",
        output="screen",
        parameters=[{
            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "body_frame": body_frame,

            "src1": "/odom/visual",
            "src2": "/odommodestate/odom2d",
            "src3": "/unitree/slam_relocation/odom",
            "src4": "/unitree/slam_mapping/odom",
            "src5": input_odom_topic,            # /dog_odom

            #"gate_cmd_vel_topic": "/cmd_vel_nav",
            "gate_cmd_vel_topic": "/cmd_vel_nav_scaled",
            "remap_yawrate_y_to_z": False, #True,

            "output_odom_topic": "/odom/raw",
            #"output_odom_topic": "/odom",

            #"publish_tf_body": True,             # TF base->robot_center (roll/pitch only)
            "publish_tf_body": False, 
            "publish_tf_main": False,
            #"publish_tf_main": True,

            "tf_rate_hz": 50.0,
            "odom_rate_hz": 50.0,

            "enable_filter": True,
            "deadband_vx": 0.02,
            "deadband_wz": 0.04,
            "ema_alpha": 0.20,


            # Gate behavior
            "cmd_eps_vx": 0.005,
            "cmd_eps_wz": 0.005,
            "cmd_timeout_sec": 1.0,
            "gate_release_hold_sec": 1.5,

            "meas_eps_vx": 0.04,
            "meas_eps_wz": 0.06,

            "turn_wz_thresh": 0.25,
            "turn_vx_small": 0.04,

            # Motion state hysteresis
            "motion_confirm_sec": 0.10,
            "idle_confirm_sec": 1.0,

            "zero_twist_when_idle": True,
            "freeze_yaw_when_idle": True,

            # -------------------------------------------------
            # Source trust policy
            # -------------------------------------------------
            # src1 = visual odometry:
            #   trusted for NAVIGATION pose
            #
            # src5 = dog_odom:
            #   NOT trusted for pose, because RL stabilization may inject drift

            "trust_src1_pose": True,    # /odom/visual
            "trust_src2_pose": False,    # /odommodestate/odom
            "trust_src3_pose": False,    # relocation
            "trust_src4_pose": False,    # mapping
            "trust_src5_pose": False,   # /dog_odom -> integrate, do NOT trust pose

            # -------------------------------------------------
            # Source selection behavior
            # -------------------------------------------------
            # Keep source fresh long enough to avoid overly aggressive fallback
            "source_timeout_sec": 0.8,

            # Lock source during motion to avoid VO -> dog_odom switching
            # in the middle of a trajectory
            "lock_source_while_moving": True,
            "locked_source_timeout_sec": 1.05,

            # -------------------------------------------------
            # Motion evidence thresholds
            # -------------------------------------------------
            # Use pose delta to detect movement even when twist is poor

            "pose_evidence_vxy": 0.01,
            "pose_evidence_wz": 0.03,

            "use_source_pose_for_trusted_sources": True,

            # Use src5 twist only as auxiliary locomotion evidence

            # Auxiliary src5 twist evidence
            "use_src5_twist_as_aux_evidence": True,

            # Trusted-pose fallback
            "enable_trusted_pose_fallback": True,
            "fallback_use_src5_twist": True,

        }],
    )

    # -------------------------
    # (C) ODOM DOG_ODOM_CALIBRATED -> /dog_odom + /dog_imu_raw + /cmd_vel_nav_scaled (Optional)
    # -------------------------
    dog_odom_nav_filter_node = Node(
        package="humanoid_nav_bridge",   
        executable="dog_odom_nav_filter",
        name="dog_odom_nav_filter",
        output="screen",
        parameters=[
            dog_odom_nav_filter_params
        ],
        #remappings=[("/odom", "/dog_odom_filtered")]
    )

    static_tf_pelvis_to_torso = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_pelvis_to_torso",
        output="screen",
        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "pelvis", "torso_link",
        ],
    )

    cmd_vel_guarded_artificial_odom = Node(
        package="humanoid_nav_bridge",
        executable="cmd_vel_guarded_artificial_odom",
        name="cmd_vel_guarded_artificial_odom",
        output="screen",
        parameters=[{
            # Frames / topic
            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "output_odom_topic": "/odom_artificial",

            # Inputs
            "cmd_topic": "/cmd_vel_nav_scaled",
            #"dog_odom_topic": input_odom_topic,   # /dog_odom
            "dog_odom_topic": "/repub/dog_odom",
            #"sub_reliable": False,
            #"pub_reliable": True,

            # Command validity
            "cmd_timeout_sec": 1.0,
            "gate_release_hold_sec": 0.3,

            # Command eps
            "cmd_eps_vx": 0.01,
            "cmd_eps_wz": 0.02,

            # Noise model (supervisor recommendation)
            #"linear_noise_ratio": 0.02,
            #"angular_noise_ratio": 0.08,
            "linear_noise_ratio": 0.0,
            "angular_noise_ratio": 0.0,
            "random_seed": 42,

            # dog_odom evidence
            "use_motion_evidence": True,
            "meas_eps_vx": 0.05,
            "meas_eps_wz": 0.08,
            "deadband_vx": 0.02,
            "deadband_wz": 0.04,
            "ema_alpha": 0.25,
            "remap_yawrate_y_to_z": False,

            # Turn-in-place logic
            "turn_wz_thresh": 0.18,
            "turn_vx_small": 0.05,

            # Moving / idle hysteresis
            "motion_confirm_sec": 0.10,
            "idle_confirm_sec": 0.40,

            # Publication
            "publish_tf": False,
            "publish_status": True,
            "odom_rate_hz": 50.0,
            "status_rate_hz": 5.0,

            "alpha_lin": 0.78,
            "alpha_ang": 0.50,

            # Coast stricter
            "coast_meas_vx_min": 0.12,
            "coast_meas_wz_min": 0.18,

            #"enable_adaptive_scale": True,
            #"linear_cmd_scale": 0.75,
            #"angular_cmd_scale": 0.65,

            "adapt_lr_lin": 0.01,
            "adapt_lr_ang": 0.01,

            "adapt_min_cmd_vx": 0.10,
            "adapt_min_cmd_wz": 0.15,

            "linear_cmd_scale_min": 0.45,
            "linear_cmd_scale_max": 1.00,
            "angular_cmd_scale_min": 0.40,
            "angular_cmd_scale_max": 1.00,

            # Freeze all adaptive scaling for clean validation
            "enable_adaptive_scale": False,
            "linear_cmd_scale": 0.80,
            "angular_cmd_scale": 0.65,

            # Contextual yaw fusion
            "enable_contextual_yaw_fusion": True,
            "alpha_yaw_base": 0.60,
            "yaw_low_speed_vx_thresh": 0.10,
            "alpha_yaw_low_speed_penalty": 0.25,
            "alpha_yaw_coast_penalty": 0.20,
            "yaw_mismatch_thresh": 0.08,
            "alpha_yaw_mismatch_penalty": 0.30,
            "alpha_yaw_min": 0.15,
            "alpha_yaw_max": 0.90,

        }],
    )

    mola_lidar_odom_launch = GroupAction([
    SetRemap(src="/tf", dst="/mola/tf"),
    #SetRemap(src="/tf_static", dst="/mola/tf_static"),

    IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("mola_lidar_odometry"),
                "ros2-launchs",
                "ros2-lidar-odometry.launch.py",
            ])
        ),
        condition=IfCondition(use_mola_odom),

        launch_arguments={
            "lidar_topic_name": mola_lidar_topic,
            "lidar_topic_type": "PointCloud2",

            # Minimal runtime: no UI, no RViz, no diagnostics aggregator
            "use_rviz": "False",
            "use_mola_gui": "False",
            "use_diagnostic_aggregator": "False",

            "odom_topic_name": "/mola_unused_odom_input",
            "forward_ros_tf_odom_to_mola": "False",

            # Real robot
            "use_sim_time": use_sim_time,

            # No GNSS / no saved maps for first tests
            "gnss_mode": "none",
            "generate_simplemap": "False",
            

            # Important for Nav2-style 2D evaluation
            "enforce_planar_motion": "True",
            #"enforce_planar_motion": "False",

            # Start simple: LiDAR odometry only
            "use_state_estimator": "False",
            "use_imu_for_lio": "False",
            #"imu_topic_name": mola_imu_topic,
            "imu_topic_name": "/mola_unused_imu",
            "imu_gravity_correction": "false",

            # Use separate MOLA frames to avoid conflict with Nav2 TF tree
            "mola_lo_reference_frame": "mola_map",
            "mola_bridge_odometry_frame": "mola_odom",
            "mola_state_estimator_reference_frame": "mola_map",

            "mola_lo_pipeline": "../pipelines/lidar3d-default.yaml",

            # Same base frame for pose interpretation
            # Use the real robot base frame because it is connected to livox_frame in TF
            "mola_tf_base_link": mola_robot_base,
            #"mola_footprint_to_base_link_tf": "[0, 0, 0, 0, 0, 0]",

            "start_active": "True",
            "start_mapping_enabled": "True",

            # Keep TF REP-105 behavior
            #"publish_localization_following_rep105": "True",
            "publish_localization_following_rep105": "False",
            
        }.items(),
    )
    ])

    mola_odom_2d_bridge = Node(
        package="humanoid_nav_bridge",
        executable="mola_odom_2d_bridge",
        name="mola_odom_2d_bridge",
        output="screen",
        parameters=[{
            "input_odom_topic": "/lidar_odometry/pose",
            "output_odom_topic": "/mola_nav_odom",
            "output_frame_id": "mola_odom_test",
            "output_child_frame_id": "base_footprint_mola_test",
            "publish_tf": True,
            "force_2d": True,
        }],
    )

    livox_to_mola_cloud_node = Node(
        package="humanoid_nav_bridge",
        executable="livox_to_mola_cloud",
        name="livox_to_mola_cloud",
        output="screen",
        parameters=[{
            "input_cloud_topic": "/utlidar/cloud_livox_mid360",
            "output_cloud_topic": "/mola/cloud_xyz_i",
            "output_frame_id": "livox_frame",
            "skip_n": 1,
            "max_points": 0,
        }],
    )

    livox_to_mola_cloud = Node(
        package="humanoid_nav_bridge",
        executable="livox_to_mola_cloud",
        name="livox_to_mola_cloud",
        output="screen",
        parameters=[{
            "input_cloud_topic": "/utlidar/cloud_livox_mid360",
            "output_cloud_topic": "/mola/cloud_xyz_i",
            "output_frame_id": "mola_livox_frame",
            "skip_n": 1,
            "max_points": 0,
        }],

    )

    # -------------------------
    # (C) EKF (robot_localization) -> TF(odom->base_footprint) + /odometry/filtered
    # -------------------------

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params_default, {"use_sim_time": use_sim_time}],
        remappings=[
            ("/odometry/filtered", "/odom"),
        ],
    )

    # ==========================================================
    # (2) STATIC TF: base_frame -> lidar_frame
    # ==========================================================
    # Required so that:
    #   TF tree is complete
    #   LaserScan can be transformed correctly
    #
    # Uses ROS2 built-in static_transform_publisher

    static_tf_base_to_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_base_to_lidar",
        output="screen",
        arguments=[
            "--x", x, "--y", y, "--z", z,
            "--roll", roll, "--pitch", pitch, "--yaw", yaw,
            "--frame-id", base_frame,
            "--child-frame-id", lidar_frame,
            ],
        )


    cloud_y_flip_node = Node(
        package="humanoid_nav_bridge",
        executable="cloud_y_flip",
        name="cloud_y_flip",
        output="screen",
        parameters=[{
            "input_cloud_topic": "/cloud_now",
            "output_cloud_topic": "/cloud_fixed",
        }],
    )
    cloud_y_flip = Node(
        package="rl_sar",
        executable="cloud_y_flip_node",
        name="cloud_y_flip_node",
        output="screen",
        additional_env=ETH0_ENV,
        parameters=[{
            "input_cloud_topic": "/cloud_fastlio_native",
            "output_cloud_topic": "/cloud_fastlio_fixed",
        }],
    )

    static_tf_base_to_livox2d = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_base_to_livox2d",
        output="screen",
        arguments=[
            "--x", x, "--y", y, "--z", z,
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", base_frame,          # base_footprint
            "--child-frame-id", lidar_frame2d, #"livox_2d"
        ],
    )

    static_tf_mola_base_to_livox = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_mola_base_to_livox",
        output="screen",
        arguments=[
            "--x", "0.08", "--y", "0.0", "--z", "1.20",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "mola_robot_base",
            "--child-frame-id", "mola_livox_frame",
        ],
    )
    # ==========================================================
    # (3) POINTCLOUD2 -> LASERSCAN
    # ==========================================================
    # Converts 3D cloud into 2D planar scan for:
    #   - slam_toolbox
    #   - Nav2 costmaps
    #
    # Output is /scan_raw (NOT /scan yet).
    #
    cloud_to_scan_fn = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_node",
        output="screen",
        additional_env=ETH0_ENV,
        remappings=[
            ("cloud_in", "/cloud_fastlio_native"),
            ("scan", "/scan_robot"),
        ],
        parameters=[{
            "target_frame": "nav_laser_fixed",
            #"target_frame": "odom",
            "transform_tolerance": 1.0,
            "min_height": -1.00,
            "max_height": 1.00,
            "range_min": 0.20,
            "range_max": 30.0,
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0087,
            "scan_time": 0.2,
            "use_inf": True,
        }],
    )

    system_logger_node = Node(
        package="humanoid_nav_bridge",
        executable="system_logger_node",
        name="system_logger_node",
        output="screen",
    )


    pc2_restamper = Node(
        package="humanoid_nav_bridge",
        executable="pc2_restamper",
        name="pc2_restamper",
        output="screen",
        parameters=[{
            "input_cloud_topic": cloud_topic,
            "output_cloud_topic": "/cloud_now",
        }],
    )
      # *********************** END SCAN ******************************

    # ==========================================================
    # (4) SCAN RESTAMPER
    # ==========================================================
    # Fixes timestamp inconsistencies between:
    #   - LiDAR driver
    #   - TF buffer
    #
    # Without this, you get:
    #   "Message Filter dropping message"
    #
    # Strategy:
    #   Override header.stamp with current node time.
    #
    scan_restamper = Node(
        package="humanoid_nav_bridge",
        executable="scan_restamper",
        name="scan_restamper",
        output="screen",
        parameters=[{
            "input_scan_topic": "/scan_raw",
            "output_scan_topic": "/scan",
            "override_stamp": True,  # ✅ Better True for navigation when 
            "pub_reliable": True,
            "qos_depth": 10,
        }],
    )

    fastlio_odom_3d_to_2d_node = Node(
        package="humanoid_nav_bridge",
        executable="fastlio_odom_3d_to_2d",
        name="fastlio_odom_3d_to_2d",
        output="screen",
        remappings=[
            ("/tf", "/fastlio/tf"),
            ("/tf_static", "/fastlio/tf_static"),
        ],
        parameters=[{
            "input_odom_topic": "/Odometry",

            # Nav2 odom topic
            "output_odom_topic": "/odom_fastlio_2d",

            # Nav2 standard frames
            "odom_frame": odom_frame,      # usually "odom"
            "base_frame": base_frame,      # usually "base_footprint"

            # IMPORTANT: only this node should publish odom -> base_footprint
            "publish_tf": publish_fastlio_tf,
        }]
    )

    smart_odom_selector = Node(
        package="humanoid_nav_bridge",
        executable="smart_odom_selector",
        name="smart_odom_selector",
        output="screen",
        additional_env=WIFI_ENV,
        parameters=[{
            "odom_frame": "odom",
            "base_frame": "base_footprint",

            "dog_topic": "/odom_dog_nav",
            "fastlio_topic": "/odom_fastlio_nav",
            "artificial_topic": "/odom_artificial",

            "output_odom_topic": "/odom",
            "max_jump_yaw": 0.50,
            #"fastlio_timeout": 0.25,
            "fastlio_timeout": 0.0,
            "artificial_timeout": 0.0,
            "dog_timeout": 1.0,
            "max_abs_xy": 50.0,

            "fastlio_recovery_samples": 10,
            "fastlio_min_hold_time": 0.8,
            "fastlio_recovery_time": 1.0,
        }],
    )

    smart_odom_fusion = Node(
        package="humanoid_nav_bridge",
        executable="smart_odom_fusion",
        name="smart_odom_fusion",
        output="screen",
        parameters=[{
            "base_odom_topic": "/dog_odom_filtered",
            "lidar_odom_topic": "/odom_fastlio_2d",
            "output_odom_topic": "/odom",

            "odom_frame": "odom",
            "base_frame": "base_footprint",
            "publish_tf": True,
            "tf_future_offset_sec": 0.05,

            "base_odom_reliable": False,
            "lidar_odom_reliable": False,
            "output_reliable": False,
            "qos_depth": 10,

            "base_timeout_sec": 0.7,
            "lidar_timeout_sec": 0.7,

            "offset_alpha_xy": 0.02,
            "offset_alpha_yaw": 0.02,

            "max_lidar_offset_xy": 2.0,
            "max_lidar_offset_yaw": 1.2,

            "max_lidar_step_m": 0.50,
            "max_lidar_yaw_step": 0.50,
        }],
    )

    ekf_dog_only_params = os.path.join(
        get_package_share_directory("humanoid_nav_bridge"),
        "params",
        "ekf_dog_only.yaml",
    )

    ekf_dog_fast_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params_odoms, {"use_sim_time": use_sim_time}],
        remappings=[
            ("/odometry/filtered", "/odom"),
        ],
    )
    aocf_configuration_yaml = PathJoinSubstitution([
        FindPackageShare("humanoid_nav_bridge"),
        "params",
        "aocf_experiments",
        "configurations",
        LaunchConfiguration("aocf_config"),
    ])
    aocf_common_yaml = PathJoinSubstitution([
        FindPackageShare("humanoid_nav_bridge"),
        "params",
        "aocf_experiments",
        "common",
        "aocf_common.yaml",
    ])

    dog_odom_normalizer_old = Node(
        package="humanoid_nav_bridge",
        executable="odom_normalizer",
        name="dog_odom_normalizer",
        output="screen",
        parameters=[
            aocf_common_yaml,
            aocf_configuration_yaml,
        ],
    )
    dog_odom_normalizer = Node(
        package="humanoid_nav_bridge",
        executable="odom_normalizer",
        name="dog_odom_normalizer",
        output="screen",
        parameters=[{
            # ==========================================================
            # ROS interfaces
            # ==========================================================
            #"input_odom_topic": "/repub/dog_odom",
            "input_odom_topic": "/dog_odom_2d",
            "output_odom_topic": "/odom_aocf",
            "cmd_vel_topic": "/cmd_vel_rlsar",

            # The output pose is expressed in F_O = odom.
            # The output twist is expressed in F_B = base_footprint.
            "odom_frame": "odom",
            "base_frame": "base_footprint",

            "publish_tf": False,

            # ==========================================================
            # Time, QoS and planar projection
            # ==========================================================
            "use_node_time": True,

            "sub_reliable": False,
            "cmd_reliable": True,
            "pub_reliable": True,
            "qos_depth": 10,

            "force_2d": True,
            "zero_z": True,
            "zero_roll_pitch": True,
            "remap_yawrate_y_to_z": False,

            # ==========================================================
            # Startup
            # ==========================================================
            "startup_freeze_sec": dog_startup_tf_freeze_sec,
            "startup_ramp_sec": dog_startup_tf_ramp_sec,

            # ==========================================================
            # Increment safety
            #
            # Translation norm is frame invariant, so this limiter may
            # operate on the increment expressed in F_B.
            # ==========================================================
            "max_step_m": 0.35,
            "max_step_yaw": 0.35,

            # ==========================================================
            # Optional increment attenuation
            #
            # Applied to corrected body-frame increments before their
            # conversion back to F_O.
            # ==========================================================
            "enable_lowpass": False,
            "position_alpha": 0.70,
            "yaw_alpha": 0.70,

            # ==========================================================
            # Output twist
            #
            # Pose is differentiated in F_O, then rotated into F_B.
            # ==========================================================
            "compute_twist_from_corrected_pose": True,

            "max_vx": 0.80,
            "max_vy": 0.40,
            "max_wz": 1.20,

            "filter_output_vx": True,
            "ema_vx_alpha": 0.20,
            "max_output_vx": 0.22,

            "filter_output_vy": True,
            "ema_vy_alpha": 0.15,
            "max_output_vy": 0.06,

            "filter_output_wz": True,
            "ema_wz_alpha": 0.25,
            "max_output_wz": 0.35,

            # ==========================================================
            # AMCE
            #
            # AMCE operates entirely in F_B:
            # measured body increment versus cmd_vel prediction.
            # ==========================================================
            "enable_amce": True,
            "amce_alpha": 0.02,

            "straight_cmd_vx_min": 0.08,
            "straight_cmd_wz_max": 0.03,
            "turn_cmd_wz_min": 0.08,

            "vx_gain_min": 0.35,
            "vx_gain_max": 1.20,

            "wz_gain_min": 0.30,
            "wz_gain_max": 1.30,

            "mci_x_min": 0.45,
            "mci_y_min": 0.10,
            "mci_yaw_min": 0.01,

            "mci_x_sensitivity": 8.0,
            "mci_y_sensitivity": 16.0,
            "mci_yaw_sensitivity": 25.0,

            # These parameters are currently declared in the node,
            # but they are not yet used by expected_motion/apply_amce.
            "max_expected_vx_margin": 0.06,
            "max_expected_vy": 0.025,
            "max_expected_wz_margin": 0.08,
            "max_expected_wz_straight": 0.008,

            # ==========================================================
            # Heading correction
            # ==========================================================
            "yaw_correction_mode": "raw_increment",

            "yaw_observation_gain": 0.02,
            "max_yaw_observation_correction": 0.001,

            "enable_straight_yaw_gate": True,
            "straight_yaw_gate_wz": 0.005,

            # ==========================================================
            # Stationary hold
            #
            # Operates on body-frame increments and a validated command.
            # ==========================================================
            "stationary_hold_enabled": True,
            "stationary_cmd_vx_max": 0.02,
            "stationary_cmd_wz_max": 0.02,
            "stationary_raw_step_max": 0.003,
            "stationary_raw_yaw_max": 0.003,
            "stationary_zero_twist": True,

            "cmd_timeout_sec": 0.35,
            "straight_yaw_gate_wz": 0.008,
            "straight_yaw_confirm_samples": 5,
            "straight_yaw_release_samples": 2
        }],
    )

    mola_odom_nav2_bridge = Node(
        package="humanoid_nav_bridge",
        executable="mola_odom_nav2_bridge",
        name="mola_odom_nav2_bridge",
        output="screen",
        parameters=[{
            "input_topic": "/odom_mola_2d",
            "output_topic": output_odom_topic,
            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "publish_tf": True,

            
            # Keep source clock correction available.
            "time_origin_mode": "auto_offset",
            "offset_filter_alpha": 0.02,
            "time_offset_sec": 0.0,
            "tf_past_offset_sec": 0.10,

            # Navigation output.
            "output_frequency_hz": 30.0,
            "max_state_age_sec": 0.5,
            "transform_time_offset_sec": 0.05,

        }],
    )
    
    # *************************SCAN***********************
    scan_relay_pc = Node(
        package="humanoid_nav_bridge",
        executable="scan_qos_relay",
        name="scan_qos_relay",
        output="screen",
        parameters=[{
            "input_scan_topic": "/repub/scan",
            "output_scan_topic": "/scan",
            "override_stamp": True,
            "sub_reliable": False,
            "pub_reliable": True,
            "qos_depth": 1,
        }],
    )

    scan_timing_monitor_eth = Node(
        package="humanoid_nav_bridge",
        executable="scan_timing_monitor",
        name="scan_timing_monitor_eth",
        output="screen",
        parameters=[{
            # ETH0-side topics
            "enable_cloud": True,
            "enable_scan_robot": True,
            "enable_scan": False,
            "enable_amcl": False,

            "cloud_topic": "/cloud_nav_deskewed",
            "scan_robot_topic": "/scan_robot",

            # No TF listener on the ETH0 monitor.
            "enable_tf_check": False,
            "enable_laser_tf_check": False,
            "enable_map_odom_tf_check": False,

            # Reduce processing load.
            "sample_every_n": 20,
            "report_period_sec": 5.0,
            "immediate_tf_sample_every_n": 5,

            # Avoid frequent logs caused by small variations.
            "age_change_threshold": 0.20,
            "hz_change_threshold": 1.00,
            "stale_timeout_sec": 3.0,
        }],
        additional_env=ETH0_ENV,
    )
  
    scan_relay = Node(
        package="humanoid_nav_bridge",
        executable="scan_qos_relay",
        name="scan_qos_relay",
        output="screen",

        # Navigation scan processing remains local to the robot.
        additional_env=ETH0_ENV,

        parameters=[{
            "input_scan_topic": "/scan_robot",
            "output_scan_topic": "/scan",
            "output_frame_id": scan_output_frame_id,

            "override_stamp": False,

            "sub_reliable": False,
            "pub_reliable": False,

            "startup_drop_sec": scan_startup_drop_sec,
            "qos_depth": 1,

            "min_finite_beams": 20,
            "max_input_age_sec": 0.15,

            "rviz_pub_hz": 2.0,

            "scan_jump_filter_enabled": True,
            "scan_jump_threshold": 0.6,
            "scan_jump_hold_frames": 5,
        }],
    )

    cloud_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_node",
        output="screen",
        additional_env=ETH0_ENV,
        remappings=[
            #("cloud_in", "/cloud_fixed"),
            ("cloud_in", "/cloud_nav_deskewed"),
            ("scan", "/scan_robot"),
        ],
        parameters=[{
            "target_frame": "nav_laser_fixed",
            "transform_tolerance": 1.0,

            #"min_height": -1.0,
            #"max_height": 1.0,
            "min_height": -0.30,
            "max_height": 0.40,

            "range_min": 0.45,
            "range_max": 5.0,

            # exactly 360 beams
            "angle_min": -3.14159265,
            #"angle_max": 3.12413936,
            "angle_increment": 0.01745329252,
            #"angle_max": 3.10668607,   # -pi + 358 * 1deg
            "angle_max": 3.14159265, 

            "scan_time": 0.2,
            "use_inf": True,
            "inf_epsilon": 1.0,
            "queue_size": 1,
        }],
    )

    odom_publisher = Node(
        package="humanoid_nav_bridge",
        executable="g1_odom_publisher",
        name="g1_odom_publisher",
        output="screen",

        additional_env=WIFI_ENV,

        arguments=[
            "eth0",
        ],

        parameters=[{
            "odom_topic": "/odom",
            "odom_frame": "odom",
            "base_frame": "base_footprint",

            "publish_rate": 50.0,
            "publish_tf": True,

            "zero_local_origin": True,

            "max_abs_position": 1000.0,
            "min_quaternion_norm": 0.5,
            "max_quaternion_norm": 1.5,
        }],
    )

    # ==========================================================
    # Launch arguments declaration
    # ==========================================================
    declared_args = [

        # Clock
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Real robot: keep false (no /clock topic).",
        ),
        DeclareLaunchArgument(
            "aocf_config",
            default_value="FULL.yaml",
            description="AOCF experimental configuration",
        ),

        DeclareLaunchArgument(
            "publish_fastlio_tf",
            default_value="false",
            description="Publish TF odom -> base_footprint from FAST-LIO odometry"
        ),

        # Input topics
        DeclareLaunchArgument("input_odom_topic", default_value="/dog_odom_2d"),
        #DeclareLaunchArgument("input_odom_topic", default_value="/repub/dog_odom"),
        DeclareLaunchArgument("output_odom_topic", default_value="/odom", description="Output topic for odom."), 
        #DeclareLaunchArgument("cloud_topic", default_value="/utlidar/cloud_livox_mid360"),
        DeclareLaunchArgument("cloud_topic", default_value="/repub/mola/cloud_xyz_i"),

        # Frames
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument("base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("lidar_frame", default_value="livox_frame"),
        DeclareLaunchArgument("lidar_frame2d", default_value="nav_laser_frame"),
        DeclareLaunchArgument("mola_robot_base", default_value="mola_robot_base"),
        DeclareLaunchArgument("frame", default_value="base"),
        DeclareLaunchArgument("body_frame", default_value="robot_center"),

        # Static TF initial guess  
        #DeclareLaunchArgument("x", default_value="0.10"),
        #DeclareLaunchArgument("y", default_value="0.00"),
        #DeclareLaunchArgument("z", default_value="0.95"),

        #DeclareLaunchArgument("roll", default_value="0.0"),
        #DeclareLaunchArgument("pitch", default_value="-0.436332"),  # -25 deg down
        #DeclareLaunchArgument("yaw", default_value="0.0"),


        DeclareLaunchArgument("x", default_value="0.08"),
        DeclareLaunchArgument("y", default_value="0.00"),
        DeclareLaunchArgument("z", default_value="1.20"),

        DeclareLaunchArgument("roll", default_value="0.0"),
        DeclareLaunchArgument("pitch", default_value="-0.20"),
        DeclareLaunchArgument("yaw", default_value="0.0"),


        # Cloud->scan tuning
        DeclareLaunchArgument("range_min", default_value="0.2"),
        DeclareLaunchArgument("range_max", default_value="30.0"),
        DeclareLaunchArgument("transform_tolerance", default_value="1.0"),
        DeclareLaunchArgument("min_height", default_value="-10.0"),
        DeclareLaunchArgument("max_height", default_value="10.0"),

        DeclareLaunchArgument(
            "pc2scan_params_file",
            default_value=pc2scan_params_path,
            description="Optional YAML params file for pointcloud_to_laserscan",
        ),

        DeclareLaunchArgument(
            "odommodestate_type",
            default_value="unitree_go/msg/SportModeState",
            description="Exact ROS 2 type for /odommodestate, e.g. unitree_go/msg/OdomModeState (from: ros2 topic info /odommodestate -v)",
        ),

        DeclareLaunchArgument(
            "ekf_params_file",
            default_value=os.path.join(humanoid_nav_bridge, "params", "ekf_g1.yaml"),
            description="robot_localization EKF params file",
        ),

        DeclareLaunchArgument(
            "g1_mode",
            default_value="real",
            description="Mode for g1_with_tf.launch.py",
        ),
        DeclareLaunchArgument(
            "network_interface",
            default_value="eno1",
            description="Network interface used by the real G1 robot",
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="/cmd_vel_rlsar",
            description="Velocity topic consumed by g1_with_tf.launch.py",
        ),
        DeclareLaunchArgument(
            "cmd_vel_timeout_sec",
            default_value="2.0",
            description="Command timeout for low-level G1 interface",
        ),
        DeclareLaunchArgument(
            "hw_mode",
            default_value="0",
            description="Hardware mode passed to g1_with_tf.launch.py",
        ),
        DeclareLaunchArgument(
            "use_mola_odom",
            default_value="true",
            description="Enable optional MOLA 3D LiDAR odometry.",
        ),

        DeclareLaunchArgument(
            "mola_lidar_topic",
            default_value="/repub/mola/cloud_xyz_i",
            description="PointCloud2 topic used by MOLA LiDAR odometry.",
        ),

        DeclareLaunchArgument(
            "mola_imu_topic",
            default_value="/dog_imu_raw",
            description="Optional IMU topic for future MOLA LIO tests.",
        ),  

        DeclareLaunchArgument(
            "scan_output_frame_id",
            default_value="nav_laser_fixed",
            description="Frame ID used in /scan header."
        ),

        DeclareLaunchArgument(
            "scan_override_stamp",
            default_value="false",
            description="If true, replace /scan timestamp with node time."
        ),

        DeclareLaunchArgument(
            "scan_startup_drop_sec",
            default_value="5.0",
            description="Drop initial scans during TF startup."
        ),

        DeclareLaunchArgument(
            "dog_startup_tf_freeze_sec",
            default_value="2.0",
            description="Freeze dog odom output during startup."
        ),

        DeclareLaunchArgument(
            "dog_startup_tf_ramp_sec",
            default_value="1.0",
            description="Ramp dog odom TF after startup freeze."
        ),

        DeclareLaunchArgument(
            "fastlio_startup_tf_freeze_sec",
            default_value="22.0",
            description="Freeze FAST-LIO odom during startup."
        ),

        DeclareLaunchArgument(
            "fastlio_startup_tf_ramp_sec",
            default_value="10.0",
            description="Ramp FAST-LIO odom after startup freeze."
        ),

        DeclareLaunchArgument(
            "start_scan_amcl_gate",
            default_value="true",
            description="Enable /scan -> /scan_amcl gate. For SLAM, usually false."
        ),

        DeclareLaunchArgument(
            "slam_start_delay",
            default_value="25.0",
            description="Delay before starting SLAM Toolbox."
        ),  
        
    ]


    # ==========================================================
    # STARTUP ORDER (critical for stability)
    # ==========================================================
    #
    # 0.5s  → odom bridge (TF must exist first)
    # 1.0s  → static TF
    # 2.0s  → cloud -> scan conversion
    # 2.5s  → restamp (after scan_raw exists)
    #

    actions = [

        TimerAction(period=0.2, actions=[system_logger_node]),
        # -------------------------------------------------
        # 1️⃣ ODOM SOURCES / AUXILIARY SOURCES
        # -------------------------------------------------
        
        #TimerAction(period=0.5, actions=[odommodestate_to_odom]),   # A

        # -------------------------------------------------
        # 2️⃣ DEBUG BRIDGE (VO / source analysis only)
        # -------------------------------------------------
        
        #TimerAction(period=0.5, actions=[smart_bridge]),            # B

        # -------------------------------------------------
        # 3️⃣ MAIN NAVIGATION ODOM FROM CMD_VEL
        # -------------------------------------------------
        #TimerAction(period=0.7, actions=[fastlio_odom_3d_to_2d_node]),
        #TimerAction(period=1.1, actions=[cmd_vel_guarded_artificial_odom]),

        # -------------------------------------------------
        # 4️⃣  ROBOT TF TREE (URDF + joint states)
        # -------------------------------------------------
        
        #TimerAction(period=1.2, actions=[static_tf_pelvis_to_torso]),
        #TimerAction(period=1.4, actions=[g1_with_tf_launch]),

        # -------------------------------------------------
        # 5️⃣ ODOM DOG_ODOM_CALIBRATED
        # -------------------------------------------------
        
        #TimerAction(period=1.5, actions=[mola_odom_nav2_bridge]),
        
        #TimerAction(period=1.9, actions=[dog_odom_normalizer]),
        #TimerAction(period=2.2, actions=[fastlio_odom_normalizer]),
        #TimerAction(period=2.5, actions=[smart_odom_selector]),
        #TimerAction(period=2.1, actions=[smart_odom_fusion]),

        #TimerAction(period=2.5, actions=[ekf_dog_fast_node]),
        

        # -------------------------------------------------
        # 5️⃣ CONNECT NAVIGATION BASE TO ROBOT MODEL (Already published in the urdf)
        # -------------------------------------------------
        #TimerAction(period=1.7, actions=[static_tf_basefootprint_to_base]),
        #TimerAction(period=1.7, actions=[static_tf_livox_to_nav_laser]),
        #TimerAction(period=1.7, actions=[odom_publisher]),
        
        
       # TimerAction(period=3.0, actions=[cloud_to_scan]),
        #TimerAction(period=3.9, actions=[scan_relay]),

        #TimerAction(period=4.2, actions=[cloud_to_scan_raw]),
        #TimerAction(period=4.2, actions=[scan_timing_monitor_eth]),
        #condition=IfCondition(start_scan_amcl_gate),
        #),

        # -------------------------------------------------
        # 6️⃣  SENSOR TF (Already published in the urdf)
        # -------------------------------------------------
        #TimerAction(period=1.0, actions=[static_tf_base_to_lidar]),
        #TimerAction(period=1.5, actions=[static_tf_base_to_livox2d]),

        # -------------------------------------------------
        # 7️⃣ CLOUD PREPROCESSING
        # -------------------------------------------------
        
        #TimerAction(period=1.8, actions=[cloud_y_flip_node]),

        # -------------------------------------------------
        # 8️⃣ LIDAR PIPELINE
        # -------------------------------------------------
        
        #TimerAction(period=2.4, actions=[pc2_restamper]),
        
        #TimerAction(period=3.1, actions=[scan_restamper]),
        
        # ------------------------------------------------
        #  ODOM MOLA
        #-------------------------------------------------
        #TimerAction(period=3.2, actions=[static_tf_mola_base_to_livox]),
        #TimerAction(period=6.0, actions=[livox_to_mola_cloud]),
        #TimerAction(period=7.5, actions=[mola_lidar_odom_launch]),
        #TimerAction(period=8.5, actions=[mola_odom_2d_bridge]),
        

    ]
    return LaunchDescription(declared_args + actions)

