import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


# ============================================================
# ROBOT-SIDE DDS ENVIRONMENT
# ============================================================

ETH0_ENV = {
    **os.environ,
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    #"CYCLONEDDS_URI": "/home/unitree/cyclonedds_ws/cyclonedds.xml",
    "CYCLONEDDS_URI":
        "/home/unitree/cyclonedds_ws/cyclonedds_g1_optimized.xml",

    
}


# ============================================================
# DOG EKF PARAMETERS
# ============================================================

EKF_PARAMETERS_PATH = os.path.join(
    get_package_share_directory("g1_nav2"),
    "params",
    "dog_odom_ekf.yaml",
)


# ============================================================
# DOG EKF FACTORY
# ============================================================

def create_dog_ekf(context):

    imu_adapter_output_topic = LaunchConfiguration(
        "imu_adapter_output_topic"
    ).perform(context)

    print("=" * 70)
    print("DOG ODOMETRY EKF")
    print("=" * 70)
    print(f"Prepared IMU topic : {imu_adapter_output_topic}")
    print("Odometry input     : /odom_dog_calibrated")
    print("EKF output         : /odom_dog_ekf")
    print("EKF publishes TF   : no")
    print("=" * 70)

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="dog_odom_ekf",
        output="screen",
        additional_env=ETH0_ENV,

        parameters=[
            EKF_PARAMETERS_PATH,
            {
                "imu0": imu_adapter_output_topic,
                "publish_tf": False,
            },
        ],

        remappings=[
            (
                "odometry/filtered",
                "/odom_dog_ekf",
            ),
        ],
    )

    return [ekf_node]


# ============================================================
# CANONICAL ODOM SELECTOR FACTORY
# ============================================================

def create_odom_selector(context):

    selected_source = LaunchConfiguration(
        "odom_source"
    ).perform(context)

    source_topics = {
        "unitree": "/odom_unitree",
        "dog": "/dog_odom_2d",
        "dogc": "/odom_dog_calibrated",
        "dogekf": "/odom_dog_ekf",
        "aocf": "/odom_aocf",
        "mola": "/odom_mola_nav",
    }

    if selected_source not in source_topics:
        valid_sources = ", ".join(source_topics.keys())

        raise RuntimeError(
            f"Invalid odom_source '{selected_source}'. "
            f"Supported values: {valid_sources}"
        )

    selected_topic = source_topics[selected_source]

    print("=" * 70)
    print("CANONICAL ODOMETRY SELECTION")
    print("=" * 70)
    print(f"Selected source : {selected_source}")
    print(f"Input topic     : {selected_topic}")
    print("Output topic    : /odom")
    print("Published TF    : odom -> base_footprint")
    print("=" * 70)

    selector_node = Node(
        package="rl_sar",
        executable="odom_source_selector",
        name="odom_source_selector",
        output="screen",
        additional_env=ETH0_ENV,

        parameters=[{
            "input_topic": selected_topic,
            "output_topic": "/odom",

            "odom_frame": "odom",
            "base_frame": "base_footprint",

            "publish_tf": True,
            "preserve_input_stamp": True,
        }],
    )

    return [selector_node]


# ============================================================
# MAIN
# ============================================================

def generate_launch_description():

    # --------------------------------------------------------
    # Launch arguments
    # --------------------------------------------------------

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

    common_time_offset_sec = LaunchConfiguration(
        "common_time_offset_sec"
    )

    lidar_source = LaunchConfiguration(
        "lidar_source"
    )

    # ========================================================
    # LOW STATE -> JOINT STATES
    # ========================================================

    lowstate_to_joint_states = Node(
        package="rl_sar",
        executable="lowstate_to_joint_states_node",
        name="lowstate_to_joint_states_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "lowstate_topic": "/lowstate",
            "output_topic": "/joint_states_source",

            "base_yaml_path":
                "/home/unitree/g1_ws/src/rl_sar/"
                "policy/g1/base_edu23.yaml",

            "policy_yaml_path":
                "/home/unitree/g1_ws/src/rl_sar/"
                "policy/g1/robomimic/locomotion/config.yaml",

            "joint_pub_rate_hz": 20.0,
        }],
    )

    # ========================================================
    # ROBOT TF / LOW-LEVEL G1
    # ========================================================

    g1_with_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("g1_nav2"),
                "launch",
                "g1_with_tf.launch.py",
            ])
        ),

        launch_arguments={
            "joint_states_topic":
                joint_states_topic,

            "mode":
                g1_mode,

            "network_interface":
                network_interface,

            "cmd_vel_topic":
                cmd_vel_topic,

            "cmd_vel_timeout_sec":
                cmd_vel_timeout_sec,

            "hw_mode":
                hw_mode,

            "publish_full_body_tf":
                "true",
        }.items(),
    )

    g1_with_tf_group = GroupAction([
        SetEnvironmentVariable(
            "ROS_DOMAIN_ID",
            "0",
        ),

        SetEnvironmentVariable(
            "RMW_IMPLEMENTATION",
            "rmw_cyclonedds_cpp",
        ),

        SetEnvironmentVariable(
            "ROS_LOCALHOST_ONLY",
            "0",
        ),

        SetEnvironmentVariable(
            "CYCLONEDDS_URI",
            "/home/unitree/cyclonedds_ws/cyclonedds.xml",
        ),

        g1_with_tf_launch,
    ])

    # ========================================================
    # DOG RAW ODOM ADAPTER
    # ========================================================

    dog_odom_adapter_node = Node(
        package="rl_sar",
        executable="dog_odom_adapter_node",
        name="dog_odom_adapter_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "input_topic": "/dog_odom",
            "output_topic": "/dog_odom_2d",

            "output_frame_id": "odom",
            "output_child_frame_id": "pelvis",

            "preserve_source_twist": True,
        }],
    )

    # ========================================================
    # DOG ODOM CALIBRATION
    # ========================================================

    dog_odom_calibration_node = Node(
        package="rl_sar",
        executable="dog_odom_calibration_node",
        name="dog_odom_calibration_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "input_topic": "/dog_odom_2d",
            "output_topic": "/odom_dog_calibrated",

            "odom_frame": "odom",
            "base_frame": "base_footprint",

            "enable_calibration": ParameterValue(
                enable_dog_odom_calibration,
                value_type=bool,
            ),

            "enable_longitudinal_scaling": True,
            "longitudinal_scale":
                1.1649814791564732,

            "enable_lateral_scaling": False,
            "lateral_scale": 1.0,

            "enable_yaw_scaling": False,
            "yaw_scale_ccw": 1.0,
            "yaw_scale_cw": 1.0,

            "scale_twist": False,

            "command_topic": "/cmd_vel_rlsar",
            "command_timeout_s": 0.30,

            "maximum_translation_increment_m": 0.20,
            "maximum_yaw_increment_rad": 0.30,

            "maximum_linear_velocity_for_rotation": 0.02,
            "minimum_angular_velocity_for_rotation": 0.03,
        }],
    )

    # ========================================================
    # DOG AOCF
    # ========================================================

    dog_odom_aocf_node = Node(
        package="rl_sar",
        executable="dog_odom_aocf_node",
        name="dog_odom_aocf_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "input_odom_topic": "/dog_odom_2d",
            "output_odom_topic": "/odom_aocf",

            "cmd_vel_topic": "/cmd_vel_rlsar",

            "odom_frame": "odom",
            "base_frame": "pelvis",

            "enable_mpm": True,
            "enable_amce": True,
            "enable_afi": False,
            "enable_ahc": True,
            "enable_atc": True,

            "enable_increment_limiter": True,
            "stationary_hold_enabled": True,

            "output_frequency_hz": 45.0,
            "max_state_age_sec": 0.5,

            "startup_freeze_sec": 2.0,
            "startup_ramp_sec": 1.0,

            "odom_period_alpha": 0.02,
            "odom_gap_factor": 5.0,
            "odom_period_init_samples": 20,

            "max_step_m": 0.12,
            "max_step_yaw": 0.12,

            "straight_cmd_vx_min": 0.08,
            "straight_cmd_wz_max": 0.08,
            "turn_cmd_wz_min": 0.08,

            "vx_gain_min": 0.35,
            "vx_gain_max": 1.20,

            "wz_gain_min": 0.30,
            "wz_gain_max": 1.30,

            "amce_alpha": 0.02,

            "mci_x_min": 0.60,
            "mci_y_min": 0.10,
            "mci_yaw_min": 0.05,

            "mci_x_sensitivity": 8.0,
            "mci_y_sensitivity": 16.0,
            "mci_yaw_sensitivity": 10.0,

            "straight_lateral_hold_enabled": True,
            "straight_lateral_velocity_max": 0.04,

            "straight_yaw_gate_wz": 0.008,

            "stationary_cmd_vx_max": 0.02,
            "stationary_cmd_wz_max": 0.02,

            "stationary_raw_step_max": 0.003,
            "stationary_raw_yaw_max": 0.003,

            "cmd_timeout_sec": 0.35,

            "filter_output_vx": True,
            "filter_output_vy": True,
            "filter_output_wz": True,

            "ema_vx_alpha": 0.20,
            "ema_vy_alpha": 0.15,
            "ema_wz_alpha": 0.25,

            "max_output_vx": 0.25,
            "max_output_vy": 0.08,
            "max_output_wz": 0.35,
        }],
    )

    # ========================================================
    # UNITREE ODOM UDP RECEIVER
    # ========================================================

    odom_unitree_sender = Node(
        package="rl_sar",
        executable="g1_odom_unitree_sender",
        name="g1_odom_unitree_sender",
        output="screen",
        additional_env=ETH0_ENV,

        arguments=[
            network_interface,
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    odom_publisher = Node(
        package="rl_sar",
        executable="g1_odom_publisher",
        name="g1_odom_publisher",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "udp_port": 17001,

            "output_topic": "/odom_unitree",
            "publish_tf": False,

            "odom_frame": "odom",
            "base_frame": "pelvis",

            "zero_local_origin": True,
        }],
    )

    # ========================================================
    # IMU ADAPTER
    # ========================================================

    lidar_imu_adapter_node = Node(
        package="rl_sar",
        executable="livox_imu_adapter_node",
        name="imu_adapter_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "input_topic":
                imu_input_topic,

            "output_topic":
                imu_adapter_output_topic,

            "expected_frame_id":
                imu_input_frame,

            "reject_unexpected_frame":
                True,

            "angular_velocity_variance_x":
                0.01,

            "angular_velocity_variance_y":
                0.01,

            "angular_velocity_variance_z":
                0.01,
            
        }],
    )

    # ========================================================
    # MOLA ODOM ADAPTER
    #
    # Light consumer only. MOLA itself belongs to lidar pipeline.
    # ========================================================

    mola_odom_adapter_node = Node(
        package="rl_sar",
        executable="mola_odom_adapter_node",
        name="mola_odom_adapter_node",
        output="screen",
        additional_env=ETH0_ENV,
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "input_topic":
                "/lidar_odometry/pose",

            "output_topic":
                "/odom_mola_2d",

            "nav_output_topic":
                "/odom_mola_nav",

            "output_frame_id":
                "odom",

            "output_child_frame_id":
                "pelvis",

            "publish_nav_tf":
                False,

            "nav_publish_frequency":
                30.0,

            "max_state_age_sec":
                0.5,

            "tf_future_offset_sec":
                0.03,

            "max_tf_prediction_sec":
                0.25,

            "enable_twist_filter":
                True,

            "twist_filter_alpha":
                0.3,

            "stationary_linear_threshold":
                0.025,

            "stationary_angular_threshold":
                0.01,

            "common_time_offset_sec":
                common_time_offset_sec,

            "apply_lidar_time_offset":
                ParameterValue(
                    PythonExpression([
                        "'",
                        lidar_source,
                        "' == 'unitree'"
                    ]),
                    value_type=bool,
                ),

            "max_dt":
                0.5,

            "max_linear_velocity":
                2.0,

            "max_angular_velocity":
                2.0,
        }],
    )

    # ========================================================
    # COMMAND BRIDGE
    # ========================================================

    nav_cmd_bridge = Node(
        package="rl_sar",
        executable="ros2_nav_bridge_node",
        name="ros2_nav_bridge_node",
        output="screen",
        additional_env=ETH0_ENV,

        parameters=[{
            "cmd_vel_topic":
                "/cmd_vel_rlsar",

            "udp_host":
                "127.0.0.1",

            "udp_port":
                15000,
        }],
    )

    # ========================================================
    # LAUNCH DESCRIPTION
    # ========================================================

    return LaunchDescription([

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

        DeclareLaunchArgument(
            "common_time_offset_sec",
            default_value="0.0",
        ),

        DeclareLaunchArgument(
            "lidar_source",
            default_value="unitree",
            description="unitree | livox",
        ),

        # ----------------------------------------------------
        # Startup
        # ----------------------------------------------------

        TimerAction(
            period=0.1,
            actions=[lowstate_to_joint_states],
        ),

        TimerAction(
            period=0.9,
            actions=[g1_with_tf_group],
        ),

        TimerAction(
            period=2.3,
            actions=[dog_odom_adapter_node],
        ),

        TimerAction(
            period=2.4,
            actions=[odom_unitree_sender],
        ),

        TimerAction(
            period=4.0,
            actions=[dog_odom_calibration_node],
        ),

        #TimerAction(period=4.3, actions=[dog_odom_aocf_node],),

        TimerAction(
            period=4.5,
            actions=[lidar_imu_adapter_node],
        ),

        TimerAction(
            period=4.6,
            actions=[odom_publisher],
        ),

        #TimerAction(period=4.8, actions=[OpaqueFunction(function=create_dog_ekf)],),

        #TimerAction(period=4.9, actions=[mola_odom_adapter_node],),

        TimerAction(
            period=5.2,
            actions=[
                OpaqueFunction(
                    function=create_odom_selector
                )
            ],
        ),

        TimerAction(period=5.5, actions=[nav_cmd_bridge],),
    ])