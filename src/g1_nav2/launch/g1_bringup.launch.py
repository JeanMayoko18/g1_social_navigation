import importlib.util
import os

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


_HELPER_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "nav2_launch_helpers.py",
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "g1_nav2_launch_helpers",
    _HELPER_PATH,
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load Nav2 launch helpers: {_HELPER_PATH}")

_NAV2_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_NAV2_HELPERS)
_yaml_has_twist_mux_config = _NAV2_HELPERS.yaml_has_twist_mux_config
_validate_navigation_mode = _NAV2_HELPERS.validate_navigation_mode

def _resolve_calibration_flag(context):
    raw_value = LaunchConfiguration(
        "enable_locomotion_calibration"
    ).perform(context).strip().lower()

    if raw_value not in {"true", "false"}:
        raise RuntimeError(
            "enable_locomotion_calibration must be true or false, "
            f"received {raw_value!r}."
        )

    #_value = "false" if raw_value == "true" else "true"
    _value = raw_value 

    print(
        "[G1 CALIBRATION] "
        f"enable_locomotion_calibration={raw_value} "
        f"-> enable_calibration={_value}"
    )

    return _value

def resolve_navigation_goal_gate(context):
    """
    Resolve the Nav2 action-remapping gate flag.

    This launch does not decide whether social navigation requires the gate.
    That policy belongs to g1_social_bringup/system.launch.py.

    The resolved value is used only to keep the Nav2 action remapping atomic
    with NavigationGoalGate ownership in the higher-level navigation launch.
    """

    requested = (
        LaunchConfiguration("enable_navigation_goal_gate")
        .perform(context)
        .strip()
        .lower()
    )

    if requested not in {"true", "false"}:
        raise RuntimeError(
            "enable_navigation_goal_gate must be true or false, "
            f"received {requested!r}."
        )

    return [
        SetLaunchConfiguration(
            "resolved_navigation_goal_gate",
            requested,
        ),
        LogInfo(
            msg=(
                "[G1 GOAL GATE] "
                f"enabled={requested}"
            )
        ),
    ]

def generate_launch_description():

    # -------------------------
    # Core args
    # -------------------------
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_rviz = LaunchConfiguration("start_rviz")
    navigation_mode = LaunchConfiguration("navigation_mode")
    controller = LaunchConfiguration("controller")
    ros_distro_arg = DeclareLaunchArgument(
        "ros_distro",
        default_value="auto",
        description=(
            "Nav2 API compatibility: auto, foxy or humble. "
            "auto reads the ROS_DISTRO environment variable."
        ),
    )
    map_yaml = LaunchConfiguration("map")
    navigation_speed = LaunchConfiguration("navigation_speed")

    resolved_navigation_goal_gate = (
        LaunchConfiguration(
            "resolved_navigation_goal_gate"
        )
    )

    # -------------------------
    # Visual pipeline args
    # -------------------------
    start_visual_odometry = LaunchConfiguration("start_visual_odometry")

    # -------------------------
    # Bridge args
    # -------------------------
    input_odom_topic = LaunchConfiguration("input_odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")

    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    body_frame = LaunchConfiguration("body_frame")
    frame = LaunchConfiguration("base")
    lidar_frame = LaunchConfiguration("lidar_frame")
    lidar_frame2d = LaunchConfiguration("lidar_frame2d")

    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw_mode = LaunchConfiguration("yaw_mode")

    min_height = LaunchConfiguration("min_height")
    max_height = LaunchConfiguration("max_height")
    range_min = LaunchConfiguration("range_min")
    range_max = LaunchConfiguration("range_max")
    transform_tolerance = LaunchConfiguration("transform_tolerance")
    pc2scan_params_file = LaunchConfiguration("pc2scan_params_file")
    odommodestate_type = LaunchConfiguration("odommodestate_type")

    # -------------------------
    # Locomotion calibration
    # -------------------------

    enable_locomotion_calibration = LaunchConfiguration(
        "enable_locomotion_calibration"
    )
    forward_lateral_compensation_profile = LaunchConfiguration(
        "forward_lateral_compensation_profile"
    )
    forward_heading_compensation_profile = LaunchConfiguration(
        "forward_heading_compensation_profile"
    )

    
    # -------------------------
    # Social navigation args
    # -------------------------
    perception_profile = LaunchConfiguration(
        "perception_profile"
    )


    # -------------------------
    # Paths
    # -------------------------
    g1_nav2_share = get_package_share_directory("g1_nav2")

    humanoid_nav_bridge_share = get_package_share_directory("humanoid_nav_bridge")
    #g1_visual_odometry_share = get_package_share_directory("g1_visual_odometry")

    pc2scan_params_path_nav2 = os.path.join(g1_nav2_share, "params", "pc2scan.yaml")
    twist_mux_params_path = os.path.join(g1_nav2_share, "params", "twist_mux.yaml")
    pc2scan_params_path_default = pc2scan_params_path_nav2

    nav2_localization_launch = os.path.join(
        g1_nav2_share,
        "launch",
        "nav2_localization.launch.py",
    )
    slam_launch = os.path.join(
        g1_nav2_share,
        "launch",
        "slam.launch.py",
    )

    # -------------------------
    # yaw_mode -> radians
    # -------------------------
    yaw_rad = PythonExpression([
        "(0.0 if '", yaw_mode, "' == '0' else "
        "1.57079632679 if '", yaw_mode, "' == '90p' else "
        "-1.57079632679 if '", yaw_mode, "' == '90m' else "
        "3.14159265359 if '", yaw_mode, "' == '180' else "
        "(float('", yaw_mode, "') * 3.14159265359 / 180.0))"
    ])

    # -------------------------
    # (1) Bridge launch
    # -------------------------
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                humanoid_nav_bridge_share,
                "launch",
                "real_robot_bridge.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,

            "pc2scan_params_file": pc2scan_params_file,

            "input_odom_topic": input_odom_topic,
            "cloud_topic": cloud_topic,

            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "body_frame": body_frame,
            "lidar_frame": lidar_frame,
            "lidar_frame2d": lidar_frame2d,
            "odommodestate_type": odommodestate_type,

            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw_rad,

            "min_height": min_height,
            "max_height": max_height,
            "range_min": range_min,
            "range_max": range_max,
            "transform_tolerance": transform_tolerance,
        }.items(),
    )
    
    # ------------------------------------------------------------------
    # Exclusive navigation branches
    # ------------------------------------------------------------------
    nav2_mode_condition = IfCondition(
        PythonExpression(
            ["'", navigation_mode, "'.lower() == 'nav2'"]
        )
    )
    slam_mode_condition = IfCondition(
        PythonExpression(
            ["'", navigation_mode, "'.lower() == 'slam'"]
        )
    )

    navigation_mode_validation = OpaqueFunction(
        function=_validate_navigation_mode,
    )

    navigation_goal_gate_resolution = (
        OpaqueFunction(
            function=(
                resolve_navigation_goal_gate
            ),
        )
    )

    nav2_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            nav2_localization_launch
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "ros_distro": LaunchConfiguration(
                "ros_distro"
            ),
            "map": map_yaml,
            "base_frame": base_frame,
            "odom_frame": odom_frame,
            "controller": controller,
            "perception_profile": (
                perception_profile
            ),
            "navigation_speed": (
                navigation_speed
            ),
            "enable_navigation_goal_gate": (
                resolved_navigation_goal_gate
            ),
            "enable_locomotion_calibration": (
                enable_locomotion_calibration
            ),
        }.items(),
        condition=nav2_mode_condition,
    )

    slam_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "odom_frame": odom_frame,
            "base_frame": base_frame,
            "map_frame": "map",
            "scan_topic": "/scan",
        }.items(),
        condition=slam_mode_condition,
    )


    locomotion_calibration_ = Node(
        package="cmd_vel_qos_adapter",
        executable="locomotion_calibration_node",
        name="locomotion_calibration_node",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel",
            "output_topic": "/cmd_vel_nav_scaled",

            # Global activation
            #"enable_calibration": True,
            "enable_calibration": False,

            # ================================================================
            # LATENCY DIAGNOSTICS
            # ================================================================

            # Measure the processing overhead introduced by the locomotion
            # calibration node.
            "enable_latency_diagnostics": True,

            "computation_latency_topic":
                "/locomotion_calibration/computation_latency_ms",

            "callback_latency_topic":
                "/locomotion_calibration/callback_latency_ms",

            # ================================================================
            # STRAIGHT-MOTION CALIBRATION
            # ================================================================
            #
            # Derived from:
            #   straight_compensation_campaign_2
            #
            # Candidate to be validated in:
            #   straight_compensation_campaign_3
            # ================================================================

            "enable_forward_heading_compensation": True,
            #"forward_heading_compensation_wz": 0.0122,
            #"forward_heading_compensation_wz": 0.0093,
            "forward_heading_compensation_wz":
                0.009887956027345258,
            "forward_heading_compensation_min_vx": 0.08,

            "enable_forward_lateral_compensation": True,
            #"forward_lateral_compensation_vy": 0.0120,
            #"forward_lateral_compensation_vy": 0.0132,
            #"forward_lateral_compensation_vy": 0.01107385364079094,
            "forward_lateral_compensation_vy":
                0.003491403315154268,
            "forward_lateral_compensation_min_vx": 0.08,

            # ================================================================
            # ROTATION TRANSLATION COMPENSATION
            # ================================================================
            #
            # Derived from:
            #   rotation_translation_compensation_campaign_3
            #
            # Candidate to be validated in:
            #   rotation_translation_campaign_4
            # ================================================================

            "enable_rotation_translation_compensation": True,

            #"clockwise_rotation_vx_compensation": 0.02523,
            #"counterclockwise_rotation_vx_compensation": -0.01150,

            # Right / clockwise
            #"clockwise_rotation_vx_compensation": 0.0417270436913812,
            #"clockwise_rotation_vy_compensation": 0.014577868185518315,
            "clockwise_rotation_vx_compensation":
                0.03738548769622477,
            "clockwise_rotation_vy_compensation":
                0.025340145547797777,

            # Left / counterclockwise
            #"counterclockwise_rotation_vx_compensation": -0.02147190472881444,
            #"counterclockwise_rotation_vy_compensation": 0.010033438238241923,

            "counterclockwise_rotation_vx_compensation":
                -0.016206228428362657,
            "counterclockwise_rotation_vy_compensation":
                0.013911252254673667,

            "rotation_translation_max_linear_command": 1e-6,
            "rotation_translation_min_wz": 1e-6,

            # ================================================================
            # ROTATION SCALING
            # ================================================================
            #
            # Not active. Existing estimates are retained for traceability.
            # ================================================================

            "enable_rotation_scaling": False,
            "clockwise_rotation_scale": 1.066228,
            "counterclockwise_rotation_scale": 0.964944,

            # ================================================================
            # CALIBRATION REFERENCE COMMANDS
            # ================================================================
            #
            # Experimental reference commands used during calibration:
            #
            #   Straight motion:
            #       vx_ref = +0.12 m/s
            #
            #   Pure rotation:
            #       CW  : wz_ref = -0.20 rad/s
            #       CCW : wz_ref = +0.20 rad/s
            #
            # Additive locomotion compensations are interpreted as correction
            # values identified at these reference commands.
            #
            # Runtime corrections are scaled proportionally to the actual
            # incoming command magnitude.
            #
            # Extrapolation is disabled because no multi-speed calibration
            # evidence is currently available above the reference values.
            # ================================================================

            "forward_compensation_reference_vx": 0.12,
            "rotation_translation_reference_wz": 0.20,

            "allow_compensation_extrapolation": False,
            
            # ================================================================
            # TRANSLATION SCALING
            # ================================================================
            #
            # Physical measurements do not currently justify a locomotion-side
            # translation scale. Odometry scaling remains a separate DOG
            # calibration problem.
            # ================================================================

            "enable_translation_scaling": False,
            "translation_scale": 1.0,
        }],
    )

    def launch_locomotion_calibration(context):
        enable_calibration = _resolve_calibration_flag(context)

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        g1_nav2_share,
                        "launch",
                        "locomotion_calibration.launch.py",
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,

                    "enable_calibration": enable_calibration,

                    "forward_lateral_compensation_profile": (
                        forward_lateral_compensation_profile
                    ),

                    "forward_heading_compensation_profile": (
                        forward_heading_compensation_profile
                    ),

                    "input_topic": "/cmd_vel",
                    "output_topic": "/cmd_vel_nav_scaled",
                }.items(),
            )
        ]

    # -------------------------------------------------------------------------
    # Locomotion calibration
    # -------------------------------------------------------------------------
    #
    # The main launch only selects:
    #
    #   - whether calibration is globally active;
    #   - which lateral compensation profile is loaded.
    #
    # G1 parameters and model coefficients are resolved by:
    #
    #   g1_nav2/launch/locomotion_calibration.launch.py
    #
    # Mathematical evaluation remains implemented by:
    #
    #   cmd_vel_qos_adapter/compensation_models.py
    # -------------------------------------------------------------------------

    loccalibration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                g1_nav2_share,
                "launch",
                "locomotion_calibration.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,

            "enable_calibration": (
                enable_locomotion_calibration
           ),

            "forward_lateral_compensation_profile": (
                forward_lateral_compensation_profile
            ),

            "forward_heading_compensation_profile": (
                forward_heading_compensation_profile
            ),

            "input_topic": "/cmd_vel",
            "output_topic": "/cmd_vel_nav_scaled",
        }.items(),
    )
    locomotion_calibration = OpaqueFunction(
        function=launch_locomotion_calibration,
    )

    cmd_vel_nav_scalers = Node(
        package="cmd_vel_qos_adapter",
        executable="cmd_vel_rl_scaler",
        name="cmd_vel_nav_scaler",
        output="screen",
        parameters=[{
            "input_topic": "/cmd_vel",
            "output_topic": "/cmd_vel_nav_scaled",

            # --------------------------------------------------
            # Linear scaling
            # --------------------------------------------------
            # Stop tiny commands completely
            "linear_stop_deadzone": 0.03,

            # Preserve small commands near the goal
            "linear_approach_threshold": 0.10,
            "linear_approach_gain": 0.8,

            # Minimum réellement efficace du G1
            "linear_min_effective": 0.12,

            # Dès que Nav2 dépasse ce niveau,
            # on peut commencer à conserver/amplifier sa variation
            "linear_full_threshold": 0.16,

            # Autoriser un peu de marge au-dessus du seuil minimum
            "max_linear": 0.16,

            # Boost only normal motion commands
            "linear_base_boost": 0.08,
            "linear_gain": 0.4,

            "max_angular": 0.08,
            "angular_min_effective": 0.05,
            "angular_full_threshold": 0.25,

            # --------------------------------------------------
            # Angular scaling
            # --------------------------------------------------
            # Ignore tiny heading corrections
            "angular_deadzone": 0.05,

            # Soft turning near goal orientation
            "angular_approach_threshold": 0.15,
            "angular_approach_gain": 1.0,

            # Stronger turning when really needed
            "angular_gain": 0.3,

            #"enable_adaptive_linear_scale": True,
            "enable_adaptive_linear_scale": False,
            "linear_cmd_scale": 1.0,
            "linear_cmd_scale_min": 0.50,
            "linear_cmd_scale_max": 1.00,
            "linear_adapt_lr": 0.02,
            "adapt_min_cmd_vx": 0.12,
            "adapt_min_odom_vx": 0.03, 

            
            "g1_turn_enter_wz": 0.10,
            "g1_turn_exit_wz": 0.04,
            "g1_forward_max_vx": 0.12,

            "g1_policy": "motion_compatibility",
            #"g1_policy": "exclusive",
            #"g1_policy": "state_machine",

            #"g1_use_path_heading_error": False,
            "g1_use_path_heading_error": True,
            "g1_align_source": "path_error",
            #"g1_align_source": "raw_wz",
            "g1_align_kp": 0.6,
            "g1_align_max_wz": 0.12,
            "g1_align_enter_heading_deg": 14.0,
            "g1_align_exit_heading_deg": 7.0,
            "g1_forward_soft_wz": 0.04,
        }],
    )

    scan_timing_monitor_wifi = Node(
        package="humanoid_nav_bridge",
        executable="scan_timing_monitor",
        name="scan_timing_monitor_wifi",
        output="screen",
        parameters=[{
            "enable_cloud": False,
            "enable_scan_robot": False,
            "enable_scan": True,
            "enable_amcl": True,

            "enable_tf_check": True,

            # Check the same temporal chain required by AMCL.
            "enable_odom_scan_tf_check": True,

            # Check the complete localization chain.
            "enable_map_scan_tf_check": True,

            # Check the latest AMCL localization transform.
            "enable_map_odom_tf_check": True,

            "scan_frame": "nav_laser_fixed",
            "odom_frame": "odom",
            "map_frame": "map",

            # Do not block the executor.
            "tf_lookup_timeout_sec": 0.0,

            "sample_every_n": 10,
            "report_period_sec": 5.0,
            "immediate_tf_sample_every_n": 5,

            "age_change_threshold": 0.20,
            "hz_change_threshold": 1.00,
            "stale_timeout_sec": 3.0,
        }],
    )

    # -------------------------
    # twist_mux
    # -------------------------
    '''
    default_twist_mux_params = {
        "topics": {
            "nav": {"topic": "/cmd_vel", "timeout": 0.3, "priority": 50},
            "teleop": {"topic": "/cmd_vel_teleop", "timeout": 0.3, "priority": 100},
        },
        "output_topic": "/cmd_vel_mux",
    }

    '''

    default_twist_mux_params = {
        "topics": {
            "nav": {"topic": "/cmd_vel_nav_scaled", "timeout": 0.3, "priority": 50},
            "umbmark": {"topic": "/cmd_vel_umbmark", "timeout": 0.3, "priority": 90},
            "teleop": {"topic": "/cmd_vel_teleop", "timeout": 0.3, "priority": 100},
        },
        "output_topic": "/cmd_vel_mux",
    }

    use_yaml = _yaml_has_twist_mux_config(twist_mux_params_path)

    twist_mux_node = Node(
        package="twist_mux",
        executable="twist_mux",
        name="twist_mux",
        output="screen",
        parameters=[
            twist_mux_params_path
        ] if use_yaml else [
            default_twist_mux_params
        ],
        remappings=[
            ("/cmd_vel_out", "/cmd_vel_mux"),
        ],
    )

    qos_adapter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("cmd_vel_qos_adapter"),
                "launch",
                "cmd_vel_qos_adapter.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "input_topic": "/cmd_vel_mux",
            "output_topic": "/cmd_vel_rlsar",
        }.items(),
    )

    # -------------------------
    # RViz
    # -------------------------
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "rviz_launch.py",
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(start_rviz),
    )

    log_params = LogInfo(
        msg=[
            "G1 REAL ROBOT BRINGUP | mode = ", navigation_mode,
            " | map = ", map_yaml,
            " | controller = ", controller,
            " | navigation speed = ", navigation_speed, " m/s",
        ]
    )

    # 3. map_republisher
    map_republisher = Node(
        package="g1_nav2",
        executable="map_republisher",
        name="map_republisher",
        output="screen",
        condition=nav2_mode_condition,
    )

    # -------------------------
    # Startup order
    # -------------------------
    
    scan_timing_monitor_wifi_delayed = TimerAction(
        period=49.0,
        actions=[scan_timing_monitor_wifi]
    )
    

    bridge_delayed = TimerAction(
        period=2.0,
        actions=[bridge_launch],
    )

    map_republisher_delayed = TimerAction(
        period=2.0,
        actions=[map_republisher],
    )


    #nav_scaler_delayed = TimerAction(period=12.0, actions=[cmd_vel_nav_scalers])
    locomotion_calibration_delayed = TimerAction(period=12.0, actions=[locomotion_calibration])
    twist_mux_delayed = TimerAction(period=13.0, actions=[twist_mux_node])
    qos_adapter_delayed = TimerAction(period=14.0, actions=[qos_adapter])

    rviz_delayed = TimerAction(period=15.0, actions=[rviz])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false", description="Real robot: keep false unless /clock exists",),
        DeclareLaunchArgument("start_rviz", default_value="false", description="Start RViz (true/false).", ),
        DeclareLaunchArgument("start_visual_odometry", default_value="true", description="Start visual odometry pipeline.", ),
        DeclareLaunchArgument(
            "navigation_mode",
            default_value="nav2",
            description=(
                "Exclusive navigation branch: nav2 for map localization "
                "and navigation, or slam for online mapping."
            ),
        ),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(
                os.path.expanduser("~"),
                "g1_ws",
                "maps",
                "mic.yaml",
            ),
            description="Existing map YAML used only when navigation_mode=nav2.",
        ),

        DeclareLaunchArgument("input_odom_topic", default_value="/dog_odom"),
        DeclareLaunchArgument("cloud_topic", default_value="/utlidar/cloud_livox_mid360"),
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument("base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("body_frame", default_value="robot_center"),
        DeclareLaunchArgument("base", default_value="base"),
        DeclareLaunchArgument("lidar_frame", default_value="livox_frame"),
        DeclareLaunchArgument("lidar_frame2d", default_value="livox_2d"),

        DeclareLaunchArgument(
            "pc2scan_params_file",
            default_value=pc2scan_params_path_default,
            description="YAML params for pointcloud_to_laserscan",
        ),

        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.0"),
        DeclareLaunchArgument("roll", default_value="0.0"),
        DeclareLaunchArgument("pitch", default_value="0.0"),
        DeclareLaunchArgument(
            "yaw_mode",
            default_value="0",
            description="Yaw LiDAR: '0','90p','90m','180' OR numeric degrees like '45'",
        ),

        DeclareLaunchArgument(
            "controller",
            default_value="dwb",
            description=(
                "Nav2 path controller: dwb or rpp. "
                "Ignored when navigation_mode=slam."
            ),
        ),

        DeclareLaunchArgument(
            "perception_profile",
            default_value="A",
            description=(
                "Navigation perception profile: "
                "A=LiDAR, "
                "B=LiDAR+camera, "
                "C=LiDAR+social, "
                "D=LiDAR+camera+social"
            ),
        ),

        DeclareLaunchArgument(
            "navigation_speed",
            default_value="0.20",
            description=(
                "Experimental Nav2 maximum linear speed in m/s. "
                "Supported values: 0.11, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35. "
                "Applied to DWB max_vel_x/max_speed_xy or to RPP "
                "desired_linear_vel."
            ),
        ),

        DeclareLaunchArgument(
            "enable_navigation_goal_gate",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Enable Nav2 action remapping for the external "
                "NavigationGoalGate. The higher-level system launch "
                "owns the activation policy."
            ),
        ),


        DeclareLaunchArgument("min_height", default_value="0.05"),
        DeclareLaunchArgument("max_height", default_value="0.25"),
        DeclareLaunchArgument("range_min", default_value="0.45"),
        DeclareLaunchArgument("range_max", default_value="8.0"),
        DeclareLaunchArgument("transform_tolerance", default_value="0.1"),

        DeclareLaunchArgument(
            "odommodestate_type",
            default_value="unitree_go/msg/SportModeState",
            description="Exact ROS 2 type for /odommodestate",
        ),

        DeclareLaunchArgument(
            "forward_lateral_compensation_profile",
            default_value="previous_reference_scaled",
            description=(
                "Forward lateral compensation profile."
            ),
        ),

        DeclareLaunchArgument(
            "forward_heading_compensation_profile",
            default_value="previous_reference_scaled",
            description=(
                "Forward heading compensation profile."
            ),
        ),

        DeclareLaunchArgument(
            "enable_locomotion_calibration",
            default_value="false",
            description=(
                "Enable command-level locomotion calibration."
            ),
        ),

        DeclareLaunchArgument(
            "locomotion_calibration_profile",
            default_value="previous_reference_scaled",
            description=(
                "Locomotion calibration profile loaded by the "
                "G1 locomotion calibration sub-launch."
            ),
        ),


        ros_distro_arg,
        navigation_mode_validation,
        navigation_goal_gate_resolution,
        log_params,

        # Exactly one of these includes is enabled.
        nav2_stack,
        slam_stack,

        #bridge_delayed,
        map_republisher_delayed,
        #scan_timing_monitor_wifi_delayed,

        locomotion_calibration_delayed,
        twist_mux_delayed,
        qos_adapter_delayed,
        rviz_delayed,
    ])
