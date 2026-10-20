import os
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
#from launch_ros.parameter_descriptions import ParameterFile


def _yaml_has_twist_mux_config(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    if "twist_mux" in data and isinstance(data["twist_mux"], dict):
        rp = data["twist_mux"].get("ros__parameters", {})
        if isinstance(rp, dict) and (("topics" in rp) or ("output_topic" in rp)):
            return True

    if "/**" in data and isinstance(data["/**"], dict):
        rp = data["/**"].get("ros__parameters", {})
        if isinstance(rp, dict) and (("topics" in rp) or ("output_topic" in rp)):
            return True

    return False


def generate_launch_description():
    # -------------------------
    # Core args
    # -------------------------
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_rviz = LaunchConfiguration("start_rviz")
    controller = LaunchConfiguration("controller")
    map_yaml = LaunchConfiguration("map")

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
    # Paths
    # -------------------------
    g1_nav2_share = get_package_share_directory("g1_nav2")
    humanoid_nav_bridge_share = get_package_share_directory("humanoid_nav_bridge")
    g1_visual_odometry_share = get_package_share_directory("g1_visual_odometry")

    nav2_params_dwb_path = os.path.join(g1_nav2_share, "params", "nav2_g1_amcl_dwb.yaml")
    nav2_params_rpp_path = os.path.join(g1_nav2_share, "params", "nav2_g1_amcl_rpp.yaml")
    nav2_params_slam_path = os.path.join(g1_nav2_share, "params", "nav2_g1_slam_real.yaml")

    nav2_params_path = PythonExpression([
        "'",
        nav2_params_slam_path,
        "' if '", controller, "' == 'slam' else '",
        nav2_params_rpp_path,
        "' if '", controller, "' == 'rpp' else '",
        nav2_params_dwb_path,
        "'"
    ])

    amcl_params_path = os.path.join(g1_nav2_share, "params", "amcl.yaml")
    pc2scan_params_path_nav2 = os.path.join(g1_nav2_share, "params", "pc2scan.yaml")
    twist_mux_params_path = os.path.join(g1_nav2_share, "params", "twist_mux.yaml")
    pc2scan_params_path_default = pc2scan_params_path_nav2

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
    # (0) Visual odometry pipeline
    # -------------------------
    visual_pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                g1_visual_odometry_share,
                "launch",
                "g1_visual_pipeline.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            # ajoute ici les arguments réels attendus par ton launch VO
            # ex:
            # "odom_frame": odom_frame,
            # "base_frame": base_frame,
            # "body_frame": body_frame,
        }.items(),
        condition=IfCondition(start_visual_odometry),
    )

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
    
    # -------------------------
    # (2) Map server + AMCL
    # -------------------------
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "yaml_filename": map_yaml,
        }],
    )

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            amcl_params_path,
            {
                "use_sim_time": use_sim_time,
                "base_frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "global_frame_id": "map",
                "scan_topic": "/scan_amcl",
                "tf_broadcast": True,
                "transform_tolerance": 5.0,
            },
        ],
        respawn=True,
        respawn_delay=2.0,
    )

    lifecycle_manager_localization_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    # -------------------------
    # (3) Nav2 Navigation
    # -------------------------
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "navigation_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params_path,
            "namespace": "",
            "use_namespace": "false",
        }.items(),
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
            "linear_approach_threshold": 0.08,
            "linear_approach_gain": 1.0,
            "linear_min_effective": 0.18,
            "linear_full_threshold": 0.14,

            # Boost only normal motion commands
            "linear_base_boost": 0.18,
            "linear_gain": 0.3,
            "max_linear": 0.22,

            "max_angular": 0.18,
            "angular_min_effective": 0.12,
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
            "angular_gain": 0.8,

            #"enable_adaptive_linear_scale": True,
            "enable_adaptive_linear_scale": False,
            "linear_cmd_scale": 1.0,
            "linear_cmd_scale_min": 0.50,
            "linear_cmd_scale_max": 1.00,
            "linear_adapt_lr": 0.02,
            "adapt_min_cmd_vx": 0.12,
            "adapt_min_odom_vx": 0.03, 
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
            "tf_target_frame": "base_footprint",
            "tf_source_frame": "nav_laser_fixed",
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
            "REAL ROBOT bringup (AMCL + Visual Pipeline) | map = ", map_yaml,
            " | AMCL params = ", amcl_params_path,
            " | NAV2 params = ", nav2_params_path,
        ]
    )

    # 3. map_republisher
    map_republisher = Node(
        package="g1_nav2",
        executable="map_republisher",
        name="map_republisher",
        output="screen",
    )

    # -------------------------
    # Startup order
    # -------------------------
    visual_pipeline_delayed = TimerAction(period=1.0, actions=[visual_pipeline_launch])
    bridge_delayed = TimerAction(period=2.0, actions=[bridge_launch])

    map_server_delayed = TimerAction(period=15.0, actions=[map_server_node])
    
    map_republisher_delayed = TimerAction(period=17.0, actions=[map_republisher])
    amcl_delayed = TimerAction(period=19.0, actions=[amcl_node])
    lifecycle_loc_delayed = TimerAction(period=20.0, actions=[lifecycle_manager_localization_node])

    # Time for /odom/raw, EKF et TF(odom->base_footprint)
    nav2_delayed = TimerAction(period=35.0, actions=[nav2])
    nav_scaler_delayed = TimerAction(period=38.0, actions=[cmd_vel_nav_scalers])
    twist_mux_delayed = TimerAction(period=41.0, actions=[twist_mux_node])
    qos_adapter_delayed = TimerAction(period=45.0, actions=[qos_adapter])
    
    scan_timing_monitor_wifi_delayed = TimerAction(
        period=49.0,
        actions=[scan_timing_monitor_wifi]
    )

    rviz_delayed = TimerAction(period=24.0, actions=[rviz])

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Real robot: keep false unless /clock exists",
        ),
        DeclareLaunchArgument(
            "start_rviz",
            default_value="false",
            description="Start RViz (true/false).",
        ),
        DeclareLaunchArgument(
            "start_visual_odometry",
            default_value="true",
            description="Start visual odometry pipeline.",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(os.path.expanduser("~"), "g1_ws", "maps", "g1_map.yaml"),
            description="Existing map yaml file for AMCL (map_server).",
        ),

        DeclareLaunchArgument("input_odom_topic", default_value="/dog_odom"),
        #DeclareLaunchArgument("input_odom_topic", default_value="/repub/dog_odom"),
        DeclareLaunchArgument("cloud_topic", default_value="/utlidar/cloud_livox_mid360"),
        #DeclareLaunchArgument("cloud_topic", default_value="/repub/mola/cloud_xyz_i"),
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
            description="Navigation mode: 'dwb', 'rpp', or 'slam'. Default: dwb",
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


        #log_params,
        #visual_pipeline_delayed,
        #bridge_delayed,
        #scan_relay_delayed,
        #map_server_delayed,
        
        #map_republisher_delayed,
        #amcl_delayed,
        #lifecycle_loc_delayed,
        
        #scan_timing_monitor_wifi_delayed,
        nav2_delayed,

        nav_scaler_delayed,
        twist_mux_delayed,
        qos_adapter_delayed,
        rviz_delayed,
    ])
