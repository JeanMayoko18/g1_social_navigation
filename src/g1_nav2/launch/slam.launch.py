"""Launch SLAM Toolbox for the Unitree G1 in asynchronous mapping mode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the SLAM branch of the main G1 bringup."""

    use_sim_time = LaunchConfiguration("use_sim_time")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    map_frame = LaunchConfiguration("map_frame")
    scan_topic = LaunchConfiguration("scan_topic")

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "odom_frame": odom_frame,
                "base_frame": base_frame,
                "map_frame": map_frame,
                "scan_topic": scan_topic,
                "solver_plugin": "solver_plugins::CeresSolver",
                "ceres_linear_solver": "SPARSE_NORMAL_CHOLESKY",
                "ceres_preconditioner": "SCHUR_JACOBI",
                "ceres_trust_strategy": "LEVENBERG_MARQUARDT",
                "ceres_dogleg_type": "TRADITIONAL_DOGLEG",
                "ceres_loss_function": "None",
                "mode": "mapping",
                "resolution": 0.05,
                "min_laser_range": 0.3,
                "max_laser_range": 20.0,
                "minimum_time_interval": 0.5,
                "transform_publish_period": 0.05,
                "map_update_interval": 1.0,
                "transform_timeout": 0.2,
                "tf_buffer_duration": 30.0,
                "do_loop_closing": True,
                "loop_search_maximum_distance": 3.0,
                "loop_match_minimum_chain_size": 10,
                "loop_match_maximum_variance_coarse": 3.0,
                "loop_match_minimum_response_coarse": 0.35,
                "loop_match_minimum_response_fine": 0.45,
                "use_scan_matching": True,
                "use_scan_barycenter": True,
                "minimum_travel_distance": 0.3,
                "minimum_travel_heading": 0.3,
                "scan_buffer_size": 10,
                "scan_buffer_maximum_scan_distance": 10.0,
                "link_match_minimum_response_fine": 0.1,
                "link_scan_maximum_distance": 1.5,
                "correlation_search_space_dimension": 0.5,
                "correlation_search_space_resolution": 0.01,
                "correlation_search_space_smear_deviation": 0.1,
                "loop_search_space_dimension": 8.0,
                "loop_search_space_resolution": 0.05,
                "loop_search_space_smear_deviation": 0.03,
                "distance_variance_penalty": 0.5,
                "angle_variance_penalty": 1.0,
                "fine_search_angle_offset": 0.00349,
                "coarse_search_angle_offset": 0.349,
                "coarse_angle_resolution": 0.0349,
                "minimum_angle_penalty": 0.9,
                "minimum_distance_penalty": 0.5,
                "use_response_expansion": True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_footprint",
            ),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            slam_toolbox_node,
        ]
    )
