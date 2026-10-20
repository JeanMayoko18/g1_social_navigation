"""Launch map localization and Nav2 navigation for the Unitree G1."""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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

_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


def generate_launch_description():
    """Create the AMCL and Nav2 branch of the main G1 bringup."""

    g1_nav2_share = get_package_share_directory("g1_nav2")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")

    enable_locomotion_calibration = LaunchConfiguration(
        "enable_locomotion_calibration"
    )

    def launch_amcl(context):
        raw_value = enable_locomotion_calibration.perform(context)
        enabled = raw_value.strip().lower() == "true"

        amcl_params_path = os.path.join(
            g1_nav2_share,
            "params",
            "amcl.yaml" if enabled else "nav_amcl.yaml",
        )

        print(
            "[G1 AMCL] enable_locomotion_calibration =",
            repr(raw_value),
        )
        print(
            "[G1 AMCL] selected params =",
            amcl_params_path,
        )

        return _HELPERS.launch_amcl_for_distro(
            context,
            amcl_params_path,
        )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "yaml_filename": map_yaml,
            }
        ],
    )

    amcl_supervisor = Node(
        package="g1_nav2",
        executable="amcl_lifecycle_supervisor",
        name="amcl_lifecycle_supervisor",
        output="screen",
    )
    amcl = OpaqueFunction(
        function=launch_amcl,
    )

    localization_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server", "amcl"],
            }
        ],
    )

    nav2 = OpaqueFunction(
        function=_HELPERS.launch_nav2_with_runtime_profile,
        args=[g1_nav2_share],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("ros_distro", default_value="auto"),
            DeclareLaunchArgument("map"),
            DeclareLaunchArgument("base_frame", default_value="base_footprint"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("controller", default_value="dwb"),
            DeclareLaunchArgument("perception_profile", default_value="A"),
            DeclareLaunchArgument("navigation_speed", default_value="0.20"),
            DeclareLaunchArgument("enable_navigation_goal_gate", default_value="false", description=(
                "Remap the Nav2 NavigateToPose server to "
                    "/navigate_to_pose_internal when the goal gate is enabled."
                ),
            ),
            DeclareLaunchArgument(
                "enable_locomotion_calibration",
                default_value="false",
                description=(
                    "Select AMCL parameter profile: "
                    "false=nav_amcl.yaml, true=amcl.yaml."
                ),
            ),

            TimerAction(period=1.0, actions=[map_server]),
            TimerAction(period=3.0, actions=[amcl]),
            TimerAction(
                period=4.0,
                actions=[localization_lifecycle_manager],
            ),
            TimerAction(period=6.0, actions=[amcl_supervisor]),
            TimerAction(period=10.0, actions=[nav2]),
            
        ]
    )
