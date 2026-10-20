"""Helpers shared by the real-robot Nav2 launch files.

This module owns runtime configuration resolution.  It deliberately contains
no ``generate_launch_description()`` function, so importing it never starts a
second launch description or a second Nav2 stack.
"""

import copy
import math
import os
import tempfile
from typing import Any, Dict, List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import (
    Node,
    SetRemap,
)


SUPPORTED_NAVIGATION_SPEEDS = {
    0.11,
    0.12,
    0.16,
    0.20,
    0.25,
    0.30,
    0.35
}

def yaml_has_twist_mux_config(path: str) -> bool:
    """Return whether a YAML file contains a usable twist_mux section."""

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        return False

    if not isinstance(data, dict):
        return False

    for root_key in ("twist_mux", "/**"):
        root = data.get(root_key)
        if not isinstance(root, dict):
            continue

        parameters = root.get("ros__parameters", {})
        if isinstance(parameters, dict) and (
            "topics" in parameters or "output_topic" in parameters
        ):
            return True

    return False

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return ``base``."""

    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value

    return base

def resolve_ros_distro(context) -> str:
    """Resolve auto/foxy/humble from the launch argument and environment."""

    requested = LaunchConfiguration("ros_distro").perform(context).strip().lower()

    if requested == "auto":
        requested = os.environ.get("ROS_DISTRO", "").strip().lower()

    if requested not in {"foxy", "humble"}:
        raise RuntimeError(
            "Unsupported ROS distribution: "
            f"{requested!r}. Expected foxy or humble. Source the correct "
            "ROS environment or pass ros_distro:=foxy|humble."
        )

    return requested

def resolve_navigation_speed(context) -> float:
    """Parse and validate the requested experimental maximum linear speed."""

    text = LaunchConfiguration("navigation_speed").perform(context).strip()

    try:
        requested = float(text)
    except ValueError as exc:
        raise RuntimeError(
            f"navigation_speed must be a number, received {text!r}."
        ) from exc

    for supported in SUPPORTED_NAVIGATION_SPEEDS:
        if math.isclose(requested, supported, rel_tol=0.0, abs_tol=1.0e-9):
            return supported

    expected = ", ".join(f"{value:.2f}" for value in SUPPORTED_NAVIGATION_SPEEDS)
    raise RuntimeError(
        f"Unsupported navigation_speed={requested:.3f} m/s. "
        f"Expected one of: {expected} m/s."
    )

def validate_navigation_mode(context):
    """Reject invalid bringup modes before any navigation process starts."""

    mode = LaunchConfiguration("navigation_mode").perform(context).strip().lower()

    if mode not in {"nav2", "slam"}:
        raise RuntimeError(
            f"Unsupported navigation_mode={mode!r}. Expected nav2 or slam."
        )

    return [LogInfo(msg=f"[G1 BRINGUP] navigation_mode={mode}")]

def _controller_parameters(effective_params: Dict[str, Any]) -> Dict[str, Any]:
    """Return controller_server.ros__parameters with explicit validation."""

    controller_server = effective_params.get("controller_server")
    if not isinstance(controller_server, dict):
        raise RuntimeError("Nav2 YAML has no controller_server section.")

    parameters = controller_server.get("ros__parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError(
            "Nav2 YAML has no controller_server.ros__parameters section."
        )

    return parameters

def apply_navigation_speed(
    effective_params: Dict[str, Any],
    controller: str,
    speed: float,
) -> List[str]:
    """Apply the speed to the active FollowPath controller configuration.

    DWB uses ``max_vel_x`` and ``max_speed_xy``. Regulated Pure Pursuit uses
    ``desired_linear_vel``.
    """

    parameters = _controller_parameters(effective_params)
    follow_path = parameters.get("FollowPath")

    if not isinstance(follow_path, dict):
        raise RuntimeError(
            "Nav2 YAML has no controller_server.ros__parameters.FollowPath "
            "section."
        )

    if controller == "dwb":
        follow_path["max_vel_x"] = speed
        follow_path["max_speed_xy"] = speed

        changed = [
            "FollowPath.max_vel_x",
            "FollowPath.max_speed_xy",
        ]

        # Update the velocity smoother only if it is configured.
        # Do not add a node or dependency to the Foxy branch.
        smoother_section = effective_params.get("velocity_smoother")

        if smoother_section is not None:
            smoother = smoother_section.get("ros__parameters")

            if not isinstance(smoother, dict):
                raise RuntimeError(
                    "Invalid velocity_smoother.ros__parameters section."
                )

            for key in ("max_velocity", "min_velocity"):
                values = smoother.get(key)

                if not isinstance(values, list) or len(values) != 3:
                    raise RuntimeError(
                        f"velocity_smoother.{key} must contain [vx, vy, wz]."
                    )

            max_velocity = list(smoother["max_velocity"])
            min_velocity = list(smoother["min_velocity"])

            max_velocity[0] = float(speed)

            # Preserve any existing restriction on reverse motion.
            if min_velocity[0] < 0.0:
                min_velocity[0] = -float(speed)

            smoother["max_velocity"] = max_velocity
            smoother["min_velocity"] = min_velocity

            changed.extend([
                "velocity_smoother.max_velocity",
                "velocity_smoother.min_velocity",
            ])

        return changed

    if controller == "rpp":
        follow_path["desired_linear_vel"] = speed
        return ["FollowPath.desired_linear_vel"]

    raise RuntimeError(
        "Unable to identify the FollowPath controller parameters for "
        f"controller={controller!r}."
    )

def launch_amcl_for_distro(context, amcl_params_path):
    """Create AMCL using the distro-specific YAML overlay."""

    ros_distro = resolve_ros_distro(context)

    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        == "true"
    )

    base_frame = LaunchConfiguration("base_frame").perform(context)
    odom_frame = LaunchConfiguration("odom_frame").perform(context)

    distro_overlay_path = os.path.join(
        os.path.dirname(amcl_params_path),
        "distro_overlays",
        f"amcl_{ros_distro}.yaml",
    )

    effective_params = copy.deepcopy(
        _load_yaml(amcl_params_path, "base AMCL")
    )

    deep_merge(
        effective_params,
        _load_yaml(distro_overlay_path, "AMCL ROS distribution overlay"),
    )

    runtime_dir = os.path.join(
        tempfile.gettempdir(),
        "g1_nav2",
    )
    os.makedirs(runtime_dir, exist_ok=True)

    runtime_path = os.path.join(
        runtime_dir,
        f"amcl_{ros_distro}.yaml",
    )

    with open(runtime_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            effective_params,
            stream,
            sort_keys=False,
        )

    robot_model_type = (
        effective_params
        .get("amcl", {})
        .get("ros__parameters", {})
        .get("robot_model_type", "<undefined>")
    )

    return [
        LogInfo(
            msg=(
                "[G1 AMCL] "
                f"ros_distro={ros_distro} | "
                f"robot_model_type={robot_model_type} | "
                f"params={runtime_path}"
            )
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=[
                runtime_path,
                {
                    "use_sim_time": use_sim_time,
                    "base_frame_id": base_frame,
                    "odom_frame_id": odom_frame,
                    "global_frame_id": "map",
                },
            ],
            respawn=True,
            respawn_delay=2.0,
        ),
    ]

def _load_yaml(path: str, label: str) -> Dict[str, Any]:
    """Load and validate one mapping-style YAML file."""

    if not os.path.exists(path):
        raise RuntimeError(f"{label} YAML not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Unable to read {label} YAML: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid {label} YAML: {path}")

    return data

def launch_nav2_with_runtime_profile(context, g1_nav2_share: str):
    """Build one effective Nav2 YAML and start exactly one Nav2 stack.

    Merge precedence is:

        base -> ROS distribution -> perception -> experimental speed

    Consequently, the selected speed cannot be overwritten by an earlier YAML
    layer and no post-start ``ros2 param set`` race is required.
    """

    controller = LaunchConfiguration("controller").perform(context).strip().lower()
    perception = (
        LaunchConfiguration("perception_profile").perform(context).strip().upper()
    )
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    ros_distro = resolve_ros_distro(context)
    navigation_speed = resolve_navigation_speed(context)

    goal_gate_value = (
        LaunchConfiguration(
            "enable_navigation_goal_gate"
        )
        .perform(context)
        .strip()
        .lower()
    )

    if goal_gate_value not in {
        "true",
        "false",
    }:
        raise RuntimeError(
            "enable_navigation_goal_gate must be "
            f"true or false, received {goal_gate_value!r}."
        )

    goal_gate_enabled = (
        goal_gate_value == "true"
    )

    base_files = {
        "dwb": os.path.join(g1_nav2_share, "params", "nav2_g1_amcl_dwb.yaml"),
        "rpp": os.path.join(g1_nav2_share, "params", "nav2_g1_amcl_rpp.yaml"),
    }
    if controller not in base_files:
        raise RuntimeError(
            f"Unsupported controller={controller!r}. Expected dwb or rpp."
        )

    if perception not in {"A", "B", "C", "D"}:
        raise RuntimeError(
            f"Unsupported perception_profile={perception!r}. Expected A, B, C or D."
        )

    base_path = base_files[controller]
    distro_path = os.path.join(
        g1_nav2_share,
        "params",
        "distro_overlays",
        f"nav2_{ros_distro}.yaml",
    )
    perception_path = os.path.join(
        g1_nav2_share,
        "params",
        "perception_profiles",
        f"perception_{perception.lower()}.yaml",
    )

    effective_params = copy.deepcopy(_load_yaml(base_path, "base Nav2"))
    deep_merge(effective_params, _load_yaml(distro_path, "ROS distribution overlay"))
    deep_merge(effective_params, _load_yaml(perception_path, "perception overlay"))
    changed_parameters = apply_navigation_speed(
        effective_params,
        controller,
        navigation_speed,
    )

    runtime_dir = os.path.join(tempfile.gettempdir(), "g1_nav2")
    os.makedirs(runtime_dir, exist_ok=True)
    speed_tag = int(round(navigation_speed * 100.0))
    runtime_path = os.path.join(
        runtime_dir,
        (
            f"nav2_{controller}_{ros_distro}_"
            f"perception_{perception.lower()}_vx_{speed_tag:03d}.yaml"
        ),
    )

    with open(runtime_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(effective_params, stream, sort_keys=False)

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "navigation_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "autostart": "true",
            "params_file": runtime_path,
            "namespace": "",
            "map_subscribe_transient_local": "true",
        }.items(),
    )

    if goal_gate_enabled:
        nav2_action = GroupAction(
            actions=[
                # Action base name.
                SetRemap(
                    src="/navigate_to_pose",
                    dst="/navigate_to_pose_internal",
                ),

                # Action services.
                SetRemap(
                    src="/navigate_to_pose/_action/send_goal",
                    dst=(
                        "/navigate_to_pose_internal/"
                        "_action/send_goal"
                    ),
                ),
                SetRemap(
                    src="/navigate_to_pose/_action/get_result",
                    dst=(
                        "/navigate_to_pose_internal/"
                        "_action/get_result"
                    ),
                ),
                SetRemap(
                    src="/navigate_to_pose/_action/cancel_goal",
                    dst=(
                        "/navigate_to_pose_internal/"
                        "_action/cancel_goal"
                    ),
                ),

                # Action topics.
                SetRemap(
                    src="/navigate_to_pose/_action/feedback",
                    dst=(
                        "/navigate_to_pose_internal/"
                        "_action/feedback"
                    ),
                ),
                SetRemap(
                    src="/navigate_to_pose/_action/status",
                    dst=(
                        "/navigate_to_pose_internal/"
                        "_action/status"
                    ),
                ),

                nav2_launch,
            ]
        )

    else:
        nav2_action = nav2_launch


    return [
        LogInfo(
            msg=(
                "[G1 NAV2] "
                f"ros_distro={ros_distro} | "
                f"controller={controller} | "
                f"perception={perception} | "
                f"navigation_speed={navigation_speed:.2f} m/s | "
                f"goal_gate={goal_gate_enabled} | "
                f"overrides={','.join(changed_parameters)} | "
                f"params={runtime_path}"
            )
        ),

        nav2_action,
    ]
