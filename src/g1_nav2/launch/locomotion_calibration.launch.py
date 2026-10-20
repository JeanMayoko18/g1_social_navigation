#!/usr/bin/env python3

"""
Launch the G1 locomotion calibration node.

Operating modes
---------------
passthrough:

    enable_calibration = false

    The calibration profiles are not loaded and the node must publish:

        u_out = u_in

calibrated:

    enable_calibration = true

    The launch loads two independent profiles:

        1. forward lateral compensation;
        2. forward heading compensation.

Parameter precedence
--------------------
Passthrough mode:

    1. base.yaml
    2. launch overrides

Calibrated mode:

    1. base.yaml
    2. profiles/lateral/<selected_profile>.yaml
    3. profiles/heading/<selected_profile>.yaml
    4. launch overrides

A later source overrides values from earlier sources.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node


# =============================================================================
# LAUNCH VALUE PROCESSING
# =============================================================================


def _parse_boolean(
    value: str,
    *,
    argument_name: str,
) -> bool:
    """Convert one launch argument to a strict boolean."""

    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise RuntimeError(
        f"{argument_name} must be true or false, "
        f"received {value!r}."
    )


def _validate_profile_name(
    profile_name: str,
    *,
    argument_name: str,
) -> str:
    """
    Validate one profile identifier.

    Only letters, digits, underscores, and hyphens are accepted. This prevents
    a launch argument from resolving a path outside its profile directory.
    """

    normalized = profile_name.strip()

    if not normalized:
        raise RuntimeError(
            f"{argument_name} must not be empty when "
            "calibration is enabled."
        )

    if re.fullmatch(
        r"[A-Za-z0-9_-]+",
        normalized,
    ) is None:
        raise RuntimeError(
            f"Invalid {argument_name}: {profile_name!r}."
        )

    return normalized


def _require_file(
    path: Path,
    *,
    description: str,
) -> Path:
    """Require one launch parameter file."""

    if not path.is_file():
        raise RuntimeError(
            f"{description} not found: {path}"
        )

    return path


# =============================================================================
# NODE CONSTRUCTION
# =============================================================================


def _launch_locomotion_calibration(
    context: LaunchContext,
) -> list[Any]:
    """
    Resolve the operating mode and create the calibration node.

    In passthrough mode, model profiles are intentionally not loaded.
    """

    enable_calibration = _parse_boolean(
        LaunchConfiguration(
            "enable_calibration"
        ).perform(context),
        argument_name="enable_calibration",
    )

    use_sim_time = _parse_boolean(
        LaunchConfiguration(
            "use_sim_time"
        ).perform(context),
        argument_name="use_sim_time",
    )

    input_topic = (
        LaunchConfiguration(
            "input_topic"
        )
        .perform(context)
        .strip()
    )

    output_topic = (
        LaunchConfiguration(
            "output_topic"
        )
        .perform(context)
        .strip()
    )

    if not input_topic:
        raise RuntimeError(
            "input_topic must not be empty."
        )

    if not output_topic:
        raise RuntimeError(
            "output_topic must not be empty."
        )

    if input_topic == output_topic:
        raise RuntimeError(
            "input_topic and output_topic must be different "
            "to prevent a command feedback loop."
        )

    package_share = Path(
        get_package_share_directory(
            "g1_nav2"
        )
    )

    parameter_directory = (
        package_share
        / "params"
        / "locomotion_calibration"
    )

    base_path = _require_file(
        parameter_directory
        / "base.yaml",
        description=(
            "Locomotion calibration base parameters"
        ),
    )

    # Base parameters are always required because they also contain topics,
    # diagnostics, and inactive calibration defaults.
    parameter_sources: list[Any] = [
        str(
            base_path
        )
    ]

    lateral_profile_description = "not_applied"
    heading_profile_description = "not_applied"

    # Model profiles are meaningful only when global calibration is enabled.
    if enable_calibration:
        lateral_profile_name = _validate_profile_name(
            LaunchConfiguration(
                "forward_lateral_compensation_profile"
            ).perform(context),
            argument_name=(
                "forward_lateral_compensation_profile"
            ),
        )

        heading_profile_name = _validate_profile_name(
            LaunchConfiguration(
                "forward_heading_compensation_profile"
            ).perform(context),
            argument_name=(
                "forward_heading_compensation_profile"
            ),
        )

        lateral_profile_path = _require_file(
            parameter_directory
            / "profiles"
            / "lateral"
            / f"{lateral_profile_name}.yaml",
            description=(
                "Forward lateral compensation profile"
            ),
        )

        heading_profile_path = _require_file(
            parameter_directory
            / "profiles"
            / "heading"
            / f"{heading_profile_name}.yaml",
            description=(
                "Forward heading compensation profile"
            ),
        )

        parameter_sources.extend(
            [
                str(
                    lateral_profile_path
                ),
                str(
                    heading_profile_path
                ),
            ]
        )

        operating_mode = "calibrated"
        lateral_profile_description = (
            lateral_profile_name
        )
        heading_profile_description = (
            heading_profile_name
        )

    else:
        operating_mode = "passthrough"

    # Launch arguments have final authority. In particular,
    # enable_calibration is not defined in base.yaml.
    launch_overrides = {
        "use_sim_time": use_sim_time,
        "enable_calibration": enable_calibration,
        "input_topic": input_topic,
        "output_topic": output_topic,
    }

    parameter_sources.append(
        launch_overrides
    )

    locomotion_calibration_node = Node(
        package="cmd_vel_qos_adapter",
        executable="locomotion_calibration_node",
        name="locomotion_calibration_node",
        output="screen",
        parameters=parameter_sources,
    )

    return [
        LogInfo(
            msg=(
                "[G1 LOCOMOTION CALIBRATION] "
                f"mode={operating_mode} | "
                f"lateral_profile="
                f"{lateral_profile_description} | "
                f"heading_profile="
                f"{heading_profile_description} | "
                f"input={input_topic} | "
                f"output={output_topic}"
            )
        ),
        locomotion_calibration_node,
    ]


# =============================================================================
# ORCHESTRATION
# =============================================================================


def generate_launch_description() -> LaunchDescription:
    """Build the locomotion calibration launch description."""

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description=(
                    "Use the ROS simulation clock."
                ),
            ),

            DeclareLaunchArgument(
                "enable_calibration",
                default_value="false",
                description=(
                    "Enable calibrated mode. When false, the node "
                    "operates in exact Twist passthrough mode."
                ),
            ),

            DeclareLaunchArgument(
                "input_topic",
                default_value="/cmd_vel",
                description=(
                    "Input command before locomotion calibration."
                ),
            ),

            DeclareLaunchArgument(
                "output_topic",
                default_value="/cmd_vel_nav_scaled",
                description=(
                    "Passthrough or calibrated output command sent "
                    "toward twist_mux."
                ),
            ),

            DeclareLaunchArgument(
                "forward_lateral_compensation_profile",
                default_value="previous_reference_scaled",
                description=(
                    "Forward lateral compensation profile. This "
                    "argument is ignored in passthrough mode."
                ),
            ),

            DeclareLaunchArgument(
                "forward_heading_compensation_profile",
                default_value="previous_reference_scaled",
                description=(
                    "Forward heading compensation profile. This "
                    "argument is ignored in passthrough mode."
                ),
            ),

            OpaqueFunction(
                function=_launch_locomotion_calibration,
            ),
        ]
    )