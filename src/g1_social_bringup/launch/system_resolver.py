"""
G1 system architecture resolver.

FLOW
====
User configuration
    |
    +-- profile: A | B | C | D
    +-- interaction: none | dialogue
    +-- deployment:
    |      pc_all
    |      robot_all
    |      robot_perception
    |      pc_perception
    |
    +-- host: pc | robot
    +-- ros_distro: auto | foxy | humble
           |
           v
    resolve_system()
           |
           v
    SystemPlan
           |
           +-- local roles
           +-- global capabilities
           +-- validation

RESPONSIBILITIES
================
This module decides WHAT must run and WHERE it must run.

It does not create ROS nodes.
It does not launch hardware.
It does not import ROS4HRI.
It does not inspect package availability.

Matrix of roles and capabilities:
=================
DEPLOYMENT         YOLO     SOCIAL PERCEPTION    RGB INGRESS    DEPTH/INFO INGRESS
----------------------------------------------------------------------------------
pc_all             PC       PC                   yes            yes
pc_perception      PC       PC                   yes            yes
robot_perception   PC       ROBOT                yes            no
robot_all          ROBOT    ROBOT                no             no

HARDWARE BOUNDARY
=================
rl_sar is started independently on the robot.

rl_sar owns:
    - robot hardware access
    - LiDAR acquisition
    - RealSense acquisition
    - IMU acquisition
    - canonical odometry
    - hardware TF
    - final low-level velocity command interface

This resolver only manages the navigation/social/HRI architecture
above that hardware boundary.
"""

from dataclasses import dataclass
import os
from typing import Dict, Tuple


VALID_PROFILES = ("A", "B", "C", "D")
VALID_INTERACTIONS = ("none", "dialogue")
VALID_DEPLOYMENTS = (
    "pc_all",
    "robot_all",
    "robot_perception",
    "pc_perception",
)
VALID_HOSTS = ("pc", "robot")
VALID_ROS_DISTROS = ("auto", "foxy", "humble")


@dataclass(frozen=True)
class SystemPlan:
    profile: str
    interaction: str
    deployment: str
    host: str
    ros_distro: str

    camera_obstacles: bool
    social_navigation: bool
    explanation: bool
    speech_required: bool
    dialogue: bool

    role_hosts: Dict[str, str]

    @property
    def local_roles(self) -> Tuple[str, ...]:
        return tuple(
            role
            for role, role_host in self.role_hosts.items()
            if role_host == self.host
        )

    def owns(self, role: str) -> bool:
        return self.role_hosts.get(role) == self.host


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def _resolve_ros_distro(requested: str) -> str:
    requested = _normalize(requested)

    if requested not in VALID_ROS_DISTROS:
        raise RuntimeError(
            f"Invalid ros_distro '{requested}'. "
            f"Supported values: {', '.join(VALID_ROS_DISTROS)}"
        )

    if requested != "auto":
        return requested

    detected = _normalize(os.environ.get("ROS_DISTRO", ""))

    if detected not in ("foxy", "humble"):
        raise RuntimeError(
            "Unable to resolve ROS distribution. "
            "Set ROS_DISTRO to 'foxy' or 'humble', or pass "
            "ros_distro:=foxy / ros_distro:=humble explicitly."
        )

    return detected


def _resolve_role_hosts(deployment: str) -> Dict[str, str]:
    """
    Resolve movable software roles.

    Hardware is intentionally absent because rl_sar is external
    to this architecture.
    """

    if deployment == "pc_all":
        return {
            "vision_inference": "pc",
            "social_perception": "pc",
            "social_decision": "pc",
            "navigation": "pc",
            "explanation": "pc",
            "speech": "pc",
            "dialogue": "pc",
            "visualization": "pc",
        }

    if deployment == "robot_all":
        return {
            "vision_inference": "robot",
            "social_perception": "robot",
            "social_decision": "robot",
            "navigation": "robot",
            "explanation": "robot",
            "speech": "robot",
            "dialogue": "robot",
            "visualization": "pc",
        }

    if deployment == "robot_perception":
        return {
            # GPU-heavy vision inference stays on the PC by default.
            # Only the downstream social perception pipeline moves
            # to the robot.
            "vision_inference": "pc",
            "social_perception": "robot",
            "social_decision": "pc",
            "navigation": "pc",
            "explanation": "pc",
            "speech": "robot",
            "dialogue": "pc",
            "visualization": "pc",
        }

    if deployment == "pc_perception":
        return {
            "vision_inference": "pc",
            "social_perception": "pc",
            "social_decision": "pc",
            "navigation": "pc",
            "explanation": "pc",
            "speech": "robot",
            "dialogue": "pc",
            "visualization": "pc",
        }

    raise RuntimeError(
        f"Unsupported deployment: {deployment}"
    )


def resolve_system(
    *,
    profile: str,
    interaction: str,
    deployment: str,
    host: str,
    ros_distro: str,
) -> SystemPlan:
    profile = str(profile).strip().upper()
    interaction = _normalize(interaction)
    deployment = _normalize(deployment)
    host = _normalize(host)

    if profile not in VALID_PROFILES:
        raise RuntimeError(
            f"Invalid profile '{profile}'. "
            f"Supported values: {', '.join(VALID_PROFILES)}"
        )

    if interaction not in VALID_INTERACTIONS:
        raise RuntimeError(
            f"Invalid interaction '{interaction}'. "
            f"Supported values: {', '.join(VALID_INTERACTIONS)}"
        )

    if deployment not in VALID_DEPLOYMENTS:
        raise RuntimeError(
            f"Invalid deployment '{deployment}'. "
            f"Supported values: {', '.join(VALID_DEPLOYMENTS)}"
        )

    if host not in VALID_HOSTS:
        raise RuntimeError(
            f"Invalid host '{host}'. "
            f"Supported values: {', '.join(VALID_HOSTS)}"
        )

    resolved_ros_distro = _resolve_ros_distro(ros_distro)

    # ---------------------------------------------------------
    # Navigation profile
    # ---------------------------------------------------------

    camera_obstacles = profile in ("B", "D")
    social_navigation = profile in ("C", "D")

    # C and D always provide social explanation and speech.
    explanation = social_navigation
    speech_required = social_navigation

    # Dialogue is an additional interaction capability.
    dialogue = interaction == "dialogue"

    # ---------------------------------------------------------
    # Cross-configuration validation
    # ---------------------------------------------------------

    if dialogue and not social_navigation:
        raise RuntimeError(
            "interaction='dialogue' currently requires profile C or D. "
            "Profiles A and B are non-social navigation profiles."
        )

    role_hosts = _resolve_role_hosts(deployment)

    # ROS4HRI dialogue is Humble-only.
    #
    # The important distinction is the host that OWNS dialogue,
    # not necessarily the ROS distribution of every machine.
    if dialogue:
        dialogue_host = role_hosts["dialogue"]

        if dialogue_host == host and resolved_ros_distro != "humble":
            raise RuntimeError(
                "The local host owns the dialogue role, but "
                f"ROS_DISTRO='{resolved_ros_distro}'. "
                "ROS4HRI dialogue requires Humble on the dialogue host."
            )

    return SystemPlan(
        profile=profile,
        interaction=interaction,
        deployment=deployment,
        host=host,
        ros_distro=resolved_ros_distro,
        camera_obstacles=camera_obstacles,
        social_navigation=social_navigation,
        explanation=explanation,
        speech_required=speech_required,
        dialogue=dialogue,
        role_hosts=role_hosts,
    )