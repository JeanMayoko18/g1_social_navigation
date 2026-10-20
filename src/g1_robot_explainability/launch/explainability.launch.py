"""
Launch the G1 explainability subsystem.

Flow
----
SocialNavigationDecision
    -> explanation_trigger
        -> explanation request
            -> explanation_context
                -> ExplanationContext
                    -> explanation_node
                        -> ExplanationText
                            -> Common Speech API

Responsibilities
----------------
This launch owns:
- explanation triggering;
- provider-independent explanation context construction;
- explanation generation;
- local or remote LLM endpoint selection;
- explanation cache;
- private LLM environment variables.

This launch does NOT own:
- social navigation decision making;
- Piper or any TTS engine;
- audio playback;
- speech queues;
- navigation announcements;
- NavigationGoalGate.

Speech is owned by g1_social_bringup/speech.launch.py.
Navigation execution is owned by g1_social_bringup/navigation.launch.py.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


ETH0_ENV = {
    **os.environ,
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    "CYCLONEDDS_URI": "/home/unitree/cyclonedds_ws/cyclonedds.xml",
}


def parse_boolean(
    value: str,
    *,
    label: str,
) -> bool:
    """Parse a strict launch boolean."""
    normalized = value.strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "off",
    }:
        return False

    raise RuntimeError(
        f"{label} must be true or false, received {value!r}."
    )


def resolve_ros_distro(context) -> str:
    """Resolve the supported ROS distribution."""
    requested = (
        LaunchConfiguration("ros_distro")
        .perform(context)
        .strip()
        .lower()
    )

    if requested in ("", "auto"):
        requested = (
            os.environ.get("ROS_DISTRO", "")
            .strip()
            .lower()
        )

    if requested not in ("foxy", "humble"):
        raise RuntimeError(
            "ros_distro must resolve to foxy or humble, "
            f"received {requested!r}."
        )

    return requested


def resolve_optional_path(
    requested_value: str,
    *,
    default_path: Path,
) -> str:
    """Resolve an optional path where 'auto' selects the default."""
    normalized = requested_value.strip()

    if (
        not normalized
        or normalized.lower() == "auto"
    ):
        return str(
            default_path.expanduser()
        )

    return str(
        Path(normalized)
        .expanduser()
    )


def load_env_file(env_path: str) -> dict:
    """Load KEY=VALUE pairs from a private environment file."""
    env_values = {}

    path = Path(env_path)

    if not path.is_file():
        return env_values

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        key = key.strip()
        value = value.strip()

        if key:
            env_values[key] = value

    return env_values


def create_runtime_nodes(
    context,
    *,
    explanation_config,
    private_env,
):
    """
    Create the local explanation generator.

    Deployment decides where this launch is included. This function only
    resolves host-specific LLM connectivity and explanation storage.
    """

    ros_distro = resolve_ros_distro(context)

    requested_runtime_host = (
        LaunchConfiguration("runtime_host")
        .perform(context)
        .strip()
        .lower()
    )

    if requested_runtime_host in {
        "",
        "auto",
    }:
        runtime_host = (
            "robot"
            if ros_distro == "foxy"
            else "pc"
        )

    elif requested_runtime_host in {
        "pc",
        "robot",
    }:
        runtime_host = requested_runtime_host

    else:
        raise RuntimeError(
            "runtime_host must be auto, pc or robot, "
            f"received {requested_runtime_host!r}."
        )

    language = (
        LaunchConfiguration("language")
        .perform(context)
        .strip()
        .lower()
    )

    if language not in ("en", "es", "fr"):
        raise RuntimeError(
            f"Unsupported language='{language}'. Expected 'en', 'es' or 'fr'."
        )
    
    enable_explanation_node = parse_boolean(
        LaunchConfiguration(
            "enable_explanation_node"
        ).perform(context),
        label="enable_explanation_node",
    )

    pc_ip = (
        LaunchConfiguration("pc_ip")
        .perform(context)
        .strip()
    )

    requested_local_base_url = (
        LaunchConfiguration("local_base_url")
        .perform(context)
        .strip()
    )

    requested_model_path = (
        LaunchConfiguration("local_model_path")
        .perform(context)
    )

    requested_cache_file = (
        LaunchConfiguration(
            "explanation_cache_file"
        )
        .perform(context)
    )

    # Use Unitree Ethernet DDS only when the explanation role runs
    # directly on the robot.
    runtime_environment = (
        ETH0_ENV
        if runtime_host == "robot"
        else {}
    )

    user_home = Path.home()

    if runtime_host == "pc":
        default_model_path = (
            user_home
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Qwen--Qwen3-4B-GGUF"
            / "snapshots"
            / "bc640142c66e1fdd12af0bd68f40445458f3869b"
            / "Qwen3-4B-Q4_K_M.gguf"
        )

    else:
        default_model_path = (
            user_home
            / "models"
            / "Qwen3-4B-Q4_K_M.gguf"
        )

    default_cache_file = (
        user_home
        / ".cache"
        / "g1_robot_explainability"
        / "explanations.json"
    )

    local_model_path = resolve_optional_path(
        requested_model_path,
        default_path=default_model_path,
    )

    cache_file = resolve_optional_path(
        requested_cache_file,
        default_path=default_cache_file,
    )

    # Explicit endpoint always wins.
    if (
        requested_local_base_url
        and requested_local_base_url.lower() != "auto"
    ):
        local_base_url_override = requested_local_base_url

    # A robot-hosted explanation role may explicitly use an LLM server
    # running on the PC. Open WebUI exposes its OpenAI-compatible API
    # below /api.
    elif (
        runtime_host == "robot"
        and pc_ip
    ):
        local_base_url_override = (
            f"http://{pc_ip}:3000/api"
        )

        # The model is remote from this process.
        local_model_path = ""

    else:
        # No launch-level override.
        #
        # The explanation YAML remains authoritative. This is important
        # for local backends such as Open WebUI because the backend and
        # endpoint belong to provider configuration, not deployment.
        local_base_url_override = ""

    print("=" * 70)
    print("G1 EXPLAINABILITY DEPLOYMENT")
    print("=" * 70)
    print(f"Runtime host     : {runtime_host}")
    print(f"ROS distribution : {ros_distro}")
    print(
        "LLM endpoint     : "
        f"{local_base_url_override or 'from config'}"
    )
    print(
        "Local model path : "
        f"{local_model_path or 'remote model'}"
    )
    print(f"Cache file       : {cache_file}")
    print("=" * 70)

    if not enable_explanation_node:
        return []

    explanation_trigger = Node(
        package="g1_social_perception",
        executable="explanation_trigger",
        name="explanation_trigger",
        output="screen",
        additional_env={
            **runtime_environment,
        },
    )

    explanation_context = Node(
        package="g1_social_perception",
        executable="explanation_context",
        name="explanation_context",
        output="screen",
        parameters=[
            {
                "language": language,
                "fallback_language": language,
            },
        ],
        additional_env={
            **runtime_environment,
        },
    )

    explanation_parameters = {
        "cache_file": cache_file,
    }

    if local_base_url_override:
        explanation_parameters["local_base_url"] = (
            local_base_url_override
        )

    explanation_node = Node(
        package="g1_robot_explainability",
        executable="explanation_node",
        name="explanation_node",
        output="screen",
        parameters=[
            explanation_config,
            explanation_parameters,
        ],
        additional_env={
            **private_env,
            **runtime_environment,
        },
    )

    return [
        explanation_node,
    ]


def generate_launch_description():
    """Create the explanation-only launch description."""

    package_share = get_package_share_directory(
        "g1_robot_explainability"
    )

    default_config = os.path.join(
        get_package_share_directory("g1_social_bringup"),
        "config",
        "social_system.yaml",
    )

    explanation_config = LaunchConfiguration(
        "explanation_config"
    )

    default_env_file = os.environ.get(
        "G1_EXPLAINABILITY_ENV_FILE",
        os.path.expanduser(
            "~/g1_ws/src/g1_robot_explainability/.env"
        ),
    )

    private_env = load_env_file(
        default_env_file
    )

    default_local_model_path = os.environ.get(
        "G1_LOCAL_LLM_MODEL_PATH",
        "",
    )

    default_explanation_cache_file = os.environ.get(
        "G1_EXPLANATION_CACHE_FILE",
        os.path.join(
            str(Path.home()),
            ".cache",
            "g1_robot_explainability",
            "explanations.json",
        ),
    )

    default_pc_ip = os.environ.get(
        "G1_EXPLAINABILITY_PC_IP",
        "",
    )

    launch_arguments = [
        DeclareLaunchArgument(
            "runtime_host",
            default_value="auto",
            description=(
                "Explanation host: auto, pc or robot. "
                "auto selects robot on Foxy and pc on Humble."
            ),
        ),
        DeclareLaunchArgument(
            "ros_distro",
            default_value="auto",
            description=(
                "ROS distribution: auto, foxy or humble."
            ),
        ),

        DeclareLaunchArgument(
            "language",
            default_value="en",
            choices=["en", "es", "fr"],
            description=(
                "Language used for generated social explanations."
            ),
        ),

        DeclareLaunchArgument(
            "pc_ip",
            default_value=default_pc_ip,
            description=(
                "PC address hosting the LLM server when the explanation "
                "role runs on the robot."
            ),
        ),
        DeclareLaunchArgument(
            "local_base_url",
            default_value="auto",
            description=(
                "Explicit OpenAI-compatible LLM endpoint. "
                "auto derives it from runtime_host and pc_ip."
            ),
        ),
        DeclareLaunchArgument(
            "local_model_path",
            default_value=(
                default_local_model_path
                or "auto"
            ),
            description=(
                "Local GGUF model path or auto."
            ),
        ),
        DeclareLaunchArgument(
            "explanation_cache_file",
            default_value=default_explanation_cache_file,
            description=(
                "Semantic explanation cache path or auto."
            ),
        ),

        DeclareLaunchArgument(
            "explanation_config",
            default_value=default_config,
            description=(
                "Path to the central G1 social-system configuration file."
            ),
        ),
        DeclareLaunchArgument(
            "enable_explanation_node",
            default_value="true",
            choices=["true", "false"],
            description=(
                "Start the explanation generator on this machine."
            ),
        ),
    ]

    runtime_nodes = OpaqueFunction(
        function=create_runtime_nodes,
        kwargs={
            "explanation_config": explanation_config,
            "private_env": private_env,
        },
    )

    return LaunchDescription(
        launch_arguments
        + [
            runtime_nodes,
        ]
    )
