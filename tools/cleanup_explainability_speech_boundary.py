#!/usr/bin/env python3
"""
Step 10G - Final Explainability / Speech ownership separation.

Final architecture
------------------

system.launch.py
    |
    +-- explainability.launch.py
    |       |
    |       +-- explanation_node
    |           +-- explanation generation
    |           +-- LLM endpoint/model configuration
    |           +-- explanation cache
    |
    +-- speech.launch.py
    |       |
    |       +-- PiperTTSNode
    |           +-- single speech queue
    |           +-- Piper synthesis
    |           +-- audio backend
    |
    +-- navigation.launch.py
            |
            +-- NavigationGoalGate
            +-- navigation interaction announcements

Explainability must not own:
- Piper;
- TTS enablement;
- speech language;
- audio playback;
- Unitree audio;
- navigation announcements;
- NavigationGoalGate.

The Piper implementation remains installed in g1_robot_explainability because
speech.launch.py uses its piper_tts executable. This migration only removes
speech ownership from the Explainability launch API.
"""

from __future__ import annotations

import ast
import py_compile
import shutil
from datetime import datetime
from pathlib import Path


WS = Path.home() / "g1_ws"

EXPLAINABILITY_LAUNCH = (
    WS
    / "src"
    / "g1_robot_explainability"
    / "launch"
    / "explainability.launch.py"
)

SYSTEM_LAUNCH = (
    WS
    / "src"
    / "g1_social_bringup"
    / "launch"
    / "system.launch.py"
)

SPEECH_LAUNCH = (
    WS
    / "src"
    / "g1_social_bringup"
    / "launch"
    / "speech.launch.py"
)

PIPER_SOURCE = (
    WS
    / "src"
    / "g1_robot_explainability"
    / "g1_robot_explainability"
    / "tts_node.py"
)

BACKUP_ROOT = (
    WS
    / "tools"
    / "step10g_backups"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


NEW_EXPLAINABILITY_LAUNCH = r'''"""
Launch the G1 explanation generation subsystem.

Flow
----

Social/navigation context
    -> explanation_node
        -> explanation text
            -> Common Speech API

Responsibilities
----------------

This launch owns only explanation generation:
- explanation configuration;
- local or remote LLM endpoint selection;
- local model path selection;
- explanation cache;
- private LLM environment variables;
- explanation_node lifecycle.

This launch does NOT own:
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
    config_file,
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
        local_base_url = requested_local_base_url

    # A robot-hosted explanation role may use an LLM server on the PC.
    elif (
        runtime_host == "robot"
        and pc_ip
    ):
        local_base_url = (
            f"http://{pc_ip}:8080/v1"
        )

        # The model is remote from this process.
        local_model_path = ""

    else:
        local_base_url = (
            "http://127.0.0.1:8080/v1"
        )

    print("=" * 70)
    print("G1 EXPLAINABILITY DEPLOYMENT")
    print("=" * 70)
    print(f"Runtime host     : {runtime_host}")
    print(f"ROS distribution : {ros_distro}")
    print(f"LLM endpoint     : {local_base_url}")
    print(
        "Local model path : "
        f"{local_model_path or 'remote model'}"
    )
    print(f"Cache file       : {cache_file}")
    print("=" * 70)

    if not enable_explanation_node:
        return []

    explanation_node = Node(
        package="g1_robot_explainability",
        executable="explanation_node",
        name="explanation_node",
        output="screen",
        parameters=[
            config_file,
            {
                "local_base_url": local_base_url,
                "local_model_path": local_model_path,
                "cache_file": cache_file,
            },
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
        package_share,
        "config",
        "explainability.yaml",
    )

    config_file = LaunchConfiguration(
        "config_file"
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
            "config_file",
            default_value=default_config,
            description=(
                "Path to the G1 explainability configuration file."
            ),
        ),
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
            "config_file": config_file,
            "private_env": private_env,
        },
    )

    return LaunchDescription(
        launch_arguments
        + [
            runtime_nodes,
        ]
    )
'''


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def backup_file(path: Path) -> None:
    require(
        path.is_file(),
        f"Missing required file: {path}",
    )

    destination = (
        BACKUP_ROOT
        / path.relative_to(WS)
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        path,
        destination,
    )

    print(
        f"[BACKUP] {path.relative_to(WS)}"
    )


def compile_python(path: Path) -> None:
    py_compile.compile(
        str(path),
        doraise=True,
    )


def validate_new_explainability(text: str) -> None:
    ast.parse(text)

    required = (
        'executable="explanation_node"',
        '"runtime_host"',
        '"ros_distro"',
        '"pc_ip"',
        '"local_base_url"',
        '"local_model_path"',
        '"explanation_cache_file"',
        '"enable_explanation_node"',
        '"cache_file": cache_file',
    )

    for token in required:
        require(
            token in text,
            f"Required Explainability contract missing: {token}",
        )

    forbidden = (
        "piper_",
        "enable_tts",
        "audio_backend",
        "audio_player",
        "unitree_volume",
        "unitree_network_interface",
        "enable_goal_speech",
        "enable_navigation_interaction_speech",
        "assume_idle_on_start",
        "navigation_goal_gate",
        "resolve_audio_backend",
        "tts_path_model_resolver",
    )

    remaining = [
        token
        for token in forbidden
        if token in text
    ]

    require(
        not remaining,
        (
            "Speech/navigation ownership remains in Explainability: "
            + ", ".join(remaining)
        ),
    )


def patch_system_launch(text: str) -> str:
    """
    Remove transitional arguments that no longer belong to
    explainability.launch.py.
    """

    old = '''                    "runtime_host": plan.host,
                    "ros_distro": plan.ros_distro,
                    "enable_explanation_node": "true",

                    # Transitional duplicate-announcement policy.
                    # Explainability does not own the gate.
                    "enable_navigation_goal_gate": (
                        "true"
                        if plan.social_navigation
                        else "false"
                    ),

                    # Navigation speech ownership moved to navigation.launch.py.
                    "enable_navigation_interaction_speech": "false",

                    # Speech engine ownership moved to speech.launch.py.
                    "enable_tts": "false",
'''

    new = '''                    "runtime_host": plan.host,
                    "ros_distro": plan.ros_distro,
                    "enable_explanation_node": "true",
'''

    require(
        old in text,
        (
            "Expected transitional Explainability argument block was "
            "not found in system.launch.py. Refusing an approximate patch."
        ),
    )

    text = text.replace(
        old,
        new,
        1,
    )

    # Remove the obsolete transitional comment immediately above the role.
    text = text.replace(
        '''    # The current legacy explainability launch still contains
    # speech-related processes. This include is transitional.
    # Speech ownership will be separated in a later step.

''',
        '''    # Explanation role: LLM/context/explanation generation only.
    # Speech and navigation execution are owned by separate launch roles.

''',
        1,
    )

    return text


def validate_system(text: str) -> None:
    ast.parse(text)

    # The whole system may legitimately mention these tokens for the
    # Navigation/Speech roles. Therefore validate only the Explainability
    # include region.
    marker = '"explainability.launch.py"'

    index = text.find(marker)

    require(
        index >= 0,
        "Explainability include was not found in system.launch.py.",
    )

    region = text[
        max(0, index - 1200):
        min(len(text), index + 2600)
    ]

    forbidden = (
        '"enable_navigation_goal_gate"',
        '"enable_navigation_interaction_speech"',
        '"enable_tts"',
    )

    remaining = [
        token
        for token in forbidden
        if token in region
    ]

    require(
        not remaining,
        (
            "Transitional arguments remain in the Explainability include: "
            + ", ".join(remaining)
        ),
    )

    require(
        '"enable_explanation_node": "true"' in region,
        "Explanation role activation was accidentally removed.",
    )


def validate_speech_owner() -> None:
    require(
        SPEECH_LAUNCH.is_file(),
        "speech.launch.py is missing.",
    )

    require(
        PIPER_SOURCE.is_file(),
        "Piper implementation tts_node.py is missing.",
    )

    speech_text = SPEECH_LAUNCH.read_text(
        encoding="utf-8",
    )

    require(
        'executable="piper_tts"' in speech_text,
        "speech.launch.py no longer owns Piper.",
    )

    require(
        '"audio_backend"' in speech_text,
        "speech.launch.py no longer exposes audio_backend.",
    )


def main() -> None:
    for path in (
        EXPLAINABILITY_LAUNCH,
        SYSTEM_LAUNCH,
        SPEECH_LAUNCH,
        PIPER_SOURCE,
    ):
        require(
            path.is_file(),
            f"Required file missing: {path}",
        )

    validate_speech_owner()

    # ---------------------------------------------------------------
    # Back up both modified files.
    # ---------------------------------------------------------------

    backup_file(
        EXPLAINABILITY_LAUNCH
    )

    backup_file(
        SYSTEM_LAUNCH
    )

    # ---------------------------------------------------------------
    # Build both new files in memory.
    # ---------------------------------------------------------------

    explainability_text = (
        NEW_EXPLAINABILITY_LAUNCH
    )

    validate_new_explainability(
        explainability_text
    )

    system_text = SYSTEM_LAUNCH.read_text(
        encoding="utf-8",
    )

    system_text = patch_system_launch(
        system_text
    )

    validate_system(
        system_text
    )

    # ---------------------------------------------------------------
    # Validate temporary files before installation.
    # ---------------------------------------------------------------

    explainability_tmp = (
        EXPLAINABILITY_LAUNCH.with_name(
            "explainability.launch.py.step10g.tmp"
        )
    )

    system_tmp = SYSTEM_LAUNCH.with_name(
        "system.launch.py.step10g.tmp"
    )

    explainability_tmp.write_text(
        explainability_text,
        encoding="utf-8",
    )

    system_tmp.write_text(
        system_text,
        encoding="utf-8",
    )

    try:
        compile_python(
            explainability_tmp
        )

        compile_python(
            system_tmp
        )

    except Exception:
        explainability_tmp.unlink(
            missing_ok=True
        )

        system_tmp.unlink(
            missing_ok=True
        )

        print()
        print(
            "[FAILED] Generated Python did not compile."
        )
        print(
            f"[BACKUPS] {BACKUP_ROOT}"
        )

        raise

    # ---------------------------------------------------------------
    # Install.
    # ---------------------------------------------------------------

    EXPLAINABILITY_LAUNCH.write_text(
        explainability_text,
        encoding="utf-8",
    )

    SYSTEM_LAUNCH.write_text(
        system_text,
        encoding="utf-8",
    )

    explainability_tmp.unlink(
        missing_ok=True
    )

    system_tmp.unlink(
        missing_ok=True
    )

    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------

    compile_python(
        EXPLAINABILITY_LAUNCH
    )

    compile_python(
        SYSTEM_LAUNCH
    )

    validate_new_explainability(
        EXPLAINABILITY_LAUNCH.read_text(
            encoding="utf-8"
        )
    )

    validate_system(
        SYSTEM_LAUNCH.read_text(
            encoding="utf-8"
        )
    )

    validate_speech_owner()

    print()
    print("=" * 68)
    print("STEP 10G - EXPLAINABILITY / SPEECH BOUNDARY CLEAN")
    print("=" * 68)
    print("[PASS] Explainability owns only explanation generation")
    print("[PASS] Piper configuration removed from Explainability")
    print("[PASS] Audio configuration removed from Explainability")
    print("[PASS] Navigation speech configuration removed")
    print("[PASS] NavigationGoalGate configuration removed")
    print("[PASS] system.launch.py caller contract updated")
    print("[PASS] speech.launch.py remains Piper owner")
    print("[PASS] tts_node.py preserved")
    print("[PASS] Python syntax")
    print("=" * 68)
    print(f"[BACKUPS] {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
