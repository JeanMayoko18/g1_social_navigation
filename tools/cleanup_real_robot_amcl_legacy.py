#!/usr/bin/env python3
"""
Step 10D - Remove legacy social ownership from
real_robot_amcl_localization.launch.py.

Architecture after migration
----------------------------

g1_social_bringup/system.launch.py
    |
    +-- owns social orchestration
    +-- owns explainability
    +-- owns speech
    +-- owns social visualization
    |
    +-- navigation.launch.py
            |
            +-- owns NavigationGoalGate policy and lifecycle
            |
            +-- g1_nav2 launches
                    |
                    +-- own Nav2 / localization
                    +-- do NOT launch the social system

This migration intentionally preserves:
- AMCL;
- map server;
- Nav2;
- perception_profile;
- locomotion calibration;
- twist_mux;
- cmd_vel QoS adapter;
- standard Nav2 RViz;
- existing bridge/localization behavior.

This migration removes:
- social_system.launch.py ownership;
- social_rviz.launch.py ownership;
- social deployment arguments;
- explainability arguments;
- Piper arguments;
- delayed social startup actions.

Hardware ownership is not changed.
rl_sar remains external.
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


WORKSPACE = Path.home() / "g1_ws"

TARGET = (
    WORKSPACE
    / "src"
    / "g1_nav2"
    / "launch"
    / "real_robot_amcl_localization.launch.py"
)


# ============================================================================
# Migration contract
# ============================================================================

LEGACY_ASSIGNMENTS = {
    # Old deployment / explainability configuration.
    "runtime_host",
    "pc_ip",
    "enable_explainability",
    "enable_tts",
    "use_social_rviz",

    # Old social-system activation.
    "start_social_system",

    # Old Piper configuration.
    "piper_backend",
    "piper_model_directory",
    "piper_executable",
    "piper_spanish_model",
    "piper_english_model",

    # Old package/path ownership.
    "social_perception_share",
    "g1_social_bringup_share",
    "social_rviz_launch",

    # Old social launch actions.
    "social_system_condition",
    "social_bringup",
    "social_rviz",

    # Old delayed startup actions.
    "social_bringup_delayed",
    "social_rviz_delayed",
}


LEGACY_LAUNCH_ARGUMENTS = {
    "runtime_host",
    "pc_ip",
    "enable_explainability",
    "enable_tts",
    "use_social_rviz",
    "start_social_system",
    "piper_backend",
    "piper_model_directory",
    "piper_executable",
    "piper_spanish_model",
    "piper_english_model",
}


REQUIRED_TOKENS = {
    # Localization.
    "map_server_node",
    "amcl",
    "lifecycle_manager_localization_node",

    # Navigation.
    "nav2",
    "perception_profile",

    # Command pipeline.
    "locomotion_calibration",
    "twist_mux_node",
    "qos_adapter",

    # Standard Nav2 visualization.
    "rviz",
}


FORBIDDEN_ACTIVE_TOKENS = {
    "social_system.launch.py",
    "social_bringup_delayed",
    "social_rviz_delayed",
    "social_rviz_launch",
    "start_social_system",
    "enable_explainability",
    "enable_tts",
    "use_social_rviz",
    "piper_backend",
    "piper_model_directory",
    "piper_executable",
    "piper_spanish_model",
    "piper_english_model",
    "g1_social_bringup_share",
    "social_perception_share",
}


# ============================================================================
# Generic helpers
# ============================================================================

def require(condition: bool, message: str) -> None:
    """Abort the migration when an architectural invariant fails."""
    if not condition:
        raise RuntimeError(message)


def node_range(node: ast.AST) -> tuple[int, int]:
    """
    Return a zero-based source range.

    The end index is exclusive.
    """
    require(
        hasattr(node, "lineno")
        and hasattr(node, "end_lineno"),
        "AST source positions are unavailable.",
    )

    return (
        int(node.lineno) - 1,
        int(node.end_lineno),
    )


def assignment_name(node: ast.AST) -> str | None:
    """Return a simple variable assigned by a statement."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None

        target = node.targets[0]

        if isinstance(target, ast.Name):
            return target.id

    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return node.target.id

    return None


def declared_launch_argument(node: ast.AST) -> str | None:
    """
    Detect:

        DeclareLaunchArgument("argument_name", ...)
    """
    if not isinstance(node, ast.Call):
        return None

    if not isinstance(node.func, ast.Name):
        return None

    if node.func.id != "DeclareLaunchArgument":
        return None

    if not node.args:
        return None

    first_argument = node.args[0]

    if (
        isinstance(first_argument, ast.Constant)
        and isinstance(first_argument.value, str)
    ):
        return first_argument.value

    return None


def remove_ranges(
    text: str,
    ranges: list[tuple[int, int]],
) -> str:
    """
    Remove complete source ranges.

    Reverse ordering prevents earlier source positions from changing while
    later ranges are removed.
    """
    lines = text.splitlines(keepends=True)

    for start, end in sorted(
        ranges,
        key=lambda item: item[0],
        reverse=True,
    ):
        del lines[start:end]

    return "".join(lines)


# ============================================================================
# Structural cleanup
# ============================================================================

def remove_legacy_assignments(text: str) -> str:
    """
    Remove obsolete assignments from generate_launch_description().

    Only explicitly listed variable names are removed.
    """
    tree = ast.parse(text)

    generate_function = None

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "generate_launch_description"
        ):
            generate_function = node
            break

    require(
        generate_function is not None,
        "generate_launch_description() was not found.",
    )

    ranges: list[tuple[int, int]] = []
    found: set[str] = set()

    for statement in generate_function.body:
        name = assignment_name(statement)

        if name in LEGACY_ASSIGNMENTS:
            ranges.append(node_range(statement))
            found.add(name)

    print(
        "[FOUND] Legacy assignments: "
        + (
            ", ".join(sorted(found))
            if found
            else "none"
        )
    )

    return remove_ranges(
        text,
        ranges,
    )


def remove_legacy_launch_arguments(text: str) -> str:
    """Remove old social-system arguments from LaunchDescription."""
    tree = ast.parse(text)

    lines = text.splitlines(keepends=True)

    ranges: list[tuple[int, int]] = []
    found: set[str] = set()

    for node in ast.walk(tree):
        name = declared_launch_argument(node)

        if name not in LEGACY_LAUNCH_ARGUMENTS:
            continue

        start, end = node_range(node)

        # Some formatting styles place the list comma outside the AST call.
        if end < len(lines):
            if lines[end].strip() == ",":
                end += 1

        ranges.append(
            (start, end)
        )

        found.add(name)

    print(
        "[FOUND] Legacy launch arguments: "
        + (
            ", ".join(sorted(found))
            if found
            else "none"
        )
    )

    return remove_ranges(
        text,
        ranges,
    )


def remove_stale_action_entries(text: str) -> str:
    """
    Remove deleted social actions from the final LaunchDescription list.

    Standard Nav2 RViz is intentionally NOT removed.
    """
    stale_actions = (
        "social_bringup",
        "social_rviz",
        "social_bringup_delayed",
        "social_rviz_delayed",
    )

    for action_name in stale_actions:
        text = re.sub(
            rf"^[ \t]*{re.escape(action_name)},[ \t]*\n",
            "",
            text,
            flags=re.MULTILINE,
        )

    return text


def remove_legacy_social_comment_section(text: str) -> str:
    """
    Remove the obsolete social-system section header/comments.

    The executable statements inside the section are already removed
    structurally by remove_legacy_assignments().
    """
    start_marker = (
        "    # -------------------------------------------------------------------------\n"
        "    # SOCIAL PERCEPTION / EXPLAINABILITY\n"
    )

    end_marker = (
        "    locomotion_calibration_ = Node("
    )

    start = text.find(start_marker)

    if start == -1:
        print(
            "[INFO] Legacy social comment section "
            "already absent."
        )
        return text

    end = text.find(
        end_marker,
        start,
    )

    require(
        end != -1,
        (
            "Legacy social section was found, but the following "
            "locomotion_calibration_ section was not found."
        ),
    )

    section = text[start:end]

    # This section must not contain anything that still belongs to Nav2.
    for required in (
        "map_server_node",
        "amcl =",
        "nav2 =",
        "locomotion_calibration =",
    ):
        require(
            required not in section,
            (
                "Refusing to remove the legacy social section because "
                f"it unexpectedly contains {required!r}."
            ),
        )

    print(
        "[REMOVE] Legacy social-system section comments"
    )

    return (
        text[:start]
        + text[end:]
    )


def normalize_spacing(text: str) -> str:
    """Reduce blank-line accumulation without reformatting the source."""
    return re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )


# ============================================================================
# Validation
# ============================================================================

def validate_python(text: str) -> None:
    """Validate Python syntax without executing the launch."""
    ast.parse(text)


def validate_required_architecture(text: str) -> None:
    """
    Verify that localization/navigation functionality survived the cleanup.
    """
    missing = [
        token
        for token in sorted(REQUIRED_TOKENS)
        if token not in text
    ]

    require(
        not missing,
        (
            "Required AMCL/Nav2 components were accidentally removed: "
            + ", ".join(missing)
        ),
    )

    # Profile A/B/C/D remains a Nav2 costmap configuration contract.
    require(
        '"perception_profile"' in text,
        "perception_profile launch argument was removed.",
    )

    require(
        "perception_profiles" in text,
        "Nav2 perception-profile YAML loading was removed.",
    )

    # Standard Nav2 RViz remains valid.
    require(
        '"rviz_launch.py"' in text,
        "Standard Nav2 RViz launch was removed.",
    )


def validate_no_legacy_social_ownership(text: str) -> None:
    """Reject old social ownership remaining in this launch."""
    remaining = [
        token
        for token in sorted(FORBIDDEN_ACTIVE_TOKENS)
        if token in text
    ]

    require(
        not remaining,
        (
            "Legacy social ownership remains: "
            + ", ".join(remaining)
        ),
    )


def validate_hardware_boundary(text: str) -> None:
    """
    rl_sar is the external hardware owner.

    Topic names such as /cmd_vel_rlsar are allowed. What is forbidden here is
    launching the rl_sar package itself.
    """
    require(
        'package="rl_sar"' not in text
        and "package='rl_sar'" not in text,
        (
            "real_robot_amcl_localization.launch.py unexpectedly "
            "launches rl_sar."
        ),
    )


def validate_no_gate_invention(text: str) -> None:
    """
    Step 10D is cleanup only.

    Do not silently invent a second NavigationGoalGate contract inside this
    legacy localization launch.
    """
    require(
        "navigation_goal_gate_node" not in text,
        (
            "Unexpected NavigationGoalGate node ownership detected "
            "inside the AMCL launch."
        ),
    )


# ============================================================================
# Main migration
# ============================================================================

def main() -> None:
    require(
        TARGET.is_file(),
        f"Target file does not exist: {TARGET}",
    )

    original = TARGET.read_text(
        encoding="utf-8",
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = TARGET.with_name(
        TARGET.name
        + f".before_step10d_{timestamp}.bak"
    )

    shutil.copy2(
        TARGET,
        backup,
    )

    print(
        f"[BACKUP] {backup}"
    )

    text = original

    # ------------------------------------------------------------------
    # Remove legacy social orchestration.
    # ------------------------------------------------------------------

    text = remove_legacy_assignments(text)

    text = remove_legacy_launch_arguments(text)

    text = remove_stale_action_entries(text)

    text = remove_legacy_social_comment_section(text)

    text = normalize_spacing(text)

    # ------------------------------------------------------------------
    # Static architecture validation.
    # ------------------------------------------------------------------

    validate_python(text)

    validate_required_architecture(text)

    validate_no_legacy_social_ownership(text)

    validate_hardware_boundary(text)

    validate_no_gate_invention(text)

    # ------------------------------------------------------------------
    # Compile a temporary copy before replacing the source file.
    # ------------------------------------------------------------------

    temporary = TARGET.with_name(
        TARGET.name + ".step10d.tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    try:
        py_compile.compile(
            str(temporary),
            doraise=True,
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        print()
        print(
            "[FAILED] Generated AMCL launch is not valid Python."
        )
        print(
            f"[BACKUP] Original preserved at: {backup}"
        )

        raise

    # ------------------------------------------------------------------
    # Install only after all validation passed.
    # ------------------------------------------------------------------

    TARGET.write_text(
        text,
        encoding="utf-8",
    )

    temporary.unlink(
        missing_ok=True
    )

    py_compile.compile(
        str(TARGET),
        doraise=True,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STEP 10D - REAL ROBOT AMCL CLEANUP"
    )
    print(
        "============================================================"
    )
    print(
        "[PASS] Python syntax"
    )
    print(
        "[PASS] AMCL ownership preserved"
    )
    print(
        "[PASS] Map server ownership preserved"
    )
    print(
        "[PASS] Nav2 ownership preserved"
    )
    print(
        "[PASS] perception_profile preserved"
    )
    print(
        "[PASS] Locomotion calibration preserved"
    )
    print(
        "[PASS] twist_mux / QoS adapter preserved"
    )
    print(
        "[PASS] Standard Nav2 RViz preserved"
    )
    print(
        "[PASS] Legacy social-system ownership removed"
    )
    print(
        "[PASS] Legacy social RViz ownership removed"
    )
    print(
        "[PASS] Legacy explainability / Piper ownership removed"
    )
    print(
        "[PASS] No NavigationGoalGate ownership added"
    )
    print(
        "[PASS] rl_sar remains external"
    )
    print(
        "============================================================"
    )
    print(
        f"[UPDATED] {TARGET}"
    )
    print(
        f"[BACKUP]  {backup}"
    )


if __name__ == "__main__":
    main()
