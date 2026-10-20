#!/usr/bin/env python3
"""
Step 10 - Remove legacy social ownership from g1_bringup.launch.py.

Target architecture
-------------------

g1_social_bringup/system.launch.py
    |
    +-- owns profile / deployment / interaction policy
    +-- owns social perception
    +-- owns social decision
    +-- owns explainability
    +-- owns speech
    +-- owns visualization
    |
    +-- navigation.launch.py
            |
            +-- owns NavigationGoalGate
            +-- resolves whether the gate is enabled
            |
            +-- g1_nav2/g1_bringup.launch.py
                    |
                    +-- owns Nav2 / SLAM
                    +-- owns Nav2 action remapping
                    +-- does NOT own social components

Important invariants
--------------------

1. perception_profile remains in g1_nav2 because it selects the Nav2
   costmap configuration A/B/C/D.

2. enable_navigation_goal_gate remains in g1_nav2 because the Nav2
   launch helpers use it to atomically remap NavigateToPose.

3. NavigationGoalGate itself is NOT launched here.

4. Social perception, HRI, explainability, speech and social RViz are
   NOT launched here.

5. rl_sar remains completely external.

Safety
------

The script:
- creates a timestamped backup;
- works structurally on Python source;
- validates Python syntax before replacing the target;
- checks required Nav2 contracts after migration;
- aborts if unexpected legacy ownership remains.
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
    / "g1_bringup.launch.py"
)


# ============================================================================
# Migration configuration
# ============================================================================

# Variable assignments owned exclusively by the old social orchestration.
LEGACY_ASSIGNMENTS = {
    "use_social_rviz",
    "runtime_host",
    "pc_ip",
    "enable_explainability",
    "enable_tts",
    "start_social_system",
    "enable_social_perception",
    "enable_collaboration",
    "enable_hri",
    "enable_speaker_association",
    "enable_explanation_node",
    "enable_navigation_interaction_speech",
    "piper_backend",
    "piper_model_directory",
    "piper_executable",
    "piper_spanish_model",
    "piper_english_model",
    "unitree_volume",
    "assume_idle_on_start",
    "social_perception_share",
    "g1_social_bringup_share",
    "social_rviz_launch",
    "social_bringup",
    "social_rviz",
    "social_system_condition",
    "social_bringup_delayed",
    "social_rviz_delayed",
}


# Launch arguments belonging to the old social-system API.
LEGACY_LAUNCH_ARGUMENTS = {
    "runtime_host",
    "pc_ip",
    "enable_explainability",
    "enable_tts",
    "use_social_rviz",
    "piper_backend",
    "piper_model_directory",
    "piper_executable",
    "piper_spanish_model",
    "piper_english_model",
    "enable_social_perception",
    "start_social_system",
    "enable_explanation_node",
    "language",
    "volume",
    "idle_on_start",
    "enable_navigation_interaction_speech",
    "enable_collaboration",
    "enable_hri",
    "enable_speaker_association",
}


# Names that must remain after the migration.
REQUIRED_NAMES = {
    "perception_profile",
    "enable_navigation_goal_gate",
    "resolved_navigation_goal_gate",
}


# ============================================================================
# Generic helpers
# ============================================================================

def require(condition: bool, message: str) -> None:
    """Raise a clear migration error when an invariant is violated."""
    if not condition:
        raise RuntimeError(message)


def node_range(node: ast.AST) -> tuple[int, int]:
    """
    Return zero-based source-line indexes for a complete AST node.

    The returned end index is exclusive.
    """
    require(
        hasattr(node, "lineno") and hasattr(node, "end_lineno"),
        "Python AST does not provide source positions.",
    )

    start = int(node.lineno) - 1
    end = int(node.end_lineno)

    return start, end


def assignment_name(node: ast.AST) -> str | None:
    """Return the simple variable name assigned by an AST statement."""
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


def call_name(node: ast.AST) -> str | None:
    """Return a simple function name for a call expression."""
    if not isinstance(node, ast.Call):
        return None

    if isinstance(node.func, ast.Name):
        return node.func.id

    return None


def declared_launch_argument(node: ast.AST) -> str | None:
    """
    Return the argument name when the node is:

        DeclareLaunchArgument("name", ...)
    """
    if not isinstance(node, ast.Call):
        return None

    if call_name(node) != "DeclareLaunchArgument":
        return None

    if not node.args:
        return None

    first = node.args[0]

    if (
        isinstance(first, ast.Constant)
        and isinstance(first.value, str)
    ):
        return first.value

    return None


def remove_line_ranges(
    text: str,
    ranges: list[tuple[int, int]],
) -> str:
    """
    Remove source ranges safely.

    Ranges are processed backwards so earlier line numbers remain valid.
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
# Structural migration
# ============================================================================

def remove_legacy_assignments(text: str) -> str:
    """Remove top-level assignments inside generate_launch_description()."""
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

    if found:
        print(
            "[REMOVE] Legacy assignments: "
            + ", ".join(sorted(found))
        )

    return remove_line_ranges(text, ranges)


def remove_legacy_launch_arguments(text: str) -> str:
    """
    Remove obsolete DeclareLaunchArgument expressions.

    They normally live inside the list passed to LaunchDescription().
    """
    tree = ast.parse(text)

    ranges: list[tuple[int, int]] = []
    found: set[str] = set()

    for node in ast.walk(tree):
        argument_name = declared_launch_argument(node)

        if argument_name not in LEGACY_LAUNCH_ARGUMENTS:
            continue

        start, end = node_range(node)

        lines = text.splitlines(keepends=True)

        # Include a standalone trailing comma if formatting placed it
        # outside the AST Call range.
        if end < len(lines):
            if lines[end].strip() == ",":
                end += 1

        ranges.append((start, end))
        found.add(argument_name)

    if found:
        print(
            "[REMOVE] Legacy launch arguments: "
            + ", ".join(sorted(found))
        )

    return remove_line_ranges(text, ranges)


def remove_stale_launchdescription_entries(text: str) -> str:
    """
    Remove references to actions deleted above from LaunchDescription([...]).
    """
    stale_names = {
        "social_bringup",
        "social_rviz",
        "social_bringup_delayed",
        "social_rviz_delayed",
    }

    for name in stale_names:
        text = re.sub(
            rf"^[ \t]*{re.escape(name)},[ \t]*\n",
            "",
            text,
            flags=re.MULTILINE,
        )

    return text


def remove_legacy_comment_block(text: str) -> str:
    """
    Remove the obsolete social-system architecture comment if it is still
    present.

    This operation is intentionally bounded by stable section markers.
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
        return text

    end = text.find(end_marker, start)

    require(
        end != -1,
        (
            "Found the legacy social section header but could not find "
            "the locomotion calibration section following it."
        ),
    )

    # Only remove this bounded section if no required Nav2 object is inside.
    candidate = text[start:end]

    require(
        "nav2_stack =" not in candidate,
        "Refusing to remove a section containing nav2_stack.",
    )

    require(
        "slam_stack =" not in candidate,
        "Refusing to remove a section containing slam_stack.",
    )

    print("[REMOVE] Legacy social architecture section")

    return text[:start] + text[end:]


def normalize_spacing(text: str) -> str:
    """Keep the resulting launch readable without reformatting the whole file."""
    return re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )


# ============================================================================
# Validation
# ============================================================================

def validate_python(text: str) -> None:
    """Validate source syntax in memory."""
    ast.parse(text)


def validate_required_contracts(text: str) -> None:
    """Ensure the migration did not remove required Nav2 contracts."""
    for required in REQUIRED_NAMES:
        require(
            required in text,
            f"Required Nav2 contract was removed: {required}",
        )

    require(
        "def resolve_navigation_goal_gate" in text,
        "resolve_navigation_goal_gate() is missing.",
    )

    require(
        "nav2_stack =" in text,
        "nav2_stack was accidentally removed.",
    )

    require(
        "slam_stack =" in text,
        "slam_stack was accidentally removed.",
    )

    require(
        '"enable_navigation_goal_gate": (' in text,
        (
            "The Nav2 sub-launch no longer receives "
            "enable_navigation_goal_gate."
        ),
    )


def validate_no_legacy_ownership(text: str) -> None:
    """Reject remaining active ownership from the old social architecture."""
    forbidden = (
        "social_system.launch.py",
        "social_bringup_delayed",
        "social_rviz_delayed",
        "social_rviz_launch",
        "start_social_system",
        "enable_social_perception",
        "enable_collaboration",
        "enable_hri",
        "enable_speaker_association",
        "enable_explanation_node",
        "enable_navigation_interaction_speech",
        "piper_backend",
        "piper_model_directory",
        "piper_executable",
        "piper_spanish_model",
        "piper_english_model",
        "g1_social_bringup_share",
        "social_perception_share",
    )

    remaining = [
        token
        for token in forbidden
        if token in text
    ]

    require(
        not remaining,
        (
            "Legacy social ownership remains after migration: "
            + ", ".join(remaining)
        ),
    )


def validate_hardware_boundary(text: str) -> None:
    """
    This launch must never become the owner of the external rl_sar hardware
    bringup.
    """
    require(
        "package=\"rl_sar\"" not in text
        and "package='rl_sar'" not in text,
        (
            "Unexpected rl_sar launch ownership detected. "
            "Hardware must remain external."
        ),
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    require(
        TARGET.is_file(),
        f"Target file does not exist: {TARGET}",
    )

    original = TARGET.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup = TARGET.with_name(
        TARGET.name
        + f".before_step10_{timestamp}.bak"
    )

    shutil.copy2(
        TARGET,
        backup,
    )

    print(f"[BACKUP] {backup}")

    text = original

    # ------------------------------------------------------------------
    # Remove obsolete social-system structures.
    # ------------------------------------------------------------------

    text = remove_legacy_comment_block(text)

    text = remove_legacy_assignments(text)

    text = remove_legacy_launch_arguments(text)

    text = remove_stale_launchdescription_entries(text)

    text = normalize_spacing(text)

    # ------------------------------------------------------------------
    # Validate before touching the real source.
    # ------------------------------------------------------------------

    validate_python(text)
    validate_required_contracts(text)
    validate_no_legacy_ownership(text)
    validate_hardware_boundary(text)

    temporary = TARGET.with_name(
        TARGET.name + ".step10.tmp"
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
        temporary.unlink(missing_ok=True)

        print()
        print("[FAILED] Generated launch file is not valid Python.")
        print(f"[BACKUP] Original preserved at: {backup}")

        raise

    # ------------------------------------------------------------------
    # Replace source only after all checks succeeded.
    # ------------------------------------------------------------------

    TARGET.write_text(
        text,
        encoding="utf-8",
    )

    temporary.unlink(missing_ok=True)

    py_compile.compile(
        str(TARGET),
        doraise=True,
    )

    print()
    print("============================================================")
    print("STEP 10 - G1 NAV2 BRINGUP CLEANUP")
    print("============================================================")
    print("[PASS] Python syntax")
    print("[PASS] Legacy social-system ownership removed")
    print("[PASS] Legacy social RViz ownership removed")
    print("[PASS] Legacy HRI ownership removed")
    print("[PASS] Legacy Piper ownership removed")
    print("[PASS] perception_profile preserved")
    print("[PASS] NavigationGoalGate remapping contract preserved")
    print("[PASS] Nav2 / SLAM branches preserved")
    print("[PASS] rl_sar remains external")
    print("============================================================")
    print(f"[UPDATED] {TARGET}")
    print(f"[BACKUP]  {backup}")


if __name__ == "__main__":
    main()