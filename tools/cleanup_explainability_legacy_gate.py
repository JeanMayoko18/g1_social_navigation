#!/usr/bin/env python3
"""
Step 10F - Remove the legacy NavigationGoalGate from
g1_robot_explainability.

Final ownership
---------------

g1_social_bringup/system.launch.py
    |
    +-- navigation.launch.py
            |
            +-- g1_nav2/navigation_goal_gate_node.py
            |
            +-- g1_nav2/g1_bringup.launch.py
                    |
                    +-- Nav2 action remapping

g1_robot_explainability
    |
    +-- explanation_node
    +-- Piper implementation used by speech.launch.py
    +-- navigation interaction speech implementation
    |
    +-- NO NavigationGoalGate ownership

This migration removes only the obsolete gate contract.

It deliberately does NOT remove the remaining historical speech/Piper/audio
configuration from explainability.launch.py. That cleanup is performed
separately after this migration so the two architectural changes remain
independently verifiable.
"""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


WS = Path.home() / "g1_ws"

PACKAGE = (
    WS
    / "src"
    / "g1_robot_explainability"
)

LAUNCH_FILE = (
    PACKAGE
    / "launch"
    / "explainability.launch.py"
)

SETUP_FILE = PACKAGE / "setup.py"

LEGACY_GATE_FILE = (
    PACKAGE
    / "g1_robot_explainability"
    / "navigation_goal_gate_node.py"
)

ACTIVE_GATE_FILE = (
    WS
    / "src"
    / "g1_nav2"
    / "g1_nav2"
    / "navigation_goal_gate_node.py"
)

NAVIGATION_LAUNCH = (
    WS
    / "src"
    / "g1_social_bringup"
    / "launch"
    / "navigation.launch.py"
)

BACKUP_ROOT = (
    WS
    / "tools"
    / "step10f_backups"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


# ============================================================================
# Generic helpers
# ============================================================================

def require(condition: bool, message: str) -> None:
    """Abort before installing an unsafe migration."""
    if not condition:
        raise RuntimeError(message)


def backup_file(path: Path) -> None:
    """Preserve a file using its workspace-relative path."""
    require(
        path.is_file(),
        f"Expected file does not exist: {path}",
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
    """Compile Python without executing ROS launch code."""
    py_compile.compile(
        str(path),
        doraise=True,
    )


def find_function(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef:
    """Return a named top-level function."""
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == name
        ):
            return node

    raise RuntimeError(
        f"Function {name}() was not found."
    )


def assignment_name(node: ast.AST) -> str | None:
    """Return the variable name for a simple assignment."""
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


def source_range(node: ast.AST) -> tuple[int, int]:
    """
    Return zero-based source line indexes.

    The end index is exclusive.
    """
    require(
        hasattr(node, "lineno")
        and hasattr(node, "end_lineno"),
        "AST source locations are unavailable.",
    )

    return (
        int(node.lineno) - 1,
        int(node.end_lineno),
    )


def remove_ranges(
    text: str,
    ranges: list[tuple[int, int]],
) -> str:
    """Remove source ranges from bottom to top."""
    lines = text.splitlines(keepends=True)

    for start, end in sorted(
        ranges,
        key=lambda value: value[0],
        reverse=True,
    ):
        del lines[start:end]

    return "".join(lines)


# ============================================================================
# explainability.launch.py migration
# ============================================================================

def remove_gate_assignments(text: str) -> str:
    """
    Remove the legacy gate resolution and effective goal-speech calculation.

    These values previously coupled explainability speech behavior to the
    NavigationGoalGate. The gate now belongs exclusively to navigation.
    """
    tree = ast.parse(text)

    function = find_function(
        tree,
        "create_runtime_nodes",
    )

    targets = {
        "enable_navigation_goal_gate",
        "requested_goal_speech",
        "effective_goal_speech",
    }

    ranges: list[tuple[int, int]] = []
    found: set[str] = set()

    for statement in function.body:
        name = assignment_name(statement)

        if name in targets:
            ranges.append(
                source_range(statement)
            )
            found.add(name)

    require(
        "enable_navigation_goal_gate" in found,
        (
            "Legacy enable_navigation_goal_gate assignment "
            "was not found."
        ),
    )

    print(
        "[REMOVE] Gate runtime variables: "
        + ", ".join(sorted(found))
    )

    return remove_ranges(
        text,
        ranges,
    )


def remove_gate_prints(text: str) -> str:
    """
    Remove the deployment-summary output describing the old gate.

    The bounded block starts at:
        print("Goal gate enabled ...")

    and ends immediately before:
        print("=" * 70)
    """
    start_pattern = re.compile(
        r'''
        \n
        [ \t]*print\(\s*
        ["']Goal\ gate\ enabled\ :\s*["']
        .*?
        (?=
            \n[ \t]*print\(
            ["']=["']\s*\*\s*70
        )
        ''',
        re.VERBOSE | re.DOTALL,
    )

    updated, count = start_pattern.subn(
        "\n",
        text,
        count=1,
    )

    require(
        count == 1,
        (
            "Expected exactly one legacy gate deployment-summary "
            f"block, found {count}."
        ),
    )

    print(
        "[REMOVE] Legacy gate deployment summary"
    )

    return updated


def declared_argument_name(
    node: ast.AST,
) -> str | None:
    """Detect DeclareLaunchArgument('name', ...)."""
    if not isinstance(node, ast.Call):
        return None

    if not isinstance(node.func, ast.Name):
        return None

    if node.func.id != "DeclareLaunchArgument":
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


def remove_gate_launch_argument(text: str) -> str:
    """Remove enable_navigation_goal_gate from the launch API."""
    tree = ast.parse(text)

    lines = text.splitlines(keepends=True)

    ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if (
            declared_argument_name(node)
            != "enable_navigation_goal_gate"
        ):
            continue

        start, end = source_range(node)

        # Handle a comma placed on a separate line.
        if (
            end < len(lines)
            and lines[end].strip() == ","
        ):
            end += 1

        ranges.append(
            (start, end)
        )

    require(
        len(ranges) == 1,
        (
            "Expected exactly one "
            "DeclareLaunchArgument("
            "'enable_navigation_goal_gate'), "
            f"found {len(ranges)}."
        ),
    )

    print(
        "[REMOVE] Launch argument enable_navigation_goal_gate"
    )

    return remove_ranges(
        text,
        ranges,
    )


def remove_stale_gate_comments(text: str) -> str:
    """Remove comments that describe the obsolete gate/speech coupling."""
    text = re.sub(
        r'''
        ^[ \t]*\#\ The\ goal\ gate\ announces\ goal\ reception.*\n
        ^[ \t]*\#\ Disable\ the\ equivalent\ announcement.*\n
        ^[ \t]*\#\ prevent\ the\ same\ navigation\ goal.*\n
        ''',
        "",
        text,
        flags=re.MULTILINE | re.VERBOSE,
    )

    return text


def normalize_spacing(text: str) -> str:
    """Remove excessive blank lines without reformatting the source."""
    return re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )


# ============================================================================
# setup.py migration
# ============================================================================

def remove_legacy_gate_entrypoint(text: str) -> str:
    """
    Remove only the old g1_robot_explainability gate console entrypoint.

    Keep:
    - explanation_node
    - piper_tts
    - navigation_interaction_speech
    """
    tree = ast.parse(text)

    ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue

        if not isinstance(node.value, str):
            continue

        if not node.value.startswith(
            "navigation_goal_gate = "
        ):
            continue

        # The entrypoint is represented as a parenthesized expression made
        # from adjacent string literals. Walk upward structurally is not
        # available in the standard AST, so use the containing source line
        # only as a locator and remove the complete parenthesized block with
        # a tightly scoped regular expression below.
        ranges.append(
            source_range(node)
        )

    require(
        ranges,
        "Legacy navigation_goal_gate entrypoint was not found.",
    )

    pattern = re.compile(
        r'''
        \n[ \t]*\(
        [ \t\r\n]*
        ["']navigation_goal_gate\ =\ ["']
        [ \t\r\n]*
        ["']g1_robot_explainability\.["']
        [ \t\r\n]*
        ["']navigation_goal_gate_node:main["']
        [ \t\r\n]*
        \),
        ''',
        re.VERBOSE,
    )

    updated, count = pattern.subn(
        "",
        text,
        count=1,
    )

    require(
        count == 1,
        (
            "Expected exactly one legacy gate entrypoint block, "
            f"found {count}."
        ),
    )

    print(
        "[REMOVE] g1_robot_explainability gate entrypoint"
    )

    return updated


# ============================================================================
# Validation
# ============================================================================

def validate_launch(text: str) -> None:
    """Validate the new Explainability ownership boundary."""
    ast.parse(text)

    forbidden = (
        "enable_navigation_goal_gate",
        "effective_goal_speech",
        "Goal gate enabled",
        "navigation_goal_gate",
    )

    remaining = [
        token
        for token in forbidden
        if token in text
    ]

    require(
        not remaining,
        (
            "Legacy gate contract remains in explainability.launch.py: "
            + ", ".join(remaining)
        ),
    )

    # Explainability itself must remain.
    require(
        'executable="explanation_node"' in text,
        "explanation_node ownership was accidentally removed.",
    )


def validate_setup(text: str) -> None:
    """Ensure only the obsolete gate entrypoint was removed."""
    ast.parse(text)

    require(
        "navigation_goal_gate" not in text,
        "Legacy gate entrypoint still exists.",
    )

    for required in (
        "explanation_node = ",
        "piper_tts = ",
        "navigation_interaction_speech = ",
    ):
        require(
            required in text,
            f"Required entrypoint was accidentally removed: {required}",
        )


def validate_active_gate_boundary() -> None:
    """Verify the replacement gate still exists before deleting the old one."""
    require(
        ACTIVE_GATE_FILE.is_file(),
        (
            "Active g1_nav2 NavigationGoalGate implementation "
            "is missing."
        ),
    )

    require(
        NAVIGATION_LAUNCH.is_file(),
        "navigation.launch.py is missing.",
    )

    navigation_text = NAVIGATION_LAUNCH.read_text(
        encoding="utf-8",
    )

    require(
        'package="g1_nav2"' in navigation_text
        and 'executable="navigation_goal_gate"' in navigation_text,
        (
            "navigation.launch.py does not appear to launch "
            "the g1_nav2 NavigationGoalGate."
        ),
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    for path in (
        LAUNCH_FILE,
        SETUP_FILE,
        LEGACY_GATE_FILE,
    ):
        require(
            path.is_file(),
            f"Expected legacy file missing: {path}",
        )

    validate_active_gate_boundary()

    # ------------------------------------------------------------------
    # Backup everything before changing the package.
    # ------------------------------------------------------------------

    for path in (
        LAUNCH_FILE,
        SETUP_FILE,
        LEGACY_GATE_FILE,
    ):
        backup_file(path)

    # ------------------------------------------------------------------
    # Prepare explainability.launch.py in memory.
    # ------------------------------------------------------------------

    launch_text = LAUNCH_FILE.read_text(
        encoding="utf-8",
    )

    launch_text = remove_gate_assignments(
        launch_text
    )

    launch_text = remove_stale_gate_comments(
        launch_text
    )

    launch_text = remove_gate_prints(
        launch_text
    )

    launch_text = remove_gate_launch_argument(
        launch_text
    )

    launch_text = normalize_spacing(
        launch_text
    )

    validate_launch(
        launch_text
    )

    # ------------------------------------------------------------------
    # Prepare setup.py in memory.
    # ------------------------------------------------------------------

    setup_text = SETUP_FILE.read_text(
        encoding="utf-8",
    )

    setup_text = remove_legacy_gate_entrypoint(
        setup_text
    )

    validate_setup(
        setup_text
    )

    # ------------------------------------------------------------------
    # Validate temporary files before touching source.
    # ------------------------------------------------------------------

    launch_tmp = LAUNCH_FILE.with_suffix(
        ".launch.py.step10f.tmp"
    )

    setup_tmp = SETUP_FILE.with_suffix(
        ".py.step10f.tmp"
    )

    launch_tmp.write_text(
        launch_text,
        encoding="utf-8",
    )

    setup_tmp.write_text(
        setup_text,
        encoding="utf-8",
    )

    try:
        compile_python(launch_tmp)
        compile_python(setup_tmp)

    except Exception:
        launch_tmp.unlink(missing_ok=True)
        setup_tmp.unlink(missing_ok=True)

        print()
        print(
            "[FAILED] Generated Python did not compile."
        )
        print(
            f"[BACKUPS] {BACKUP_ROOT}"
        )

        raise

    # ------------------------------------------------------------------
    # Install modified sources.
    # ------------------------------------------------------------------

    LAUNCH_FILE.write_text(
        launch_text,
        encoding="utf-8",
    )

    SETUP_FILE.write_text(
        setup_text,
        encoding="utf-8",
    )

    launch_tmp.unlink(missing_ok=True)
    setup_tmp.unlink(missing_ok=True)

    # Old implementation is now redundant.
    LEGACY_GATE_FILE.unlink()

    # ------------------------------------------------------------------
    # Final static checks.
    # ------------------------------------------------------------------

    compile_python(LAUNCH_FILE)
    compile_python(SETUP_FILE)
    compile_python(ACTIVE_GATE_FILE)

    require(
        not LEGACY_GATE_FILE.exists(),
        "Legacy gate source was not deleted.",
    )

    validate_active_gate_boundary()

    print()
    print("=" * 68)
    print("STEP 10F - EXPLAINABILITY LEGACY GATE REMOVED")
    print("=" * 68)
    print("[PASS] explainability gate launch argument removed")
    print("[PASS] explainability gate runtime logic removed")
    print("[PASS] explainability gate deployment logging removed")
    print("[PASS] old explainability gate entrypoint removed")
    print("[PASS] old explainability gate source removed")
    print("[PASS] g1_nav2 gate implementation preserved")
    print("[PASS] navigation.launch.py remains gate owner")
    print("[PASS] explanation_node preserved")
    print("[PASS] Piper entrypoint preserved")
    print("[PASS] navigation interaction speech entrypoint preserved")
    print("[PASS] Python syntax")
    print("=" * 68)
    print(f"[BACKUPS] {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
