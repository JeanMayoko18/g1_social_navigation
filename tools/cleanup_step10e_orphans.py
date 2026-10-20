#!/usr/bin/env python3
"""
Step 10E - Remove confirmed orphaned legacy architecture files.

Confirmed architecture
----------------------

g1_social_bringup/system.launch.py
    -> current system orchestrator

g1_social_bringup/navigation.launch.py
    -> owns NavigationGoalGate lifecycle

g1_nav2/navigation_goal_gate_node.py
    -> current gate implementation

g1_social_bringup/speech.launch.py
    -> owns the single PiperTTSNode

g1_robot_explainability/tts_node.py
    -> Piper implementation, MUST remain

g1_hri/ros4hri_tts_adapter_node.py
    -> ROS4HRI TTS compatibility adapter

Files removed here are only components proven orphaned by the Step 10E audit.

This script does NOT modify rl_sar.
"""

from pathlib import Path
from datetime import datetime
import ast
import py_compile
import shutil
import re


WS = Path.home() / "g1_ws"
SRC = WS / "src"

BACKUP_ROOT = (
    WS
    / "tools"
    / "step10e_backups"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def backup_file(path: Path) -> None:
    """Preserve the original path relative to the workspace."""
    require(path.is_file(), f"Missing expected file: {path}")

    relative = path.relative_to(WS)
    destination = BACKUP_ROOT / relative

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(path, destination)

    print(f"[BACKUP] {relative}")


def delete_file(path: Path) -> None:
    """Backup and delete a confirmed orphan."""
    backup_file(path)
    path.unlink()
    print(f"[DELETE] {path.relative_to(WS)}")


def compile_file(path: Path) -> None:
    py_compile.compile(
        str(path),
        doraise=True,
    )


# =====================================================================
# 1. Confirm expected current architecture before changing anything
# =====================================================================

system_launch = (
    SRC
    / "g1_social_bringup"
    / "launch"
    / "system.launch.py"
)

navigation_launch = (
    SRC
    / "g1_social_bringup"
    / "launch"
    / "navigation.launch.py"
)

active_gate = (
    SRC
    / "g1_nav2"
    / "g1_nav2"
    / "navigation_goal_gate_node.py"
)

speech_launch = (
    SRC
    / "g1_social_bringup"
    / "launch"
    / "speech.launch.py"
)

piper_source = (
    SRC
    / "g1_robot_explainability"
    / "g1_robot_explainability"
    / "tts_node.py"
)

for required_file in (
    system_launch,
    navigation_launch,
    active_gate,
    speech_launch,
    piper_source,
):
    require(
        required_file.is_file(),
        f"Required current architecture file missing: {required_file}",
    )


navigation_text = navigation_launch.read_text(encoding="utf-8")
speech_text = speech_launch.read_text(encoding="utf-8")

require(
    'executable="navigation_goal_gate"' in navigation_text,
    "navigation.launch.py does not appear to own the active gate.",
)

require(
    'executable="piper_tts"' in speech_text,
    "speech.launch.py does not appear to own PiperTTSNode.",
)


# =====================================================================
# 2. Delete obsolete social_system.launch.py
# =====================================================================

legacy_social_system = (
    SRC
    / "g1_social_bringup"
    / "launch"
    / "social_system.launch.py"
)

delete_file(legacy_social_system)


# =====================================================================
# 3. Delete unused legacy HRI speech backend abstraction
# =====================================================================

legacy_native_backend = (
    SRC
    / "g1_hri"
    / "g1_hri"
    / "backends"
    / "native_backend.py"
)

legacy_speech_backend = (
    SRC
    / "g1_hri"
    / "g1_hri"
    / "backends"
    / "speech_backend.py"
)

delete_file(legacy_native_backend)
delete_file(legacy_speech_backend)


# =====================================================================
# 4. Fix stale social_decision documentation
# =====================================================================

social_decision = (
    SRC
    / "g1_social_bringup"
    / "launch"
    / "social_decision.launch.py"
)

backup_file(social_decision)

text = social_decision.read_text(encoding="utf-8")

text = text.replace(
    "perception.launch.py",
    "social_perception.launch.py",
)

social_decision.write_text(
    text,
    encoding="utf-8",
)

compile_file(social_decision)

print(
    "[PATCH] "
    "src/g1_social_bringup/launch/social_decision.launch.py"
)


# =====================================================================
# 5. Validate deleted HRI backends are no longer imported
# =====================================================================

hri_root = SRC / "g1_hri"

remaining_hri_text = ""

for path in hri_root.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue

    remaining_hri_text += path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

require(
    "NativeSpeechBackend" not in remaining_hri_text,
    "NativeSpeechBackend still has a live reference.",
)

require(
    "SpeechBackend" not in remaining_hri_text,
    "SpeechBackend still has a live reference.",
)

require(
    "native_backend" not in remaining_hri_text,
    "native_backend still has a live import.",
)

require(
    "speech_backend" not in remaining_hri_text,
    "speech_backend still has a live import.",
)


# =====================================================================
# 6. Validate social_system has no remaining caller
# =====================================================================

remaining_social_system_refs = []

for path in SRC.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix not in {
        ".py",
        ".xml",
        ".yaml",
        ".yml",
    } and path.name != "CMakeLists.txt":
        continue

    if (
        "__pycache__" in path.parts
        or "build" in path.parts
        or "install" in path.parts
        or "log" in path.parts
    ):
        continue

    content = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if "social_system.launch.py" in content:
        remaining_social_system_refs.append(
            str(path.relative_to(WS))
        )

require(
    not remaining_social_system_refs,
    (
        "social_system.launch.py still referenced by: "
        + ", ".join(remaining_social_system_refs)
    ),
)


# =====================================================================
# 7. Validate essential current architecture remains
# =====================================================================

require(
    active_gate.is_file(),
    "Active g1_nav2 gate was accidentally removed.",
)

require(
    piper_source.is_file(),
    "Piper implementation was accidentally removed.",
)

require(
    system_launch.is_file(),
    "Current system.launch.py was accidentally removed.",
)

require(
    speech_launch.is_file(),
    "Current speech.launch.py was accidentally removed.",
)


# =====================================================================
# 8. Compile modified/current critical launch files
# =====================================================================

for path in (
    system_launch,
    navigation_launch,
    speech_launch,
    social_decision,
):
    compile_file(path)


print()
print("============================================================")
print("STEP 10E - CONFIRMED ORPHANS CLEANED")
print("============================================================")
print("[PASS] legacy social_system.launch.py removed")
print("[PASS] legacy NativeSpeechBackend removed")
print("[PASS] legacy SpeechBackend removed")
print("[PASS] stale perception.launch.py documentation corrected")
print("[PASS] current system.launch.py preserved")
print("[PASS] current navigation.launch.py preserved")
print("[PASS] active g1_nav2 NavigationGoalGate preserved")
print("[PASS] speech.launch.py preserved")
print("[PASS] Piper implementation preserved")
print("[PASS] critical launch Python syntax")
print("============================================================")
print(f"[BACKUPS] {BACKUP_ROOT}")
