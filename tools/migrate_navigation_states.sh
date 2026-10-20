#!/usr/bin/env bash
set -euo pipefail

WS="${HOME}/g1_ws"
SRC="${WS}/src"

COMMON="${SRC}/g1_social_common"
COMMON_PY="${COMMON}/g1_social_common"
BACKUP="${WS}/tools/navigation_states_migration_backup"

echo "[INFO] Starting canonical NavigationState migration"

mkdir -p "${BACKUP}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

backup_file() {
    local file="$1"
    local relative="${file#${SRC}/}"
    local destination="${BACKUP}/${relative}"

    mkdir -p "$(dirname "${destination}")"

    if [[ ! -e "${destination}" ]]; then
        cp -a "${file}" "${destination}"
        echo "[BACKUP] ${relative}"
    fi
}

require_file() {
    local file="$1"

    if [[ ! -f "${file}" ]]; then
        echo "[ERROR] Required file not found: ${file}" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. Ensure g1_social_common structure exists
# ---------------------------------------------------------------------------

mkdir -p "${COMMON}/resource"
mkdir -p "${COMMON_PY}"

touch "${COMMON}/resource/g1_social_common"
touch "${COMMON_PY}/__init__.py"

echo "[OK] g1_social_common structure"

# ---------------------------------------------------------------------------
# 2. Install canonical navigation-state contract
# ---------------------------------------------------------------------------

cat > "${COMMON_PY}/navigation_states.py" <<'PY'
"""Canonical navigation-state semantics for the G1 social stack.

This module is the single source of truth for NavigationState semantic
classification.

Consumers must use these constants and helpers instead of maintaining
local copies of navigation-state groups.
"""

NAVIGATION_UNKNOWN = "UNKNOWN"
NAVIGATION_IDLE = "IDLE"
NAVIGATION_NAVIGATING = "NAVIGATING"
NAVIGATION_SUCCEEDED = "SUCCEEDED"
NAVIGATION_ABORTED = "ABORTED"
NAVIGATION_CANCELED = "CANCELED"


INACTIVE_NAVIGATION_STATES = frozenset({
    NAVIGATION_UNKNOWN,
    NAVIGATION_IDLE,
    NAVIGATION_SUCCEEDED,
    NAVIGATION_ABORTED,
    NAVIGATION_CANCELED,
})


TERMINAL_NAVIGATION_STATES = frozenset({
    NAVIGATION_SUCCEEDED,
    NAVIGATION_ABORTED,
    NAVIGATION_CANCELED,
})


def normalize_navigation_state(state: str) -> str:
    """Normalize a navigation-state value."""

    return str(state or "").strip().upper()


def is_inactive_navigation_state(state: str) -> bool:
    """Return True when no navigation goal is actively executing."""

    return normalize_navigation_state(state) in INACTIVE_NAVIGATION_STATES


def is_terminal_navigation_state(state: str) -> bool:
    """Return True when the state represents a terminal navigation outcome."""

    return normalize_navigation_state(state) in TERMINAL_NAVIGATION_STATES


def is_navigating_state(state: str) -> bool:
    """Return True when navigation is actively executing."""

    return normalize_navigation_state(state) == NAVIGATION_NAVIGATING
PY

echo "[OK] canonical navigation_states.py"

# ---------------------------------------------------------------------------
# 3. Add package.xml runtime dependency idempotently
# ---------------------------------------------------------------------------

add_exec_dependency() {
    local package_xml="$1"

    require_file "${package_xml}"

    if grep -q '<exec_depend>g1_social_common</exec_depend>' "${package_xml}"; then
        echo "[KEEP] dependency already present: ${package_xml}"
        return
    fi

    backup_file "${package_xml}"

    python3 - "${package_xml}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

dependency = "  <exec_depend>g1_social_common</exec_depend>\n"

if "</package>" not in text:
    raise SystemExit(f"Invalid package.xml: {path}")

text = text.replace(
    "</package>",
    dependency + "\n</package>",
    1,
)

path.write_text(text)
PY

    echo "[PATCH] dependency added: ${package_xml}"
}

add_exec_dependency \
    "${SRC}/g1_social_perception/package.xml"

add_exec_dependency \
    "${SRC}/g1_robot_explainability/package.xml"

# ---------------------------------------------------------------------------
# 4. Python migration helper
# ---------------------------------------------------------------------------

python3 - "${SRC}" "${BACKUP}" <<'PY'
from pathlib import Path
import re
import shutil
import sys

src = Path(sys.argv[1])
backup_root = Path(sys.argv[2])


def backup(path: Path) -> None:
    relative = path.relative_to(src)
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        shutil.copy2(path, destination)


def add_import(text: str, names: list[str]) -> str:
    module = "g1_social_common.navigation_states"

    # Already imports from the canonical module.
    existing_pattern = re.compile(
        rf"from\s+{re.escape(module)}\s+import\s*\((.*?)\)",
        re.DOTALL,
    )

    match = existing_pattern.search(text)

    if match:
        current = [
            item.strip().rstrip(",")
            for item in match.group(1).splitlines()
            if item.strip()
        ]

        merged = []
        for name in current + names:
            if name not in merged:
                merged.append(name)

        replacement = (
            f"from {module} import (\n"
            + "".join(f"    {name},\n" for name in merged)
            + ")"
        )

        return (
            text[:match.start()]
            + replacement
            + text[match.end():]
        )

    import_block = (
        f"from {module} import (\n"
        + "".join(f"    {name},\n" for name in names)
        + ")\n"
    )

    lines = text.splitlines(keepends=True)

    insert_at = 0

    # Preserve shebang and module docstring/import organization as much as
    # possible. Insert immediately before the first existing import.
    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_at = index
            break
    else:
        insert_at = 0

    lines.insert(insert_at, import_block + "\n")
    return "".join(lines)


def patch_file(path: Path, transform) -> None:
    if not path.is_file():
        raise SystemExit(f"[ERROR] Missing required file: {path}")

    original = path.read_text()
    updated = transform(original)

    if updated == original:
        print(f"[KEEP]  {path.relative_to(src)}")
        return

    backup(path)
    path.write_text(updated)
    print(f"[PATCH] {path.relative_to(src)}")


# -----------------------------------------------------------------------
# social_navigation_decision_node.py
#
# This file currently owns duplicate NAVIGATION_* constants and the
# inactive-state set. Replace those definitions with the canonical source.
# -----------------------------------------------------------------------

social_decision = (
    src
    / "g1_social_perception"
    / "g1_social_perception"
    / "social_navigation_decision_node.py"
)


def migrate_social_decision(text: str) -> str:
    text = add_import(
        text,
        [
            "NAVIGATION_UNKNOWN",
            "NAVIGATION_IDLE",
            "NAVIGATION_NAVIGATING",
            "NAVIGATION_SUCCEEDED",
            "NAVIGATION_ABORTED",
            "NAVIGATION_CANCELED",
            "INACTIVE_NAVIGATION_STATES",
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    # Remove local NAVIGATION_* constant definitions.
    text = re.sub(
        r'^NAVIGATION_UNKNOWN\s*=\s*"UNKNOWN"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^NAVIGATION_IDLE\s*=\s*"IDLE"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^NAVIGATION_NAVIGATING\s*=\s*"NAVIGATING"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^NAVIGATION_SUCCEEDED\s*=\s*"SUCCEEDED"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^NAVIGATION_ABORTED\s*=\s*"ABORTED"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^NAVIGATION_CANCELED\s*=\s*"CANCELED"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove a local inactive-state set when it is exactly composed of the
    # canonical NAVIGATION_* constants.
    text = re.sub(
        r"""
        ^[A-Z_]*NAVIGATION[A-Z_]*\s*=\s*(?:frozenset|set)\s*\(\s*\{\s*
        NAVIGATION_UNKNOWN\s*,\s*
        NAVIGATION_IDLE\s*,\s*
        NAVIGATION_SUCCEEDED\s*,\s*
        NAVIGATION_ABORTED\s*,\s*
        NAVIGATION_CANCELED\s*,?\s*
        \}\s*\)\s*\n
        """,
        "",
        text,
        flags=re.MULTILINE | re.VERBOSE,
    )

    text = re.sub(
        r'state\s*==\s*NAVIGATION_NAVIGATING',
        'is_navigating_state(state)',
        text,
    )

    return text


patch_file(social_decision, migrate_social_decision)


# -----------------------------------------------------------------------
# explanation_context_node.py
# -----------------------------------------------------------------------

context_node = (
    src
    / "g1_social_perception"
    / "g1_social_perception"
    / "explanation_context_node.py"
)


def migrate_context(text: str) -> str:
    text = add_import(
        text,
        [
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    text = re.sub(
        r'msg\.navigation_state\s*==\s*"NAVIGATING"',
        'is_navigating_state(msg.navigation_state)',
        text,
    )

    text = re.sub(
        r"""
        msg\.navigation_state\s*
        in\s*\(\s*
        "IDLE"\s*,\s*
        "UNKNOWN"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \)
        """,
        "is_inactive_navigation_state(msg.navigation_state)",
        text,
        flags=re.VERBOSE,
    )

    return text


patch_file(context_node, migrate_context)


# -----------------------------------------------------------------------
# explanation_trigger_node.py
# -----------------------------------------------------------------------

trigger_node = (
    src
    / "g1_social_perception"
    / "g1_social_perception"
    / "explanation_trigger_node.py"
)


def migrate_trigger(text: str) -> str:
    text = add_import(
        text,
        [
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    text = re.sub(
        r'msg\.navigation_state\s*==\s*"NAVIGATING"',
        'is_navigating_state(msg.navigation_state)',
        text,
    )

    text = re.sub(
        r"""
        msg\.navigation_state\s*
        in\s*\(\s*
        "UNKNOWN"\s*,\s*
        "IDLE"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \)
        """,
        "is_inactive_navigation_state(msg.navigation_state)",
        text,
        flags=re.VERBOSE,
    )

    return text


patch_file(trigger_node, migrate_trigger)


# -----------------------------------------------------------------------
# explanation_node.py
# -----------------------------------------------------------------------

explanation_node = (
    src
    / "g1_robot_explainability"
    / "g1_robot_explainability"
    / "explanation_node.py"
)


def migrate_explanation(text: str) -> str:
    text = add_import(
        text,
        [
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    text = re.sub(
        r'context\.navigation_state\s*==\s*"NAVIGATING"',
        'is_navigating_state(context.navigation_state)',
        text,
    )

    text = re.sub(
        r"""
        context\.navigation_state\s*
        in\s*\(\s*
        "UNKNOWN"\s*,\s*
        "IDLE"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \)
        """,
        "is_inactive_navigation_state(context.navigation_state)",
        text,
        flags=re.VERBOSE,
    )

    # Also migrate the historical variant without UNKNOWN.
    text = re.sub(
        r"""
        context\.navigation_state\s*
        in\s*\(\s*
        "IDLE"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \)
        """,
        "is_inactive_navigation_state(context.navigation_state)",
        text,
        flags=re.VERBOSE,
    )

    return text


patch_file(explanation_node, migrate_explanation)


# -----------------------------------------------------------------------
# deterministic_fallback.py
# -----------------------------------------------------------------------

fallback = (
    src
    / "g1_robot_explainability"
    / "g1_robot_explainability"
    / "deterministic_fallback.py"
)


def migrate_fallback(text: str) -> str:
    text = add_import(
        text,
        [
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    text = re.sub(
        r'^STATE_NAVIGATING\s*=\s*"NAVIGATING"\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove the historical local inactive-state definition.
    text = re.sub(
        r"""
        ^[A-Z_]*WITHOUT[A-Z_]*\s*=\s*(?:frozenset|set)\s*\(\s*\{\s*
        "IDLE"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \}\s*\)\s*
        """,
        "",
        text,
        flags=re.MULTILINE | re.VERBOSE,
    )

    text = re.sub(
        r'context\.navigation_state\s*==\s*STATE_NAVIGATING',
        'is_navigating_state(context.navigation_state)',
        text,
    )

    # Replace references to the historical local inactive set.
    text = re.sub(
        r'context\.navigation_state\s+in\s+[A-Z_]*WITHOUT[A-Z_]*',
        'is_inactive_navigation_state(context.navigation_state)',
        text,
    )

    return text


patch_file(fallback, migrate_fallback)


# -----------------------------------------------------------------------
# tts_node.py
# -----------------------------------------------------------------------

tts = (
    src
    / "g1_robot_explainability"
    / "g1_robot_explainability"
    / "tts_node.py"
)


def migrate_tts(text: str) -> str:
    text = add_import(
        text,
        [
            "is_inactive_navigation_state",
            "is_navigating_state",
        ],
    )

    text = re.sub(
        r'msg\.navigation_state\s*==\s*"NAVIGATING"',
        'is_navigating_state(msg.navigation_state)',
        text,
    )

    text = re.sub(
        r'state\.navigation_state\s*==\s*"NAVIGATING"',
        'is_navigating_state(state.navigation_state)',
        text,
    )

    text = re.sub(
        r"""
        msg\.navigation_state\s*
        in\s*\(\s*
        "IDLE"\s*,\s*
        "SUCCEEDED"\s*,\s*
        "ABORTED"\s*,\s*
        "CANCELED"\s*,?\s*
        \)
        """,
        "is_inactive_navigation_state(msg.navigation_state)",
        text,
        flags=re.VERBOSE,
    )

    return text


patch_file(tts, migrate_tts)

print("[OK] Python consumer migration completed")
PY

# ---------------------------------------------------------------------------
# 5. Syntax validation
# ---------------------------------------------------------------------------

echo
echo "[INFO] Running Python syntax validation..."

python3 -m py_compile \
    "${COMMON_PY}/navigation_states.py" \
    "${SRC}/g1_social_perception/g1_social_perception/explanation_context_node.py" \
    "${SRC}/g1_social_perception/g1_social_perception/explanation_trigger_node.py" \
    "${SRC}/g1_social_perception/g1_social_perception/social_navigation_decision_node.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/explanation_node.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/deterministic_fallback.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/tts_node.py"

echo "[PASS] Python syntax"

# ---------------------------------------------------------------------------
# 6. Report remaining semantic duplicates
# ---------------------------------------------------------------------------

echo
echo "[INFO] Remaining NavigationState literals in migrated consumers:"
echo

grep -nE \
    '"UNKNOWN"|"IDLE"|"NAVIGATING"|"SUCCEEDED"|"ABORTED"|"CANCELED"' \
    "${SRC}/g1_social_perception/g1_social_perception/explanation_context_node.py" \
    "${SRC}/g1_social_perception/g1_social_perception/explanation_trigger_node.py" \
    "${SRC}/g1_social_perception/g1_social_perception/social_navigation_decision_node.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/explanation_node.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/deterministic_fallback.py" \
    "${SRC}/g1_robot_explainability/g1_robot_explainability/tts_node.py" \
    || true

echo
echo "[DONE] NavigationState migration completed."
echo "[INFO] Backup: ${BACKUP}"