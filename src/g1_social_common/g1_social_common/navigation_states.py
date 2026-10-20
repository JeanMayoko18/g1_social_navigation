"""Canonical navigation-state semantics for the G1 social stack.

This module is the single source of truth for NavigationState semantic
classification.

These states describe the navigation lifecycle reported by the
NavigationGoalGate. They must not be confused with collaboration-policy
states such as STOP, WAIT, REPLAN or RESUME.

Consumers must use these constants and helpers instead of maintaining
local copies of navigation-state groups.

Important:
    Navigation lifecycle state alone must never be used as proof that
    the physical robot is stably immobile.
"""

NAVIGATION_UNKNOWN = "UNKNOWN"
NAVIGATION_IDLE = "IDLE"

NAVIGATION_WAITING_FOR_SPEECH = "WAITING_FOR_SPEECH"
NAVIGATION_FORWARDING = "FORWARDING"
NAVIGATION_NAVIGATING = "NAVIGATING"
NAVIGATION_CANCELING = "CANCELING"

NAVIGATION_SUCCEEDED = "SUCCEEDED"
NAVIGATION_ABORTED = "ABORTED"
NAVIGATION_CANCELED = "CANCELED"


NAVIGATION_STATES = frozenset({
    NAVIGATION_UNKNOWN,
    NAVIGATION_IDLE,
    NAVIGATION_WAITING_FOR_SPEECH,
    NAVIGATION_FORWARDING,
    NAVIGATION_NAVIGATING,
    NAVIGATION_CANCELING,
    NAVIGATION_SUCCEEDED,
    NAVIGATION_ABORTED,
    NAVIGATION_CANCELED,
})


ACTIVE_NAVIGATION_STATES = frozenset({
    NAVIGATION_FORWARDING,
    NAVIGATION_NAVIGATING,
    NAVIGATION_CANCELING,
})


INACTIVE_NAVIGATION_STATES = frozenset({
    NAVIGATION_UNKNOWN,
    NAVIGATION_IDLE,
    NAVIGATION_WAITING_FOR_SPEECH,
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


def is_navigation_state(state: str) -> bool:
    """Return True when state belongs to the canonical navigation lifecycle."""

    return normalize_navigation_state(state) in NAVIGATION_STATES


def is_active_navigation_state(state: str) -> bool:
    """Return True when navigation execution is active or being canceled."""

    return normalize_navigation_state(state) in ACTIVE_NAVIGATION_STATES


def is_inactive_navigation_state(state: str) -> bool:
    """Return True when no navigation trajectory is actively executing."""

    return normalize_navigation_state(state) in INACTIVE_NAVIGATION_STATES


def is_terminal_navigation_state(state: str) -> bool:
    """Return True when the state represents a terminal navigation outcome."""

    return normalize_navigation_state(state) in TERMINAL_NAVIGATION_STATES


def is_waiting_for_speech_state(state: str) -> bool:
    """Return True when the goal gate is waiting for speech completion."""

    return (
        normalize_navigation_state(state)
        == NAVIGATION_WAITING_FOR_SPEECH
    )


def is_forwarding_state(state: str) -> bool:
    """Return True while a navigation goal is being forwarded."""

    return (
        normalize_navigation_state(state)
        == NAVIGATION_FORWARDING
    )


def is_navigating_state(state: str) -> bool:
    """Return True when the internal navigation goal is executing."""

    return (
        normalize_navigation_state(state)
        == NAVIGATION_NAVIGATING
    )


def is_canceling_state(state: str) -> bool:
    """Return True while cancellation of the navigation goal is in progress."""

    return (
        normalize_navigation_state(state)
        == NAVIGATION_CANCELING
    )


def is_canceled_state(state: str) -> bool:
    """Return True when navigation cancellation has completed."""

    return (
        normalize_navigation_state(state)
        == NAVIGATION_CANCELED
    )