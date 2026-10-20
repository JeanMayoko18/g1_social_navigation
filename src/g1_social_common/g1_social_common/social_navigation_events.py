"""Canonical social-navigation event semantics for the G1 social stack.

This module is the single source of truth for factual social-navigation
events produced by perception/decision nodes.

These events describe observed social situations.
They do not prescribe navigation actions.
"""

SOCIAL_EVENT_INTRUSION_CONFIRMED = "INTRUSION_CONFIRMED"
SOCIAL_EVENT_CONFLICT_PERSISTENT = "CONFLICT_PERSISTENT"
SOCIAL_EVENT_CLEAR = "CLEAR"
SOCIAL_EVENT_OBSERVATION_LOST = "OBSERVATION_LOST"
SOCIAL_EVENT_OBSERVATION_REACQUIRED = "OBSERVATION_REACQUIRED"


SOCIAL_NAVIGATION_EVENTS = frozenset({
    SOCIAL_EVENT_INTRUSION_CONFIRMED,
    SOCIAL_EVENT_CONFLICT_PERSISTENT,
    SOCIAL_EVENT_CLEAR,
    SOCIAL_EVENT_OBSERVATION_LOST,
    SOCIAL_EVENT_OBSERVATION_REACQUIRED,
})

def normalize_social_navigation_event(event: str) -> str:
    """Normalize a social-navigation event value."""

    return str(event or "").strip().upper()


def is_social_navigation_event(event: str) -> bool:
    """Return True for a canonical social-navigation event."""

    return (
        normalize_social_navigation_event(event)
        in SOCIAL_NAVIGATION_EVENTS
    )
