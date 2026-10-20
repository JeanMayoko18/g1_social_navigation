"""
Canonical CollaborationManager FSM states.

These states describe collaboration policy and coordination only.

They are intentionally distinct from:
    - navigation lifecycle states;
    - social-episode perception states;
    - trajectory-validation results.

Only CollaborationManager owns transitions between these states.
"""


STATE_NAVIGATING = "NAVIGATING"

STATE_STOPPING_FOR_SOCIAL_CONFLICT = (
    "STOPPING_FOR_SOCIAL_CONFLICT"
)

STATE_WAITING = "WAITING"

STATE_ANNOUNCING_DETECTION = (
    "ANNOUNCING_DETECTION"
)

STATE_SOCIAL_RESOLUTION = "SOCIAL_RESOLUTION"

STATE_OBSERVING = "OBSERVING"

STATE_ASKING = "ASKING"

STATE_WAITING_FOR_HUMAN = (
    "WAITING_FOR_HUMAN"
)

STATE_CONFIRMING_HUMAN_INTENT = (
    "CONFIRMING_HUMAN_INTENT"
)

STATE_WAITING_FOR_CONFIRMATION = (
    "WAITING_FOR_CONFIRMATION"
)

STATE_VALIDATING_TRAJECTORY = (
    "VALIDATING_TRAJECTORY"
)

STATE_REPLANNING = "REPLANNING"

STATE_PROPOSING_ALTERNATIVE = (
    "PROPOSING_ALTERNATIVE"
)

STATE_RESUMING = "RESUMING"


COLLABORATION_STATES = frozenset({
    STATE_NAVIGATING,
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
    STATE_WAITING,
    STATE_ANNOUNCING_DETECTION,
    STATE_SOCIAL_RESOLUTION,
    STATE_OBSERVING,
    STATE_ASKING,
    STATE_WAITING_FOR_HUMAN,
    STATE_CONFIRMING_HUMAN_INTENT,
    STATE_WAITING_FOR_CONFIRMATION,
    STATE_VALIDATING_TRAJECTORY,
    STATE_REPLANNING,
    STATE_PROPOSING_ALTERNATIVE,
    STATE_RESUMING,
})


def normalize_collaboration_state(state: str) -> str:
    """Normalize a CollaborationManager FSM state."""

    return str(state or "").strip().upper()


def is_collaboration_state(state: str) -> bool:
    """Return True when state belongs to the canonical collaboration FSM."""

    return (
        normalize_collaboration_state(state)
        in COLLABORATION_STATES
    )


def collaboration_state_is(state: str, expected: str) -> bool:
    """Compare two collaboration states after normalization."""

    return (
        normalize_collaboration_state(state)
        == normalize_collaboration_state(expected)
    )