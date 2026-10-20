"""Canonical social-episode states for the G1 social stack."""

EPISODE_IDLE = "IDLE"
EPISODE_CONFLICT = "CONFLICT"
EPISODE_OBSERVATION_LOST = "OBSERVATION_LOST"
EPISODE_CLEARED = "CLEARED"


SOCIAL_EPISODE_STATES = frozenset({
    EPISODE_IDLE,
    EPISODE_CONFLICT,
    EPISODE_OBSERVATION_LOST,
    EPISODE_CLEARED,
})


def normalize_social_episode_state(state: str) -> str:
    return str(state or "").strip().upper()


def is_social_episode_state(state: str) -> bool:
    return (
        normalize_social_episode_state(state)
        in SOCIAL_EPISODE_STATES
    )