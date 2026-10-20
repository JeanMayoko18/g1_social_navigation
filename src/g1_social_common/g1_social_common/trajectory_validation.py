"""
Canonical trajectory social-validation semantics.

Trajectory validation is trajectory-specific.

A VALID result for one trajectory must never authorize execution
of another trajectory.

UNKNOWN is intentionally distinct from CONFLICT:
absence of a known conflict is not positive authorization.

Important:
    These helpers classify validation results only.

    Execution authorization additionally requires:
        - an exact path identifier match;
        - a fresh validation result.

    Therefore:

        validation == VALID

    alone is never sufficient to authorize movement.
"""


TRAJECTORY_VALIDATION_UNKNOWN = "UNKNOWN"
TRAJECTORY_VALIDATION_VALID = "VALID"
TRAJECTORY_VALIDATION_CONFLICT = "CONFLICT"


TRAJECTORY_VALIDATION_RESULTS = frozenset({
    TRAJECTORY_VALIDATION_UNKNOWN,
    TRAJECTORY_VALIDATION_VALID,
    TRAJECTORY_VALIDATION_CONFLICT,
})


def normalize_trajectory_validation(result: str) -> str:
    """Normalize a trajectory-validation result."""

    return str(result or "").strip().upper()


def is_trajectory_validation_result(result: str) -> bool:
    """Return True when result is a canonical validation result."""

    return (
        normalize_trajectory_validation(result)
        in TRAJECTORY_VALIDATION_RESULTS
    )


def is_valid_trajectory_validation(result: str) -> bool:
    """Return True only for a positive VALID result."""

    return (
        normalize_trajectory_validation(result)
        == TRAJECTORY_VALIDATION_VALID
    )


def is_conflicting_trajectory_validation(result: str) -> bool:
    """Return True when the trajectory has a confirmed social conflict."""

    return (
        normalize_trajectory_validation(result)
        == TRAJECTORY_VALIDATION_CONFLICT
    )


def is_unknown_trajectory_validation(result: str) -> bool:
    """Return True when trajectory validity is not currently known."""

    return (
        normalize_trajectory_validation(result)
        == TRAJECTORY_VALIDATION_UNKNOWN
    )