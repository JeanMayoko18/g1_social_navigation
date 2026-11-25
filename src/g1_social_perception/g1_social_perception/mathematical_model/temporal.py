"""
Shared temporal equations for prediction and observational memory.

Receives:
    Source timestamps in integer nanoseconds.
    Local monotonic reception times in seconds.
    Explicit freshness and prediction limits.

Calculates:
    Source interval:
        delta_t = (current_stamp_ns - previous_stamp_ns) / 1e9

    Source age:
        source_age = (now_source_ns - observation_stamp_ns) / 1e9

    Reception age:
        receipt_age = now_monotonic - received_monotonic

Provides:
    Durations and freshness decisions without modifying stored evidence.

Clock rules:
    Source timestamps must belong to the same clock domain.
    Monotonic times must belong to the same local clock.
    Source and monotonic timestamps are never subtracted from each other.

Memory rule:
    Freshness uses the timestamp of the feature being inspected.
    A recent position does not renew an older appearance descriptor.

Boundary:
    Fresh evidence does not imply confirmed identity or social clearance.
    Callers handle source-session changes and clock resets.
"""


from typing import Optional

from .types import Measurement
from .validation import finite, integer, positive


NANOSECONDS_PER_SECOND = 1_000_000_000


# ============================================================================
# SOURCE-TIME DIFFERENCES
# ============================================================================


def source_interval_sec(
    current_stamp_ns: int,
    previous_stamp_ns: int,
) -> float:
    """
    Calculate the signed interval between two source timestamps.

    Equation:
        delta_t = (t_current - t_previous) / 1e9

    Receives:
        Two positive source timestamps in nanoseconds.

    Returns:
        Positive: current observation is newer.
        Zero: duplicate timestamp.
        Negative: reversed temporal order.

    Integer subtraction occurs before conversion to seconds to preserve
    precision when source timestamps have large absolute values.
    """
    integer("current_stamp_ns", current_stamp_ns, minimum=1)
    integer("previous_stamp_ns", previous_stamp_ns, minimum=1)

    return (
        current_stamp_ns - previous_stamp_ns
    ) / NANOSECONDS_PER_SECOND


def motion_interval_sec(
    current_stamp_ns: int,
    previous_stamp_ns: int,
    minimum_interval_sec: float,
    maximum_interval_sec: float,
) -> Optional[float]:
    """
    Select an interval suitable for measured-velocity calculation.

    Equation used downstream:
        measured_velocity = (current_position - previous_position) / delta_t

    Processing:
        Require a positive interval within the configured limits.

    Returns:
        delta_t in seconds when admissible.
        None for duplicates, reversed time, too-short or too-long intervals.

    Why:
        Very short intervals amplify position noise.
        Long interruptions cannot establish continuous motion by themselves.

    The caller must also check identity generation and position validity.
    """
    positive("minimum_interval_sec", minimum_interval_sec)
    positive("maximum_interval_sec", maximum_interval_sec)

    if minimum_interval_sec > maximum_interval_sec:
        raise ValueError(
            "minimum_interval_sec must not exceed maximum_interval_sec"
        )

    interval = source_interval_sec(
        current_stamp_ns,
        previous_stamp_ns,
    )

    if not minimum_interval_sec <= interval <= maximum_interval_sec:
        return None

    return interval


def prediction_interval_sec(
    target_stamp_ns: int,
    position_stamp_ns: int,
    maximum_horizon_sec: float,
) -> Optional[float]:
    """
    Select the forward interval for a retained-position prediction.

    Equation used downstream:
        predicted_position = retained_position + velocity * delta_t

    Receives:
        Target observation timestamp.
        Timestamp of the retained position, not another updated feature.
        Maximum permitted prediction horizon.

    Returns:
        delta_t in [0, maximum_horizon_sec], or None when inadmissible.

    A zero interval is valid: prediction stays at the reference time.
    Backward prediction and excessive extrapolation are rejected.

    Velocity freshness is checked separately using its own timestamp.
    """
    positive("maximum_horizon_sec", maximum_horizon_sec)

    interval = source_interval_sec(
        target_stamp_ns,
        position_stamp_ns,
    )

    if not 0.0 <= interval <= maximum_horizon_sec:
        return None

    return interval


# ============================================================================
# OBSERVATION AND RECEPTION AGE
# ============================================================================


def source_age_sec(
    now_source_ns: int,
    observation_stamp_ns: int,
) -> float:
    """
    Calculate observation age using the source clock.

    Equation:
        age = (now_source_ns - observation_stamp_ns) / 1e9

    Returns:
        Signed age in seconds.

    Negative ages remain visible so the freshness rule can apply an
    explicit future tolerance. They are never silently clamped to zero.
    """
    integer("now_source_ns", now_source_ns, minimum=0)
    integer(
        "observation_stamp_ns",
        observation_stamp_ns,
        minimum=1,
    )

    return (
        now_source_ns - observation_stamp_ns
    ) / NANOSECONDS_PER_SECOND


def receipt_age_sec(
    now_monotonic: float,
    received_monotonic: float,
) -> float:
    """
    Calculate local elapsed time since message reception.

    Equation:
        age = now_monotonic - received_monotonic

    Both values must come from the same local monotonic clock.
    A negative result indicates inconsistent ordering or clock provenance.
    """
    finite("now_monotonic", now_monotonic)
    finite("received_monotonic", received_monotonic)

    return now_monotonic - received_monotonic


# ============================================================================
# FRESHNESS RULES
# ============================================================================


def source_is_fresh(
    now_source_ns: int,
    observation_stamp_ns: int,
    maximum_age_sec: float,
    future_tolerance_sec: float = 0.0,
) -> bool:
    """
    Check whether source evidence lies inside its permitted age window.

    Rule:
        -future_tolerance_sec <= source_age <= maximum_age_sec

    Future tolerance accommodates a configured clock discrepancy.
    It does not synchronize clocks or repair source timestamps.
    """
    positive("maximum_age_sec", maximum_age_sec)
    finite("future_tolerance_sec", future_tolerance_sec)

    if future_tolerance_sec < 0.0:
        raise ValueError(
            "future_tolerance_sec must be non-negative"
        )

    age = source_age_sec(
        now_source_ns,
        observation_stamp_ns,
    )

    return -future_tolerance_sec <= age <= maximum_age_sec


def observation_is_fresh(
    now_source_ns: int,
    observation_stamp_ns: int,
    now_monotonic: float,
    received_monotonic: float,
    maximum_source_age_sec: float,
    maximum_receipt_age_sec: float,
    future_tolerance_sec: float = 0.0,
) -> bool:
    """
    Require both recent source evidence and recent local reception.

    Rule:
        source_fresh AND 0 <= receipt_age <= maximum_receipt_age_sec

    Why:
        A recently received old message is still old evidence.
        A frozen source clock must not keep retained evidence fresh
        indefinitely while local monotonic time continues advancing.

    Repeated reports must preserve the original reception time of the
    actual observation; this function cannot detect a replaced timestamp.
    """
    positive(
        "maximum_receipt_age_sec",
        maximum_receipt_age_sec,
    )

    source_fresh = source_is_fresh(
        now_source_ns=now_source_ns,
        observation_stamp_ns=observation_stamp_ns,
        maximum_age_sec=maximum_source_age_sec,
        future_tolerance_sec=future_tolerance_sec,
    )

    receipt_age = receipt_age_sec(
        now_monotonic,
        received_monotonic,
    )

    return bool(
        source_fresh
        and 0.0 <= receipt_age <= maximum_receipt_age_sec
    )


def feature_is_usable_at(
    measurement: Measurement,
    target_stamp_ns: int,
    maximum_age_sec: float,
) -> bool:
    """
    Check a retained feature before comparing it with a new candidate.

    Rule:
        value exists
        AND quality > 0
        AND 0 <= target_time - feature_time <= maximum_age_sec

    Receives:
        One feature with its own timestamp and quality.
        The candidate observation timestamp.

    Returns:
        Whether the feature is available, reliable and recent enough.

    Example:
        A track received a new position at t=12 s, but its appearance
        was last measured at t=9 s. Appearance age at t=12 s is 3 s.

    This check does not compare frames or histogram schemas.
    Those compatibility checks belong to the corresponding cue module.
    """
    if not isinstance(measurement, Measurement):
        raise ValueError("measurement must be a Measurement")

    integer("target_stamp_ns", target_stamp_ns, minimum=1)
    positive("maximum_age_sec", maximum_age_sec)

    if not measurement.usable:
        return False

    return source_is_fresh(
        now_source_ns=target_stamp_ns,
        observation_stamp_ns=measurement.stamp_ns,
        maximum_age_sec=maximum_age_sec,
        future_tolerance_sec=0.0,
    )