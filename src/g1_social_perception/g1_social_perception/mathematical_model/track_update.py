"""
Update retained track state from accepted actual observations.

Receives:
    A new-track identity or an existing track.
    An actual enriched detection.
    Explicit continuity, generation and update parameters.

Calculates:
    Observation counts.
    Velocity from successive measured positions.
    Optional exponential velocity smoothing.

Returns:
    A new immutable TrackState.

Memory rules:
    Only accepted actual observations increment observation_count.
    Unavailable features do not erase retained usable references.
    Retained features keep their original timestamps.
    Predictions never enter this module as observations.

Boundary:
    Association acceptance belongs to assignment.py.
    Identity allocation, continuity decisions and generation-token
    allocation belong to the tracker.
"""


from dataclasses import dataclass, replace

from .geometry import distance_3d
from .temporal import motion_interval_sec
from .types import (
    EnrichedDetection,
    IdentityKey,
    Measurement,
    TrackState,
    Vector3,
)
from .validation import integer, positive, unit_interval


# ============================================================================
# UPDATE POLICY
# ============================================================================


@dataclass(frozen=True)
class TrackUpdatePolicy:
    """
    Explicit parameters for confirmation and velocity estimation.

    confirmation_hits:
        Required consecutive accepted observations.

    minimum_velocity_interval_sec:
        Reject shorter intervals to limit noise amplification.

    maximum_velocity_interval_sec:
        Reject derivatives spanning excessive time gaps.

    maximum_speed_mps:
        Physical plausibility limit for measured 3D speed.
        Exceeding it invalidates velocity; it does not clip the measurement.

    velocity_alpha:
        Smoothing coefficient lambda in (0, 1].
        One uses the measured velocity without smoothing.

    No numerical defaults are imposed by this mathematical module.
    """

    confirmation_hits: int
    minimum_velocity_interval_sec: float
    maximum_velocity_interval_sec: float
    maximum_speed_mps: float
    velocity_alpha: float

    def __post_init__(self):
        integer(
            "confirmation_hits",
            self.confirmation_hits,
            minimum=1,
        )

        for name in (
            "minimum_velocity_interval_sec",
            "maximum_velocity_interval_sec",
            "maximum_speed_mps",
        ):
            positive(name, getattr(self, name))

        if (
            self.minimum_velocity_interval_sec
            > self.maximum_velocity_interval_sec
        ):
            raise ValueError(
                "Minimum velocity interval must not exceed maximum"
            )

        unit_interval("velocity_alpha", self.velocity_alpha)
        positive("velocity_alpha", self.velocity_alpha)


# ============================================================================
# FEATURE RETENTION
# ============================================================================


def _retain_or_replace(
    previous: Measurement,
    current: Measurement,
) -> Measurement:
    """
    Replace a retained reference only with a usable current feature.

    Returning the previous object preserves its value, quality and timestamp.
    Its age must still be checked before a later comparison.
    """
    return current if current.usable else previous


def _observed_bbox(candidate: EnrichedDetection) -> Measurement:
    """
    Wrap the actual source box as an image-space measurement.

    Policy:
        Detector confidence is used as the box's reference quality.
        It does not establish depth or appearance reliability.
    """
    return Measurement(
        value=candidate.source.bbox,
        quality=candidate.source.confidence,
        stamp_ns=candidate.source.stamp_ns,
        frame_id=candidate.source.image_frame,
    )


# ============================================================================
# VELOCITY FROM ACCEPTED OBSERVATIONS
# ============================================================================


def _updated_velocity(
    track: TrackState,
    candidate: EnrichedDetection,
    continuity_preserved: bool,
    policy: TrackUpdatePolicy,
) -> Measurement:
    """
    Estimate velocity only from consecutive usable position observations.

    Equations:
        v_measured = (p_current - p_previous) / delta_t
        v_filtered = alpha * v_measured + (1 - alpha) * v_previous

    Initialization:
        Without a usable previous velocity, use v_measured directly.
        Do not smooth the first estimate against an invented zero velocity.

    Quality policy:
        Measured quality is the minimum of the two position qualities.
        If smoothing uses previous velocity, also include its quality.

    Failure:
        Return explicit unavailability.
        An old velocity is not republished as a fresh estimate.
    """
    if not continuity_preserved:
        return Measurement.unavailable(
            "Velocity reset after a continuity break"
        )

    previous = track.position_odom
    current = candidate.position_odom

    if not previous.usable or not current.usable:
        return Measurement.unavailable(
            "Two usable position observations are required"
        )

    if previous.stamp_ns != track.last_observation_ns:
        return Measurement.unavailable(
            "Previous position was retained across a missing measurement"
        )

    interval = motion_interval_sec(
        current_stamp_ns=current.stamp_ns,
        previous_stamp_ns=previous.stamp_ns,
        minimum_interval_sec=policy.minimum_velocity_interval_sec,
        maximum_interval_sec=policy.maximum_velocity_interval_sec,
    )

    if interval is None:
        return Measurement.unavailable(
            "Observation interval is unsuitable for velocity estimation"
        )

    measured_speed = (
        distance_3d(current.value, previous.value) / interval
    )

    if measured_speed > policy.maximum_speed_mps:
        return Measurement.unavailable(
            "Measured speed exceeds the configured physical limit"
        )

    measured = Vector3(
        x=(current.value.x - previous.value.x) / interval,
        y=(current.value.y - previous.value.y) / interval,
        z=(current.value.z - previous.value.z) / interval,
    )

    quality = min(previous.quality, current.quality)
    velocity = measured

    old_velocity = track.velocity_odom
    alpha = policy.velocity_alpha

    if (
        alpha < 1.0
        and old_velocity.usable
        and old_velocity.stamp_ns == previous.stamp_ns
    ):
        velocity = Vector3(
            x=alpha * measured.x + (1.0 - alpha) * old_velocity.value.x,
            y=alpha * measured.y + (1.0 - alpha) * old_velocity.value.y,
            z=alpha * measured.z + (1.0 - alpha) * old_velocity.value.z,
        )

        quality = min(quality, old_velocity.quality)

    return Measurement(
        value=velocity,
        quality=quality,
        stamp_ns=current.stamp_ns,
        frame_id="odom",
    )


# ============================================================================
# NEW TRACK
# ============================================================================


def create_track(
    key: IdentityKey,
    generation: int,
    candidate: EnrichedDetection,
    policy: TrackUpdatePolicy,
) -> TrackState:
    """
    Initialize a track from an eligible actual observation.

    Initial state:
        observation_count = 1
        consecutive_hits = 1
        velocity = unavailable

    The caller must first establish new-track eligibility.
    An ambiguous unmatched candidate is not automatically eligible.

    Generation tokens are allocated by the tracker.
    """
    if not isinstance(key, IdentityKey):
        raise ValueError("key must be IdentityKey")

    if not isinstance(candidate, EnrichedDetection):
        raise ValueError("candidate must be EnrichedDetection")

    if not isinstance(policy, TrackUpdatePolicy):
        raise ValueError("policy must be TrackUpdatePolicy")

    integer("generation", generation, minimum=1)

    return TrackState(
        key=key,
        generation=generation,
        last_observation_ns=candidate.source.stamp_ns,
        observation_count=1,
        consecutive_hits=1,
        confirmed=policy.confirmation_hits == 1,
        bbox_reference=_observed_bbox(candidate),
        position_odom=candidate.position_odom,
        velocity_odom=Measurement.unavailable(
            "A new track has no measured motion history"
        ),
        appearance_upper=candidate.appearance_upper,
        appearance_lower=candidate.appearance_lower,
        apparent_extent=candidate.apparent_extent,
        metric_extent=candidate.metric_extent,
    )


# ============================================================================
# ACCEPTED ASSOCIATION
# ============================================================================


def update_track(
    track: TrackState,
    candidate: EnrichedDetection,
    generation: int,
    continuity_preserved: bool,
    policy: TrackUpdatePolicy,
) -> TrackState:
    """
    Commit one accepted association to an existing identity hypothesis.

    Counts:
        Total observation_count always increases by one.
        Consecutive hits increase only when continuity is preserved.
        Otherwise the current segment starts with one hit.

    Continuity:
        Preserved continuity keeps the existing generation.
        Broken continuity requires a newer generation supplied by the tracker.

    Confirmation policy:
        A track is confirmed after the configured consecutive-hit threshold.
        A new continuity segment must satisfy that threshold again.

    Features:
        Usable current measurements replace their retained references.
        Missing features keep their older reference timestamps.
        Velocity is recalculated or explicitly marked unavailable.

    Acceptance is a caller precondition; this function cannot reconstruct
    the global assignment decision from a candidate alone.
    """
    if not isinstance(track, TrackState):
        raise ValueError("track must be TrackState")

    if not isinstance(candidate, EnrichedDetection):
        raise ValueError("candidate must be EnrichedDetection")

    if not isinstance(policy, TrackUpdatePolicy):
        raise ValueError("policy must be TrackUpdatePolicy")

    if type(continuity_preserved) is not bool:
        raise ValueError("continuity_preserved must be a boolean")

    integer("generation", generation, minimum=1)

    if candidate.source.stamp_ns <= track.last_observation_ns:
        raise ValueError(
            "Accepted observation must be newer than the previous observation"
        )

    if continuity_preserved:
        if generation != track.generation:
            raise ValueError(
                "Preserved continuity must keep the same generation"
            )

        if track.consecutive_hits == 0:
            raise ValueError(
                "A missed association requires a new continuity segment"
            )
    elif generation <= track.generation:
        raise ValueError(
            "Broken continuity requires a newer generation"
        )

    hits = (
        track.consecutive_hits + 1
        if continuity_preserved
        else 1
    )

    velocity = _updated_velocity(
        track=track,
        candidate=candidate,
        continuity_preserved=continuity_preserved,
        policy=policy,
    )

    retained = {
        name: _retain_or_replace(
            getattr(track, name),
            getattr(candidate, name),
        )
        for name in (
            "position_odom",
            "appearance_upper",
            "appearance_lower",
            "apparent_extent",
            "metric_extent",
        )
    }

    return replace(
        track,
        generation=generation,
        last_observation_ns=candidate.source.stamp_ns,
        observation_count=track.observation_count + 1,
        consecutive_hits=hits,
        confirmed=hits >= policy.confirmation_hits,
        bbox_reference=_retain_or_replace(
            track.bbox_reference,
            _observed_bbox(candidate),
        ),
        velocity_odom=velocity,
        **retained,
    )


# ============================================================================
# MISSING ASSOCIATION
# ============================================================================


def mark_track_unobserved(track: TrackState) -> TrackState:
    """
    Record a missed association without manufacturing an observation.

    Changes:
        Reset consecutive hits and invalidate motion continuity.

    Preserves:
        Identity, generation, total observation count, last observation time
        and retained feature references.

    Confirmation remains historical until reassociation starts a new segment.
    It does not mean the person is currently observed.

    Call for an unmatched track in a processed source observation.
    A timer tick alone is not a new missed source observation.
    """
    if not isinstance(track, TrackState):
        raise ValueError("track must be TrackState")

    return replace(
        track,
        consecutive_hits=0,
        velocity_odom=Measurement.unavailable(
            "No accepted association in the current source observation"
        ),
    )