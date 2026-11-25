"""
Position prediction before track/candidate association.

Receives:
    Retained track measurements.
    Target source timestamp.
    Prediction horizon and velocity freshness limits.
    Explicit quality reduction for a constant-position fallback.

Calculates:
    delta_t = (target_stamp_ns - position_stamp_ns) / 1e9
    predicted_position = position + velocity * delta_t

Returns:
    A separate prediction with its mode, quality and reference timestamps.

Memory rule:
    Prediction never replaces a measurement, increments observation_count
    or renews a feature timestamp.

Boundary:
    Candidate positions do not participate in prediction.
    The association stage compares them with the returned prediction.
"""


from dataclasses import dataclass
from typing import Optional

from .temporal import (
    feature_is_usable_at,
    prediction_interval_sec,
)
from .types import TrackState, Vector3
from .validation import (
    finite,
    integer,
    positive,
    unit_interval,
)


# ============================================================================
# PREDICTION RESULT
# ============================================================================


@dataclass(frozen=True)
class PositionPrediction:
    """
    Estimated odom position at a target time, separate from measurements.

    Fields:
        position:
            Predicted position in metres, or None when unavailable.

        target_stamp_ns:
            Time for which the prediction was calculated.

        position_stamp_ns:
            Timestamp of the actual position used as reference.

        velocity_stamp_ns:
            Timestamp of the velocity used, if any.

        horizon_sec:
            Elapsed source time since the reference position.

        quality:
            Reliability available to the downstream motion cue.
            This is not yet the pairwise quality q[m, i, j].

        mode:
            AT_REFERENCE_TIME: no extrapolation is needed.
            CONSTANT_VELOCITY: position + velocity * elapsed time.
            CONSTANT_POSITION: qualified fallback without usable velocity.
            UNAVAILABLE: no admissible prediction.

        reason:
            Explanation of the selected mode or unavailability.

    The target timestamp never becomes a new measurement timestamp.
    """

    position: Optional[Vector3]
    target_stamp_ns: int
    position_stamp_ns: Optional[int]
    velocity_stamp_ns: Optional[int]
    horizon_sec: Optional[float]
    quality: float
    mode: str
    reason: str

    def __post_init__(self):
        integer("target_stamp_ns", self.target_stamp_ns, minimum=1)
        unit_interval("prediction quality", self.quality)

        if not isinstance(self.reason, str):
            raise ValueError("Prediction reason must be a string")

        if self.mode == "UNAVAILABLE":
            if (
                self.position is not None
                or self.position_stamp_ns is not None
                or self.velocity_stamp_ns is not None
                or self.horizon_sec is not None
                or self.quality != 0.0
            ):
                raise ValueError(
                    "Unavailable prediction requires no geometry "
                    "or reference times and zero quality"
                )
            return

        if self.mode not in {
            "AT_REFERENCE_TIME",
            "CONSTANT_VELOCITY",
            "CONSTANT_POSITION",
        }:
            raise ValueError("Unknown position prediction mode")

        if not isinstance(self.position, Vector3):
            raise ValueError("Predicted position must be Vector3")

        integer(
            "position_stamp_ns",
            self.position_stamp_ns,
            minimum=1,
        )
        finite("horizon_sec", self.horizon_sec)

        expected_horizon = (
            self.target_stamp_ns - self.position_stamp_ns
        ) / 1_000_000_000

        if (
            self.horizon_sec < 0.0
            or self.horizon_sec != expected_horizon
            or self.quality <= 0.0
        ):
            raise ValueError("Invalid prediction horizon or quality")

        if self.mode == "AT_REFERENCE_TIME":
            if self.horizon_sec != 0.0:
                raise ValueError(
                    "Reference-time prediction requires a zero horizon"
                )
        elif self.horizon_sec <= 0.0:
            raise ValueError(
                "Extrapolation requires a positive horizon"
            )

        if self.mode == "CONSTANT_VELOCITY":
            integer(
                "velocity_stamp_ns",
                self.velocity_stamp_ns,
                minimum=1,
            )

            if self.velocity_stamp_ns > self.position_stamp_ns:
                raise ValueError(
                    "Velocity cannot be newer than the reference position"
                )
        elif self.velocity_stamp_ns is not None:
            raise ValueError(
                "This prediction mode does not use velocity"
            )

    @classmethod
    def unavailable(cls, target_stamp_ns, reason):
        """Return explicit unavailability without inventing a position."""
        return cls(
            position=None,
            target_stamp_ns=target_stamp_ns,
            position_stamp_ns=None,
            velocity_stamp_ns=None,
            horizon_sec=None,
            quality=0.0,
            mode="UNAVAILABLE",
            reason=reason,
        )


# ============================================================================
# POSITION PREDICTION
# ============================================================================


def predict_position_odom(
    track: TrackState,
    target_stamp_ns: int,
    maximum_horizon_sec: float,
    maximum_velocity_age_sec: float,
    constant_position_quality_scale: float,
) -> PositionPrediction:
    """
    Predict a retained track position at a candidate observation time.

    Processing:
        1. Validate the target time and retained position.
        2. Bound extrapolation from the position's own timestamp.
        3. Use a usable, recent velocity when available.
        4. Otherwise apply an explicitly qualified position fallback.

    Prediction quality policy:
        At reference time:
            q_prediction = q_position

        Constant velocity:
            q_prediction = min(q_position, q_velocity)

        Constant position:
            q_prediction = q_position * fallback_scale

    These are explicit quality policies, not calibrated probabilities.
    Pairwise quality and distance tolerance are calculated downstream.

    Constraints:
        0 <= fallback_scale < 1
        fallback_scale = 0 disables the fallback.

    Caller responsibility:
        Velocity must belong to the track's current continuity segment.
        Reset it when identity generation changes.
        Check producer session and stream freshness before association.
    """
    if not isinstance(track, TrackState):
        raise ValueError("track must be TrackState")

    integer("target_stamp_ns", target_stamp_ns, minimum=1)
    positive("maximum_horizon_sec", maximum_horizon_sec)
    positive("maximum_velocity_age_sec", maximum_velocity_age_sec)

    unit_interval(
        "constant_position_quality_scale",
        constant_position_quality_scale,
    )

    if constant_position_quality_scale >= 1.0:
        raise ValueError(
            "constant_position_quality_scale must be smaller than one"
        )

    # Do not predict backward from a later retained tracking state.
    if target_stamp_ns < track.last_observation_ns:
        return PositionPrediction.unavailable(
            target_stamp_ns,
            "Target precedes the latest accepted track observation",
        )

    position = track.position_odom

    if not position.usable:
        return PositionPrediction.unavailable(
            target_stamp_ns,
            "No usable retained odom position",
        )

    horizon = prediction_interval_sec(
        target_stamp_ns=target_stamp_ns,
        position_stamp_ns=position.stamp_ns,
        maximum_horizon_sec=maximum_horizon_sec,
    )

    if horizon is None:
        return PositionPrediction.unavailable(
            target_stamp_ns,
            "Position reference is outside the prediction horizon",
        )

    if horizon == 0.0:
        return PositionPrediction(
            position=position.value,
            target_stamp_ns=target_stamp_ns,
            position_stamp_ns=position.stamp_ns,
            velocity_stamp_ns=None,
            horizon_sec=0.0,
            quality=position.quality,
            mode="AT_REFERENCE_TIME",
            reason="Target equals the retained position timestamp",
        )

    velocity = track.velocity_odom

    velocity_usable = feature_is_usable_at(
        measurement=velocity,
        target_stamp_ns=target_stamp_ns,
        maximum_age_sec=maximum_velocity_age_sec,
    )

    # An older velocity may be retained if still fresh.
    # A velocity newer than the position cannot describe this reference.
    if (
        velocity_usable
        and velocity.stamp_ns <= position.stamp_ns
    ):
        p = position.value
        v = velocity.value

        predicted = Vector3(
            x=p.x + v.x * horizon,
            y=p.y + v.y * horizon,
            z=p.z + v.z * horizon,
        )

        return PositionPrediction(
            position=predicted,
            target_stamp_ns=target_stamp_ns,
            position_stamp_ns=position.stamp_ns,
            velocity_stamp_ns=velocity.stamp_ns,
            horizon_sec=horizon,
            quality=min(position.quality, velocity.quality),
            mode="CONSTANT_VELOCITY",
            reason="Extrapolated using a usable retained velocity",
        )

    fallback_quality = (
        position.quality * constant_position_quality_scale
    )

    if fallback_quality <= 0.0:
        return PositionPrediction.unavailable(
            target_stamp_ns,
            "No usable velocity and constant-position fallback disabled",
        )

    return PositionPrediction(
        position=position.value,
        target_stamp_ns=target_stamp_ns,
        position_stamp_ns=position.stamp_ns,
        velocity_stamp_ns=None,
        horizon_sec=horizon,
        quality=fallback_quality,
        mode="CONSTANT_POSITION",
        reason="Retained position used with reduced quality",
    )