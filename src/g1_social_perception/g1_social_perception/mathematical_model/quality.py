"""
Evidence eligibility and pairwise reliability for association.

Receives:
    Measurements with values, qualities, source timestamps and frames.
    Explicit maximum feature ages.
    Motion predictions when evaluating the motion cue.

Calculates:
    Feature eligibility:
        available AND positive_quality AND temporally_valid

    Pairwise reliability:
        q_pair = min(q_reference, q_candidate)

    Motion reliability:
        q_motion = min(q_prediction, q_candidate_position)

Returns:
    A quality value and the reason for acceptance or exclusion.

Distinction:
    Quality describes whether evidence can be trusted.
    Similarity describes whether two valid pieces of evidence agree.

Boundary:
    Feature extraction supplies the initial measurement qualities.
    This module does not infer image visibility, depth reliability
    or occlusion from a detector confidence alone.

    Composite appearance/size qualities and effective association weights
    are calculated with their corresponding scores in association_scores.py.
"""


from dataclasses import dataclass

from .prediction import PositionPrediction
from .temporal import feature_is_usable_at
from .types import Histogram, Measurement, Vector3
from .validation import integer, positive, unit_interval


# ============================================================================
# QUALITY RESULT
# ============================================================================


@dataclass(frozen=True)
class QualityResult:
    """
    Reliability of one feature or one comparison.

    Fields:
        value:
            Quality in [0, 1]. Zero excludes the evidence from fusion.

        reason:
            Explanation suitable for association diagnostics.

    A positive quality is not an identity confirmation.
    """

    value: float
    reason: str

    def __post_init__(self):
        unit_interval("quality result", self.value)

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Quality reason must be a nonempty string")

    @property
    def usable(self):
        """Return whether the evidence may contribute to a score."""
        return self.value > 0.0


# ============================================================================
# SINGLE-FEATURE ELIGIBILITY
# ============================================================================


def feature_quality_at(
    measurement: Measurement,
    target_stamp_ns: int,
    maximum_age_sec: float,
) -> QualityResult:
    """
    Evaluate one feature at the candidate observation time.

    Processing:
        Reject missing values, zero quality, future reference times
        and references older than the permitted maximum.

    Returns:
        The original quality when eligible; otherwise zero.

    Policy:
        Age is a hard eligibility limit here.
        No uncalibrated gradual decay is applied.

    The feature's own timestamp is used, never the track's latest
    observation timestamp as a substitute.
    """
    if not isinstance(measurement, Measurement):
        raise ValueError("measurement must be a Measurement")

    integer("target_stamp_ns", target_stamp_ns, minimum=1)
    positive("maximum_age_sec", maximum_age_sec)

    if measurement.value is None:
        return QualityResult(
            0.0,
            measurement.reason or "Feature unavailable",
        )

    if measurement.quality == 0.0:
        return QualityResult(
            0.0,
            measurement.reason or "Feature has zero quality",
        )

    if measurement.stamp_ns > target_stamp_ns:
        return QualityResult(
            0.0,
            "Feature reference is newer than the target observation",
        )

    if not feature_is_usable_at(
        measurement=measurement,
        target_stamp_ns=target_stamp_ns,
        maximum_age_sec=maximum_age_sec,
    ):
        return QualityResult(
            0.0,
            "Feature exceeds its permitted age",
        )

    return QualityResult(
        measurement.quality,
        "Feature is available and within its permitted age",
    )


# ============================================================================
# REFERENCE / CURRENT-CANDIDATE RELIABILITY
# ============================================================================


def measurement_pair_quality(
    reference: Measurement,
    candidate: Measurement,
    target_stamp_ns: int,
    maximum_reference_age_sec: float,
) -> QualityResult:
    """
    Calculate reliability for compatible reference/candidate features.

    Equation:
        q_pair = min(q_reference, q_candidate)

    Receives:
        A retained reference feature.
        A feature from the current candidate observation.
        The candidate source timestamp.
        Maximum permitted age of the retained reference.

    Compatibility:
        Candidate timestamp must equal the target timestamp.
        Values must have the same concrete type and frame identifier.
        Histograms must also have equal schemas and bin counts.

    Returns:
        Pairwise reliability, or zero with an exclusion reason.

    Caller responsibility:
        Pass matching semantic fields:
        upper appearance with upper appearance,
        metric extent with metric extent, and so on.

        Matching frame names do not compensate for camera motion.
        Predicted image geometry requires its own validity checks.
    """
    if not isinstance(reference, Measurement):
        raise ValueError("reference must be a Measurement")

    if not isinstance(candidate, Measurement):
        raise ValueError("candidate must be a Measurement")

    integer("target_stamp_ns", target_stamp_ns, minimum=1)
    positive(
        "maximum_reference_age_sec",
        maximum_reference_age_sec,
    )

    if candidate.value is None:
        return QualityResult(
            0.0,
            candidate.reason or "Candidate feature unavailable",
        )

    if candidate.stamp_ns != target_stamp_ns:
        return QualityResult(
            0.0,
            "Candidate feature does not belong to the current observation",
        )

    if candidate.quality == 0.0:
        return QualityResult(
            0.0,
            candidate.reason or "Candidate feature has zero quality",
        )

    reference_quality = feature_quality_at(
        measurement=reference,
        target_stamp_ns=target_stamp_ns,
        maximum_age_sec=maximum_reference_age_sec,
    )

    if not reference_quality.usable:
        return reference_quality

    if type(reference.value) is not type(candidate.value):
        return QualityResult(
            0.0,
            "Reference and candidate value types differ",
        )

    if reference.frame_id != candidate.frame_id:
        return QualityResult(
            0.0,
            "Reference and candidate frames differ",
        )

    if isinstance(reference.value, Histogram):
        if (
            reference.value.schema_id != candidate.value.schema_id
            or len(reference.value.bins) != len(candidate.value.bins)
        ):
            return QualityResult(
                0.0,
                "Appearance extraction schemas or bin counts differ",
            )

    return QualityResult(
        min(reference_quality.value, candidate.quality),
        "Compatible reference and current candidate",
    )


# ============================================================================
# PREDICTED MOTION / CURRENT POSITION RELIABILITY
# ============================================================================


def motion_pair_quality(
    prediction: PositionPrediction,
    candidate_position: Measurement,
    target_stamp_ns: int,
) -> QualityResult:
    """
    Calculate motion-cue reliability before evaluating spatial distance.

    Equation:
        q_motion = min(q_prediction, q_candidate_position)

    Receives:
        An odom prediction produced for the current candidate time.
        The candidate's actual odom position.

    Returns:
        Reliability of the motion comparison, or zero if unavailable.

    The prediction already accounts for reference eligibility and its
    selected motion mode, including any constant-position quality reduction.

    Distance does not enter this quality calculation.
    A large distance may produce a low motion similarity despite
    both inputs having high quality.
    """
    if not isinstance(prediction, PositionPrediction):
        raise ValueError("prediction must be PositionPrediction")

    if not isinstance(candidate_position, Measurement):
        raise ValueError("candidate_position must be a Measurement")

    integer("target_stamp_ns", target_stamp_ns, minimum=1)

    if prediction.target_stamp_ns != target_stamp_ns:
        return QualityResult(
            0.0,
            "Prediction targets a different observation time",
        )

    if prediction.position is None or prediction.quality == 0.0:
        return QualityResult(
            0.0,
            prediction.reason,
        )

    if not candidate_position.usable:
        return QualityResult(
            0.0,
            candidate_position.reason
            or "Candidate position is unavailable or has zero quality",
        )

    if not isinstance(candidate_position.value, Vector3):
        raise ValueError("Candidate position must contain Vector3")

    if candidate_position.frame_id != "odom":
        return QualityResult(
            0.0,
            "Candidate position is not expressed in odom",
        )

    if candidate_position.stamp_ns != target_stamp_ns:
        return QualityResult(
            0.0,
            "Candidate position does not belong to the current observation",
        )

    return QualityResult(
        min(prediction.quality, candidate_position.quality),
        "Prediction and measured position are comparable in odom",
    )