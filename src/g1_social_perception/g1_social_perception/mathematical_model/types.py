"""
Mathematical contracts shared by perception, association and track memory.

Data flow:
    SourceDetection
        Actual YOLO observation D[k, j].

    EnrichedDetection
        Candidate D_tilde[k, j] containing:
        source detection, odom position, upper/lower appearance,
        apparent/metric extents and per-feature quality.

    CueScore
        Pairwise similarity S[r, i, j] and reliability q[r, i, j].

    AssociationWeights
        Nominal weights w[r] and coefficients used to combine
        appearance regions and size cues.

    TrackState
        Retained identity hypothesis updated after accepted association.
        Each feature preserves its own last measurement timestamp.

Responsibilities:
    Define values, units, provenance and mathematical validity.
    Reject inconsistent objects at construction.

Boundary:
    These classes do not extract features, calculate association scores,
    assign identities, publish ROS messages or confirm social clearance.
"""


from dataclasses import dataclass
from typing import Generic, Optional, Tuple, TypeVar

from .validation import (
    finite,
    positive,
    unit_interval,
    integer,
    nonempty,
    normalized,
)


T = TypeVar("T")


# ============================================================================
# GEOMETRIC AND APPEARANCE VALUES
# ============================================================================


@dataclass(frozen=True)
class Vector3:
    """
    Three finite components of a position or velocity.

    Meaning:
        Position: (x, y, z), in metres.
        Velocity: (vx, vy, vz), in metres/second.

    The enclosing Measurement supplies the coordinate frame and timestamp.
    This value alone does not establish observation freshness.
    """

    x: float
    y: float
    z: float

    def __post_init__(self):
        for name in ("x", "y", "z"):
            finite(name, getattr(self, name))


@dataclass(frozen=True)
class Extent2D:
    """
    Positive width and height used by the size similarity equations.

    Meaning:
        apparent_extent: (w, h), in pixels.
        metric_extent: (W, H), in metres.

    Metric extents describe the visible detection's estimated dimensions;
    they are not guaranteed anatomical body dimensions.

    Positive dimensions are required for logarithmic size ratios.
    """

    width: float
    height: float

    def __post_init__(self):
        positive("width", self.width)
        positive("height", self.height)


@dataclass(frozen=True)
class BoundingBox2D:
    """
    Image bounding box B = (u, v, w, h), entirely in pixels.

    Fields:
        center_u, center_v: box centre in image coordinates.
        width, height: positive box dimensions.

    Used for:
        Region extraction, apparent size and geometric comparisons.

    An observed box is not automatically a predicted box.
    The association stage must explicitly obtain any prediction it uses.
    """

    center_u: float
    center_v: float
    width: float
    height: float

    def __post_init__(self):
        finite("center_u", self.center_u)
        finite("center_v", self.center_v)
        positive("width", self.width)
        positive("height", self.height)


@dataclass(frozen=True)
class Histogram:
    """
    Normalized appearance descriptor for one body region.

    Equation:
        bins[b] >= 0
        sum(bins) = 1

    Fields:
        bins: normalized frequencies used by histogram similarity.
        schema_id: extraction definition, including binning conventions.

    Comparison requires compatible schemas and equal bin counts.
    Those pairwise checks belong to the appearance calculation.
    """

    bins: Tuple[float, ...]
    schema_id: str

    def __post_init__(self):
        # Own an immutable sequence, even when the caller supplies a list.
        object.__setattr__(self, "bins", tuple(self.bins))

        nonempty("schema_id", self.schema_id)

        normalized(
            "histogram bins",
            self.bins,
            tolerance=1e-6,
        )


# ============================================================================
# MEASUREMENT VALUE, QUALITY AND PROVENANCE
# ============================================================================


@dataclass(frozen=True)
class Measurement(Generic[T]):
    """
    One feature value with its quality and observation provenance.

    Fields:
        value:
            Geometric or appearance value; None when unavailable.

        quality:
            Feature reliability in [0, 1].
            Pairwise reliability is calculated later from both features.

        stamp_ns:
            Source observation reference time, in nanoseconds.
            A retained feature keeps this timestamp until actually updated.

        frame_id:
            Coordinate frame for geometry or source image frame
            for image-derived descriptors.

        reason:
            Explanation of unavailability or reduced reliability.

    Rules:
        Missing value -> quality zero, no timestamp and no frame.
        Present value with quality zero -> inspectable but unusable.

    Units:
        Defined by the containing field: pixels, metres or metres/second.

    Boundary:
        usable indicates availability and positive quality.
        Freshness and pair compatibility require additional checks.
    """

    value: Optional[T]
    quality: float
    stamp_ns: Optional[int]
    frame_id: Optional[str]
    reason: str = ""

    def __post_init__(self):
        unit_interval("measurement quality", self.quality)

        if not isinstance(self.reason, str):
            raise ValueError("Measurement reason must be a string")

        if self.value is None:
            if (
                self.quality != 0
                or self.stamp_ns is not None
                or self.frame_id is not None
            ):
                raise ValueError(
                    "Missing measurement requires zero quality "
                    "and no stamp/frame"
                )

        else:
            integer(
                "measurement stamp_ns",
                self.stamp_ns,
                minimum=1,
            )
            nonempty("measurement frame_id", self.frame_id)

    @property
    def usable(self):
        """Return availability with positive quality, without checking age."""
        return self.value is not None and self.quality > 0

    @classmethod
    def unavailable(cls, reason):
        """Represent missing evidence without inventing a value or timestamp."""
        return cls(
            value=None,
            quality=0.0,
            stamp_ns=None,
            frame_id=None,
            reason=reason,
        )


# ============================================================================
# CURRENT SOURCE OBSERVATION
# ============================================================================


@dataclass(frozen=True)
class SourceDetection:
    """
    Actual detector observation D_YOLO[k, j], before identity association.

    Fields:
        stamp_ns: source observation time t[k], in nanoseconds.
        image_frame: coordinate frame of the source image.
        candidate_index: index j within this source detection array.
        bbox: observed image box B[k, j], in pixels.
        confidence: detector confidence in [0, 1].
        class_id: detector class identifier.
        source_session_id: identifier of the source producer session.

    The candidate index identifies an entry in one array.
    It is not a persistent person identity.
    """

    stamp_ns: int
    image_frame: str
    candidate_index: int
    bbox: BoundingBox2D
    confidence: float
    class_id: int
    source_session_id: str

    def __post_init__(self):
        integer("source stamp_ns", self.stamp_ns, minimum=1)
        nonempty("image_frame", self.image_frame)
        nonempty("source_session_id", self.source_session_id)

        integer("candidate_index", self.candidate_index)
        integer("class_id", self.class_id)
        unit_interval("detector confidence", self.confidence)

        if not isinstance(self.bbox, BoundingBox2D):
            raise ValueError("Source bbox must be BoundingBox2D")


def validate_features(owner, latest_stamp):
    """
    Validate feature types, reference times and odom geometry.

    Receives:
        An enriched detection or retained track.
        Its latest actual observation timestamp, in nanoseconds.

    Checks:
        Every feature uses Measurement with the expected value type.
        No present feature is newer than the latest actual observation.
        Odom position and velocity are explicitly expressed in "odom".

    Retention rule:
        Older feature timestamps are allowed here.
        A new position must not make an old appearance template recent.

    Returns:
        None when valid; raises ValueError on an inconsistent feature.
        Does not modify values, qualities or timestamps.
    """
    expected = {
        "position_odom": Vector3,
        "appearance_upper": Histogram,
        "appearance_lower": Histogram,
        "apparent_extent": Extent2D,
        "metric_extent": Extent2D,
    }

    if hasattr(owner, "velocity_odom"):
        expected["velocity_odom"] = Vector3

    if hasattr(owner, "bbox_reference"):
        expected["bbox_reference"] = BoundingBox2D

    for name, value_type in expected.items():
        measurement = getattr(owner, name)

        if not isinstance(measurement, Measurement):
            raise ValueError(name + " must be a Measurement")

        if measurement.value is None:
            continue

        if not isinstance(measurement.value, value_type):
            raise ValueError("Invalid value type for " + name)

        if measurement.stamp_ns > latest_stamp:
            raise ValueError(
                name + " cannot be newer than the last observation"
            )

        if (
            name in ("position_odom", "velocity_odom")
            and measurement.frame_id != "odom"
        ):
            raise ValueError(name + " must be expressed in odom")


@dataclass(frozen=True)
class EnrichedDetection:
    """
    Enriched candidate D_tilde[k, j] before track assignment.

    Mathematical correspondence:
        D_YOLO       -> source
        p_odom       -> position_odom
        a_upper      -> appearance_upper
        a_lower      -> appearance_lower
        e_apparent   -> apparent_extent
        e_metric     -> metric_extent
        Q            -> availability, quality and provenance of each feature

    Receives:
        Features extracted or derived for the current source observation.

    Provides:
        An explicit candidate for comparison against retained tracks.

    Rules:
        Every present feature references the current source timestamp.
        Unavailable depth or appearance remains explicitly unavailable.
        Historical templates belong to TrackState, not this candidate.

    The source timestamp identifies the reference observation.
    Raw sensor synchronization must be validated before construction.
    """

    source: SourceDetection

    position_odom: Measurement[Vector3]

    appearance_upper: Measurement[Histogram]
    appearance_lower: Measurement[Histogram]

    apparent_extent: Measurement[Extent2D]
    metric_extent: Measurement[Extent2D]

    def __post_init__(self):
        if not isinstance(self.source, SourceDetection):
            raise ValueError("source must be SourceDetection")

        validate_features(self, self.source.stamp_ns)

        # Current evidence cannot borrow the timestamp of an old template.
        for name in (
            "position_odom",
            "appearance_upper",
            "appearance_lower",
            "apparent_extent",
            "metric_extent",
        ):
            measurement = getattr(self, name)

            if (
                measurement.value is not None
                and measurement.stamp_ns != self.source.stamp_ns
            ):
                raise ValueError(
                    "Candidate features must reference "
                    "the same source observation"
                )

        # Image descriptors remain associated with their source image frame.
        for name in (
            "appearance_upper",
            "appearance_lower",
            "apparent_extent",
        ):
            measurement = getattr(self, name)

            if (
                measurement.value is not None
                and measurement.frame_id != self.source.image_frame
            ):
                raise ValueError(
                    name + " must refer to the source image frame"
                )

        # Apparent size is the observed bbox size, not a predicted extent.
        extent = self.apparent_extent.value

        if extent is not None and (
            extent.width != self.source.bbox.width
            or extent.height != self.source.bbox.height
        ):
            raise ValueError(
                "Apparent extent must match the actual source bbox"
            )


# ============================================================================
# RETAINED IDENTITY AND OBSERVATION REFERENCES
# ============================================================================


@dataclass(frozen=True)
class IdentityKey:
    """
    Persistent tracking key: (tracking session, person ID).

    The session distinguishes identical person labels across tracker restarts.
    This key identifies a tracking hypothesis, not a verified human identity.

    It is independent of q[r, i, j], the pairwise reliability coefficient.
    """

    session_id: str
    person_id: str

    def __post_init__(self):
        nonempty("identity session_id", self.session_id)
        nonempty("person_id", self.person_id)


@dataclass(frozen=True)
class TrackState:
    """
    Retained tracking hypothesis T[i] after accepted observations.

    Identity:
        key identifies the hypothesis.
        generation identifies its current continuity segment.
        A generation change tells downstream motion processing to reset.

    Observation history:
        last_observation_ns is the latest actual accepted observation.
        observation_count is n: total accepted observations for this hypothesis.
        consecutive_hits counts successive associations for confirmation.
        confirmed records the confirmation policy's result.

    Retained features:
        bbox_reference: last retained actual image box.
        position_odom: last retained measured position, in metres.
        velocity_odom: velocity estimate from accepted observations, in m/s.
        appearance_upper/lower: retained regional appearance descriptors.
        apparent_extent: retained image dimensions, in pixels.
        metric_extent: retained estimated visible dimensions, in metres.

    Memory rule:
        Features have independent timestamps.
        Updating one feature does not renew the others.

    Prediction rule:
        Predicted positions and boxes are calculated separately.
        Prediction alone never increments n or renews observation time.

    Updates create a new immutable state after association is accepted.
    """

    key: IdentityKey
    generation: int
    last_observation_ns: int

    observation_count: int
    consecutive_hits: int
    confirmed: bool

    bbox_reference: Measurement[BoundingBox2D]

    position_odom: Measurement[Vector3]
    velocity_odom: Measurement[Vector3]

    appearance_upper: Measurement[Histogram]
    appearance_lower: Measurement[Histogram]

    apparent_extent: Measurement[Extent2D]
    metric_extent: Measurement[Extent2D]

    def __post_init__(self):
        if not isinstance(self.key, IdentityKey):
            raise ValueError("Track key must be IdentityKey")

        integer("generation", self.generation, minimum=1)

        integer(
            "last_observation_ns",
            self.last_observation_ns,
            minimum=1,
        )

        integer(
            "observation_count",
            self.observation_count,
            minimum=1,
        )

        integer("consecutive_hits", self.consecutive_hits)

        if (
            self.consecutive_hits > self.observation_count
            or type(self.confirmed) is not bool
        ):
            raise ValueError("Invalid track confirmation state")

        validate_features(self, self.last_observation_ns)


# ============================================================================
# PAIRWISE EVIDENCE AND FUSION PARAMETERS
# ============================================================================


@dataclass(frozen=True)
class CueScore:
    """
    One association cue for track i and candidate j.

    Fields:
        score: similarity S[r, i, j] in [0, 1], or None if unavailable.
        quality: pairwise reliability q[r, i, j] in [0, 1].

    Examples:
        CueScore(None, 0.0): no usable comparison.
        CueScore(0.0, 1.0): reliable comparison showing disagreement.
        CueScore(0.9, 0.5): strong similarity with reduced reliability.

    Fusion:
        Contribution numerator = w[r] * quality * score
        Contribution denominator = w[r] * quality

    A zero-quality cue is excluded, even if a score remains inspectable.
    """

    score: Optional[float]
    quality: float

    def __post_init__(self):
        unit_interval("pair quality", self.quality)

        if self.score is None:
            if self.quality != 0:
                raise ValueError(
                    "Unavailable score must have zero quality"
                )
        else:
            unit_interval("similarity", self.score)


@dataclass(frozen=True)
class AssociationWeights:
    """
    Configured coefficients used by association-score equations.

    Nominal cue weights:
        g: geometric overlap.
        m: motion consistency.
        a: appearance similarity.
        h: size similarity.

        g, m, a, h >= 0
        g + m + a + h = 1

    Appearance coefficients:
        eta_upper > 0
        eta_lower > 0

        Combine upper/lower similarities using their pair qualities.

    Size coefficients:
        alpha > 0: apparent-size contribution.
        beta > 0: metric-size contribution.

        Combine apparent/metric similarities using their pair qualities.

    Effective weights:
        Calculated separately for every track/candidate pair:
            effective_w[r] = w[r] * q[r] / sum(w[l] * q[l])

        When the denominator is zero, association has no usable score.

    No default coefficients are imposed here.
    Their values must come from the chosen model configuration.
    """

    g: float
    m: float
    a: float
    h: float

    eta_upper: float
    eta_lower: float

    alpha: float
    beta: float

    def __post_init__(self):
        normalized(
            "nominal weights",
            (self.g, self.m, self.a, self.h),
        )

        for name in (
            "eta_upper",
            "eta_lower",
            "alpha",
            "beta",
        ):
            positive(name, getattr(self, name))