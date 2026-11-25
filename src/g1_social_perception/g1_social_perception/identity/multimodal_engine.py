
"""
Single multimodal identity authority for G1.

Inputs:
    Actual EnrichedDetection observations.
    Optional synchronised CurrentView.

Cues:
    g: predicted image-box IoU
    m: predicted odom motion
    a: upper/lower appearance
    h: metric size and conditionally apparent size

Rules:
    - w=0, q=0 or score=None excludes a cue.
    - A genuine score=0 is contradictory evidence.
    - When g participates, apparent size is excluded from h.
    - No prediction becomes an actual observation.
    - Missing measurements never renew retained evidence.
    - Global one-to-one assignment determines identities.
    - Dormant return uses two eligible appearance regions.
    - Batch processing is transactional.
"""

import copy
import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

from ..mathematical_model.types import (
    AssociationWeights,
    CueScore,
    EnrichedDetection,
    IdentityKey,
    Measurement,
    TrackState,
    Vector3,
)
from ..mathematical_model.prediction import predict_position_odom
from ..mathematical_model.quality import (
    feature_quality_at,
    measurement_pair_quality,
    motion_pair_quality,
)
from ..mathematical_model.association_scores import (
    appearance_score,
    association_score,
    geometric_score,
    histogram_similarity,
    motion_score,
    size_score,
    size_similarity,
)
from ..mathematical_model.assignment import (
    AssignmentThresholds,
    PairEvidence,
    assign_pairs,
)
from ..mathematical_model.image_prediction import (
    CurrentView,
    ReferenceView,
    predict_image_box,
    reference_from_current,
)


# ============================================================================
# DATA / CONSTANTS
# ============================================================================

EMPTY_CUE = CueScore(None, 0.0)
CUE_NAMES = ("g", "m", "a", "h")

ACTIVE_MODES = frozenset(
    "+".join(
        name
        for index, name in enumerate(CUE_NAMES)
        if mask & (1 << index)
    )
    for mask in range(1, 16)
)


def _positive(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"INVALID_POSITIVE_PARAMETER:{name}")


def _unit(name, value, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        or (positive and value == 0)
    ):
        raise ValueError(f"INVALID_UNIT_PARAMETER:{name}")


def _pair_scales(name, values):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"INVALID_SCALE_PAIR:{name}")

    for value in values:
        _positive(name, value)


def _usable_cue(cue, weight):
    return (
        weight > 0.0
        and cue.score is not None
        and cue.quality > 0.0
    )


# ============================================================================
# DATA / CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class EngineConfig:
    weights: AssociationWeights
    active_thresholds: Mapping[str, AssignmentThresholds]

    motion_tolerance_m: float
    maximum_velocity_age_sec: float

    metric_sigma: Tuple[float, float]
    apparent_sigma: Tuple[float, float]

    maximum_view_angle_rad: float
    minimum_projection_depth_m: float
    maximum_projection_depth_m: float

    active_timeout_sec: float
    gallery_retention_sec: float

    confirm_hits: int
    max_active_tracks: int
    max_gallery_entries: int

    minimum_initial_confidence: float
    minimum_initial_region_quality: float

    minimum_return_score: float
    minimum_return_support: float
    minimum_return_margin: float

    stationary_fallback_quality: float

    def __post_init__(self):
        if not isinstance(self.weights, AssociationWeights):
            raise ValueError("INVALID_ASSOCIATION_WEIGHTS")

        if (
            not isinstance(self.active_thresholds, Mapping)
            or set(self.active_thresholds) != ACTIVE_MODES
        ):
            raise ValueError("EXPECTED_15_ASSOCIATION_MODES")

        for policy in self.active_thresholds.values():
            if not isinstance(policy, AssignmentThresholds):
                raise ValueError("INVALID_ASSIGNMENT_POLICY")

        for name in (
            "motion_tolerance_m",
            "maximum_velocity_age_sec",
            "active_timeout_sec",
            "gallery_retention_sec",
            "minimum_projection_depth_m",
            "maximum_projection_depth_m",
        ):
            _positive(name, getattr(self, name))

        _pair_scales("metric_sigma", self.metric_sigma)
        _pair_scales("apparent_sigma", self.apparent_sigma)

        if self.gallery_retention_sec <= self.active_timeout_sec:
            raise ValueError("GALLERY_RETENTION_TOO_SHORT")

        if (
            self.maximum_projection_depth_m
            <= self.minimum_projection_depth_m
        ):
            raise ValueError("INVALID_PROJECTION_DEPTH_INTERVAL")

        if (
            not math.isfinite(self.maximum_view_angle_rad)
            or not 0 < self.maximum_view_angle_rad <= math.pi
        ):
            raise ValueError("INVALID_VIEW_ANGLE_LIMIT")

        for name in (
            "confirm_hits",
            "max_active_tracks",
            "max_gallery_entries",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"INVALID_INTEGER_PARAMETER:{name}")

        for name in (
            "minimum_initial_confidence",
            "minimum_initial_region_quality",
            "minimum_return_score",
            "minimum_return_support",
            "minimum_return_margin",
        ):
            _unit(name, getattr(self, name), positive=True)

        _unit(
            "stationary_fallback_quality",
            self.stationary_fallback_quality,
        )

        if self.stationary_fallback_quality >= 1:
            raise ValueError("INVALID_STATIONARY_FALLBACK_SCALE")


# ============================================================================
# DATA / RESULTS AND RETAINED STATE
# ============================================================================

@dataclass(frozen=True)
class IdentityDecision:
    person_id: str = ""
    state: str = "INSUFFICIENT"
    confirmed: bool = False
    generation: int = 0
    score: Optional[float] = None

    def __post_init__(self):
        valid = {
            "NEW",
            "MATCHED",
            "RETURNING",
            "AMBIGUOUS",
            "INSUFFICIENT",
        }

        if self.state not in valid:
            raise ValueError("INVALID_IDENTITY_STATE")

        if self.state in {"AMBIGUOUS", "INSUFFICIENT"}:
            if self.person_id or self.confirmed or self.generation:
                raise ValueError("UNRESOLVED_IDENTITY_HAS_KEY")
        elif not self.person_id or self.generation < 1:
            raise ValueError("ASSIGNED_IDENTITY_MISSING_KEY")

        if self.score is not None:
            _unit("decision_score", self.score)


@dataclass(frozen=True)
class TrackRecord:
    track: TrackState

    last_observation_receipt: float
    upper_receipt: Optional[float]
    lower_receipt: Optional[float]

    metric_reference_view: Optional[ReferenceView]

    def __post_init__(self):
        if not isinstance(self.track, TrackState):
            raise ValueError("INVALID_TRACK_STATE")

        if (
            not math.isfinite(self.last_observation_receipt)
            or self.last_observation_receipt < 0
        ):
            raise ValueError("INVALID_LAST_OBSERVATION_RECEIPT")

        for measurement, receipt in (
            (self.track.appearance_upper, self.upper_receipt),
            (self.track.appearance_lower, self.lower_receipt),
        ):
            if measurement.usable:
                if (
                    receipt is None
                    or not math.isfinite(receipt)
                    or receipt < 0
                    or receipt > self.last_observation_receipt
                ):
                    raise ValueError("INVALID_APPEARANCE_RECEIPT")
            elif receipt is not None:
                raise ValueError("RECEIPT_WITHOUT_APPEARANCE")

        if self.metric_reference_view is not None:
            view = self.metric_reference_view
            extent = self.track.metric_extent

            if (
                not extent.usable
                or extent.stamp_ns != view.stamp_ns
                or extent.frame_id != view.camera_frame
            ):
                raise ValueError("INVALID_METRIC_VIEW_PROVENANCE")


ActiveRecord = TrackRecord
DormantRecord = TrackRecord


# ============================================================================
# PROCESSING / COMMON EVIDENCE HELPERS
# ============================================================================

def _weights(config):
    w = config.weights
    return {
        "g": w.g,
        "m": w.m,
        "a": w.a,
        "h": w.h,
    }


def _retain(previous, current):
    """Never replace a real feature by an unavailable measurement."""
    if not current.usable:
        return previous

    if (
        previous.usable
        and current.stamp_ns is not None
        and previous.stamp_ns is not None
        and current.stamp_ns < previous.stamp_ns
    ):
        return previous

    return current


def _predict(track, stamp_ns, config):
    return predict_position_odom(
        track=track,
        target_stamp_ns=stamp_ns,
        maximum_horizon_sec=config.active_timeout_sec,
        maximum_velocity_age_sec=config.maximum_velocity_age_sec,
        constant_position_quality_scale=(
            config.stationary_fallback_quality
        ),
    )


# ============================================================================
# PROCESSING / APPEARANCE
# ============================================================================

def appearance_pair(track, candidate, config, maximum_age_sec):
    regions = []

    for name in ("appearance_upper", "appearance_lower"):
        reference = getattr(track, name)
        current = getattr(candidate, name)

        quality = measurement_pair_quality(
            reference=reference,
            candidate=current,
            target_stamp_ns=candidate.source.stamp_ns,
            maximum_reference_age_sec=maximum_age_sec,
        )

        if not quality.usable:
            regions.append(EMPTY_CUE)
            continue

        regions.append(
            histogram_similarity(
                reference=reference.value.bins,
                candidate=current.value.bins,
                q_reference=quality.value,
                q_candidate=quality.value,
            )
        )

    combined, _ = appearance_score(
        upper=regions[0],
        lower=regions[1],
        eta_upper=config.weights.eta_upper,
        eta_lower=config.weights.eta_lower,
    )

    return combined, tuple(regions)


# ============================================================================
# PROCESSING / MOTION
# ============================================================================

def motion_pair(track, candidate, config, prediction):
    quality = motion_pair_quality(
        prediction=prediction,
        candidate_position=candidate.position_odom,
        target_stamp_ns=candidate.source.stamp_ns,
    )

    if not quality.usable:
        return EMPTY_CUE

    return motion_score(
        predicted_position=prediction.position,
        candidate_position=candidate.position_odom.value,
        distance_tolerance_m=config.motion_tolerance_m,
        pair_quality=quality.value,
    )


# ============================================================================
# PROCESSING / METRIC SIZE
# ============================================================================

def metric_pair(track, candidate, config):
    quality = measurement_pair_quality(
        reference=track.metric_extent,
        candidate=candidate.metric_extent,
        target_stamp_ns=candidate.source.stamp_ns,
        maximum_reference_age_sec=config.active_timeout_sec,
    )

    if not quality.usable:
        return EMPTY_CUE

    old = track.metric_extent.value
    new = candidate.metric_extent.value

    return size_similarity(
        candidate=(new.width, new.height),
        reference=(old.width, old.height),
        sigma=config.metric_sigma,
        pair_quality=quality.value,
    )


# ============================================================================
# PROCESSING / GEOMETRY AND APPARENT SIZE
# ============================================================================

def projected_image_cues(box_prediction, candidate, config):
    if not box_prediction.usable:
        return EMPTY_CUE, EMPTY_CUE

    predicted = box_prediction.bbox
    observed = candidate.source.bbox

    quality = min(
        box_prediction.quality,
        candidate.source.confidence,
    )

    if quality <= 0:
        return EMPTY_CUE, EMPTY_CUE

    geometry = geometric_score(
        predicted_bbox=predicted,
        candidate_bbox=observed,
        pair_quality=quality,
    )

    apparent = size_similarity(
        candidate=(observed.width, observed.height),
        reference=(predicted.width, predicted.height),
        sigma=config.apparent_sigma,
        pair_quality=quality,
    )

    return geometry, apparent


# ============================================================================
# PROCESSING / MULTIMODAL PAIR
# ============================================================================

def active_pair(record, candidate, config, current_view):
    """
    Calculate all eligible cues for one track/candidate pair.

    Correlation policy:
        If g actually participates, apparent size from its
        predicted box is excluded from h.

    No missing evidence is replaced by an artificial score.
    """

    track = record.track
    stamp_ns = candidate.source.stamp_ns

    weights = _weights(config)

    prediction = _predict(
        track,
        stamp_ns,
        config,
    )

    geometry = EMPTY_CUE
    apparent = EMPTY_CUE

    # Compute projection only if at least one consumer is enabled.
    projection_needed = (
        weights["g"] > 0
        or (weights["h"] > 0 and config.weights.alpha > 0)
    )

    if projection_needed:
        box_prediction = predict_image_box(
            position_prediction=prediction,
            metric_extent=track.metric_extent,
            reference_view=record.metric_reference_view,
            current_view=current_view,
            maximum_extent_age_sec=config.active_timeout_sec,
            maximum_view_angle_rad=config.maximum_view_angle_rad,
            minimum_depth_m=config.minimum_projection_depth_m,
            maximum_depth_m=config.maximum_projection_depth_m,
        )

        geometry, apparent = projected_image_cues(
            box_prediction,
            candidate,
            config,
        )

    motion = motion_pair(
        track,
        candidate,
        config,
        prediction,
    )

    appearance, _ = appearance_pair(
        track,
        candidate,
        config,
        config.active_timeout_sec,
    )

    metric = metric_pair(
        track,
        candidate,
        config,
    )

    # Do not count the same predicted-image size through
    # both geometric IoU and apparent-size similarity.
    geometry_participates = _usable_cue(
        geometry,
        weights["g"],
    )

    apparent_for_size = (
        EMPTY_CUE if geometry_participates else apparent
    )

    size, _ = size_score(
        apparent=apparent_for_size,
        metric=metric,
        alpha=config.weights.alpha,
        beta=config.weights.beta,
    )

    cues = {
        "g": geometry,
        "m": motion,
        "a": appearance,
        "h": size,
    }

    # Association scores handle w*q normalization.
    fused = association_score(
        cues=cues,
        weights=weights,
    )

    mode = "+".join(
        name
        for name in CUE_NAMES
        if _usable_cue(cues[name], weights[name])
    )

    return fused, mode, cues


# ============================================================================
# PROCESSING / TRACK INITIALIZATION
# ============================================================================

def initial_eligible(candidate, config):
    confidence = candidate.source.confidence

    if (
        confidence <= 0
        or confidence < config.minimum_initial_confidence
    ):
        return False

    if candidate.position_odom.usable:
        return True

    return all(
        getattr(candidate, name).usable
        and getattr(candidate, name).quality
        >= config.minimum_initial_region_quality
        for name in ("appearance_upper", "appearance_lower")
    )


def create_track(
    candidate,
    key,
    *,
    generation=1,
    previous_count=0,
    confirmed=False,
):
    source = candidate.source

    if source.confidence <= 0:
        raise ValueError("INVALID_INITIAL_SOURCE_CONFIDENCE")

    return TrackState(
        key=key,
        generation=generation,
        last_observation_ns=source.stamp_ns,
        observation_count=previous_count + 1,
        consecutive_hits=1,
        confirmed=confirmed,
        bbox_reference=Measurement(
            value=source.bbox,
            quality=source.confidence,
            stamp_ns=source.stamp_ns,
            frame_id=source.image_frame,
            reason="ACTUAL_YOLO_OBSERVATION",
        ),
        position_odom=candidate.position_odom,
        velocity_odom=Measurement.unavailable(
            "VELOCITY_NOT_ESTIMATED"
        ),
        appearance_upper=candidate.appearance_upper,
        appearance_lower=candidate.appearance_lower,
        apparent_extent=candidate.apparent_extent,
        metric_extent=candidate.metric_extent,
    )



def _metric_view(candidate, current_view):
    """
    Associate a retained metric extent with the RGB camera view
    that was used to construct the source detection.

    The actual depth acquisition timestamp remains separately
    available through DepthEvidence at the ROS boundary.
    """

    if current_view is None:
        return None

    extent = candidate.metric_extent

    if not extent.usable:
        return None

    # The fused extent belongs to this RGB observation.
    if extent.stamp_ns != candidate.source.stamp_ns:
        return None

    if current_view.stamp_ns != candidate.source.stamp_ns:
        return None

    # Metric extent remains associated with the original
    # optical image frame, even after position is in odom.
    if extent.frame_id != candidate.source.image_frame:
        return None

    if current_view.camera_frame != candidate.source.image_frame:
        return None

    if current_view.quality <= 0.0:
        return None

    return reference_from_current(current_view)

def _new_record(track, receipt, candidate, current_view):
    return TrackRecord(
        track=track,
        last_observation_receipt=receipt,
        upper_receipt=(
            receipt if track.appearance_upper.usable else None
        ),
        lower_receipt=(
            receipt if track.appearance_lower.usable else None
        ),
        metric_reference_view=_metric_view(
            candidate,
            current_view,
        ),
    )


# ============================================================================
# PROCESSING / ACCEPTED TRACK UPDATE
# ============================================================================

def update_track(track, candidate, config):
    now_ns = candidate.source.stamp_ns

    if now_ns <= track.last_observation_ns:
        raise ValueError("NON_INCREASING_TRACK_OBSERVATION")

    old_position = track.position_odom
    new_position = candidate.position_odom

    velocity = track.velocity_odom

    if (
        old_position.usable
        and new_position.usable
        and old_position.frame_id == "odom"
        and new_position.frame_id == "odom"
        and old_position.stamp_ns < now_ns
        and new_position.stamp_ns == now_ns
    ):
        dt = (now_ns - old_position.stamp_ns) / 1e9

        p0 = old_position.value
        p1 = new_position.value

        velocity = Measurement(
            value=Vector3(
                (p1.x - p0.x) / dt,
                (p1.y - p0.y) / dt,
                (p1.z - p0.z) / dt,
            ),
            quality=min(
                old_position.quality,
                new_position.quality,
            ),
            stamp_ns=now_ns,
            frame_id="odom",
            reason="TWO_ACCEPTED_ODOM_POSITIONS",
        )

    # If a newer position has been accepted without an
    # eligible velocity estimate, do not retain a velocity
    # inconsistent with the position timestamp.
    elif (
        new_position.usable
        and new_position.stamp_ns == now_ns
    ):
        velocity = Measurement.unavailable(
            "VELOCITY_NOT_ESTIMATED_FOR_NEW_POSITION"
        )

    hits = track.consecutive_hits + 1

    return replace(
        track,
        last_observation_ns=now_ns,
        observation_count=track.observation_count + 1,
        consecutive_hits=hits,
        confirmed=(
            track.confirmed or hits >= config.confirm_hits
        ),
        bbox_reference=Measurement(
            value=candidate.source.bbox,
            quality=candidate.source.confidence,
            stamp_ns=now_ns,
            frame_id=candidate.source.image_frame,
            reason="ACTUAL_ASSOCIATED_YOLO_OBSERVATION",
        ),
        position_odom=_retain(
            track.position_odom,
            candidate.position_odom,
        ),
        velocity_odom=velocity,
        appearance_upper=_retain(
            track.appearance_upper,
            candidate.appearance_upper,
        ),
        appearance_lower=_retain(
            track.appearance_lower,
            candidate.appearance_lower,
        ),
        apparent_extent=_retain(
            track.apparent_extent,
            candidate.apparent_extent,
        ),
        metric_extent=_retain(
            track.metric_extent,
            candidate.metric_extent,
        ),
    )


def _updated_record(
    previous,
    updated,
    candidate,
    receipt,
    current_view,
):
    metric_view = previous.metric_reference_view

    # Update orientation only if a newer metric extent
    # was actually retained.
    metric_changed = (
        candidate.metric_extent.usable
        and updated.metric_extent == candidate.metric_extent
        and (
            not previous.track.metric_extent.usable
            or candidate.metric_extent.stamp_ns
            > previous.track.metric_extent.stamp_ns
        )
    )

    if metric_changed:
        metric_view = _metric_view(candidate, current_view)

    upper_changed = (
        candidate.appearance_upper.usable
        and updated.appearance_upper == candidate.appearance_upper
        and (
            not previous.track.appearance_upper.usable
            or candidate.appearance_upper.stamp_ns
            > previous.track.appearance_upper.stamp_ns
        )
    )

    lower_changed = (
        candidate.appearance_lower.usable
        and updated.appearance_lower == candidate.appearance_lower
        and (
            not previous.track.appearance_lower.usable
            or candidate.appearance_lower.stamp_ns
            > previous.track.appearance_lower.stamp_ns
        )
    )

    return TrackRecord(
        track=updated,
        last_observation_receipt=receipt,
        upper_receipt=(
            receipt if upper_changed else previous.upper_receipt
        ),
        lower_receipt=(
            receipt if lower_changed else previous.lower_receipt
        ),
        metric_reference_view=metric_view,
    )


# ============================================================================
# PROCESSING / DORMANT GALLERY
# ============================================================================

def _within_age(
    source_ns,
    receipt,
    feature_ns,
    feature_receipt,
    maximum_age_sec,
):
    if feature_ns is None or feature_receipt is None:
        return False

    source_age = (source_ns - feature_ns) / 1e9
    receipt_age = receipt - feature_receipt

    return (
        0 <= source_age <= maximum_age_sec
        and 0 <= receipt_age <= maximum_age_sec
    )


def _gallery_eligible(
    record,
    source_ns,
    receipt,
    config,
):
    track = record.track

    if not track.confirmed:
        return False

    if not _within_age(
        source_ns,
        receipt,
        track.last_observation_ns,
        record.last_observation_receipt,
        config.gallery_retention_sec,
    ):
        return False

    for measurement, feature_receipt in (
        (track.appearance_upper, record.upper_receipt),
        (track.appearance_lower, record.lower_receipt),
    ):
        if (
            not measurement.usable
            or measurement.quality
            < config.minimum_initial_region_quality
        ):
            return False

        quality = feature_quality_at(
            measurement,
            source_ns,
            config.gallery_retention_sec,
        )

        if not quality.usable:
            return False

        if not _within_age(
            source_ns,
            receipt,
            measurement.stamp_ns,
            feature_receipt,
            config.gallery_retention_sec,
        ):
            return False

    return True


def _return_pair(
    record,
    candidate,
    source_ns,
    receipt,
    config,
):
    if not _gallery_eligible(
        record,
        source_ns,
        receipt,
        config,
    ):
        return None

    for name in ("appearance_upper", "appearance_lower"):
        measurement = getattr(candidate, name)

        if (
            not measurement.usable
            or measurement.quality
            < config.minimum_initial_region_quality
        ):
            return None

    appearance, regions = appearance_pair(
        record.track,
        candidate,
        config,
        config.gallery_retention_sec,
    )

    if not all(
        region.score is not None
        and region.quality
        >= config.minimum_initial_region_quality
        for region in regions
    ):
        return None

    return appearance if appearance.score is not None else None


# ============================================================================
# PROCESSING / TRANSACTIONAL IDENTITY AUTHORITY
# ============================================================================

class MultimodalIdentityEngine:

    def __init__(self, session_id, config):
        if (
            not isinstance(session_id, str)
            or not session_id.strip()
        ):
            raise ValueError("INVALID_TRACKING_SESSION")

        if not isinstance(config, EngineConfig):
            raise ValueError("INVALID_ENGINE_CONFIG")

        self.session_id = session_id
        self.config = config

        self.active = {}
        self.dormant = {}

        self.next_id = 1

        self.source_session_id = None
        self.last_source_ns = 0
        self.last_receipt = None

    # ------------------------------------------------------------------------
    # INPUT / ATOMIC VALIDATION
    # ------------------------------------------------------------------------

    def process(
        self,
        candidates,
        *,
        stamp_ns,
        receipt,
        current_view: Optional[CurrentView],
    ):
        if (
            type(stamp_ns) is not int
            or stamp_ns <= self.last_source_ns
        ):
            raise ValueError("DUPLICATE_OR_REVERSED_SOURCE")

        if (
            isinstance(receipt, bool)
            or not isinstance(receipt, (int, float))
            or not math.isfinite(receipt)
            or receipt < 0
            or (
                self.last_receipt is not None
                and receipt < self.last_receipt
            )
        ):
            raise ValueError("INVALID_MONOTONIC_RECEIPT")

        if current_view is not None:
            if not isinstance(current_view, CurrentView):
                raise ValueError("INVALID_CURRENT_VIEW")

            if current_view.stamp_ns != stamp_ns:
                raise ValueError("CAMERA_TIMESTAMP_MISMATCH")

        candidates = tuple(candidates)
        source_session = self.source_session_id
        indices = set()

        for candidate in candidates:
            if not isinstance(candidate, EnrichedDetection):
                raise ValueError("INVALID_ENRICHED_DETECTION")

            source = candidate.source

            if source.stamp_ns != stamp_ns:
                raise ValueError("MIXED_OBSERVATION_TIMESTAMPS")

            if source_session is None:
                source_session = source.source_session_id

            elif source.source_session_id != source_session:
                raise ValueError("MIXED_SOURCE_SESSIONS")

            if source.candidate_index in indices:
                raise ValueError("DUPLICATE_SOURCE_INDEX")

            indices.add(source.candidate_index)

            if (
                current_view is not None
                and current_view.camera_frame != source.image_frame
            ):
                raise ValueError("CAMERA_FRAME_MISMATCH")

        staged = copy.deepcopy(self)

        decisions = staged._process(
            candidates,
            stamp_ns,
            float(receipt),
            current_view,
        )

        staged.last_source_ns = stamp_ns
        staged.last_receipt = float(receipt)
        staged.source_session_id = source_session

        staged._validate_state()

        self.__dict__.update(staged.__dict__)

        return decisions

    # ------------------------------------------------------------------------
    # DATA / IDENTITY ALLOCATION
    # ------------------------------------------------------------------------

    def _new_key(self):
        key = IdentityKey(
            session_id=self.session_id,
            person_id=f"person_{self.next_id}",
        )
        self.next_id += 1
        return key

    # ------------------------------------------------------------------------
    # PROCESSING / EXPIRATION WITHOUT EVIDENCE REFRESH
    # ------------------------------------------------------------------------

    def _retire(self, stamp_ns, receipt):
        expired = []

        for key, record in self.active.items():
            age_source = (
                stamp_ns - record.track.last_observation_ns
            ) / 1e9

            age_receipt = (
                receipt - record.last_observation_receipt
            )

            if (
                age_source > self.config.active_timeout_sec
                or age_receipt > self.config.active_timeout_sec
            ):
                expired.append(key)

        for key in expired:
            record = self.active.pop(key)

            if _gallery_eligible(
                record,
                stamp_ns,
                receipt,
                self.config,
            ):
                self.dormant[key] = record

        self.dormant = {
            key: record
            for key, record in self.dormant.items()
            if _gallery_eligible(
                record,
                stamp_ns,
                receipt,
                self.config,
            )
        }

        if len(self.dormant) > self.config.max_gallery_entries:
            ordered = sorted(
                self.dormant,
                key=lambda key: (
                    self.dormant[key].last_observation_receipt,
                    self.dormant[key].track.last_observation_ns,
                    key.person_id,
                ),
            )

            overflow = (
                len(self.dormant)
                - self.config.max_gallery_entries
            )

            for key in ordered[:overflow]:
                del self.dormant[key]

    # ------------------------------------------------------------------------
    # PROCESSING / ACTIVE ASSOCIATION
    # ------------------------------------------------------------------------

    def _active_assignment(self, candidates, current_view):
        records = [
            self.active[key]
            for key in sorted(
                self.active,
                key=lambda item: item.person_id,
            )
        ]

        pairs = []

        for i, record in enumerate(records):
            for j, candidate in enumerate(candidates):
                fused, mode, cues = active_pair(
                    record,
                    candidate,
                    self.config,
                    current_view,
                )

                if not mode:
                    continue

                # A motion contradiction can reject the edge
                # only when motion is enabled and actually usable.
                m_active = _usable_cue(
                    cues["m"],
                    self.config.weights.m,
                )

                hard_gate = not (
                    m_active
                    and cues["m"].score == 0.0
                )

                pairs.append(
                    PairEvidence(
                        track_index=i,
                        candidate_index=j,
                        score=fused["score"],
                        support=fused["support"],
                        mode=mode,
                        hard_gate_passed=hard_gate,
                    )
                )

        assignment = assign_pairs(
            track_count=len(records),
            candidate_count=len(candidates),
            pairs=pairs,
            thresholds_by_mode=self.config.active_thresholds,
        )

        return records, pairs, assignment

    def _admissible(self, edge):
        policy = self.config.active_thresholds[edge.mode]

        return (
            edge.hard_gate_passed
            and edge.score is not None
            and edge.score >= policy.minimum_score
            and edge.support >= policy.minimum_support
        )

    # ------------------------------------------------------------------------
    # PROCESSING / DORMANT GLOBAL ASSOCIATION
    # ------------------------------------------------------------------------

    def _dormant_assignment(
        self,
        candidates,
        free_indices,
        stamp_ns,
        receipt,
    ):
        records = [
            self.dormant[key]
            for key in sorted(
                self.dormant,
                key=lambda item: item.person_id,
            )
        ]

        threshold = AssignmentThresholds(
            minimum_score=self.config.minimum_return_score,
            minimum_support=self.config.minimum_return_support,
            minimum_margin=self.config.minimum_return_margin,
        )

        pairs = []

        # j always indexes the original complete observation.
        for i, record in enumerate(records):
            for j in free_indices:
                cue = _return_pair(
                    record,
                    candidates[j],
                    stamp_ns,
                    receipt,
                    self.config,
                )

                if cue is None:
                    continue

                pairs.append(
                    PairEvidence(
                        track_index=i,
                        candidate_index=j,
                        score=cue.score,
                        support=cue.quality,
                        mode="return",
                        hard_gate_passed=True,
                    )
                )

        assignment = assign_pairs(
            track_count=len(records),
            candidate_count=len(candidates),
            pairs=pairs,
            thresholds_by_mode={"return": threshold},
        )

        plausible = {
            edge.candidate_index
            for edge in pairs
            if (
                edge.score is not None
                and edge.score >= threshold.minimum_score
                and edge.support >= threshold.minimum_support
            )
        }

        return records, assignment, plausible

    # ------------------------------------------------------------------------
    # PROCESSING / ONE ACTUAL OBSERVATION
    # ------------------------------------------------------------------------

    def _process(
        self,
        candidates,
        stamp_ns,
        receipt,
        current_view,
    ):
        self._retire(stamp_ns, receipt)

        decisions = [
            IdentityDecision()
            for _ in candidates
        ]

        blocked = set()
        matched_keys = set()

        # A. Active tracks: one-to-one assignment.

        records, pairs, assignment = self._active_assignment(
            candidates,
            current_view,
        )

        for selected in assignment.accepted:
            edge = selected.evidence
            j = edge.candidate_index

            previous = records[edge.track_index]

            updated = update_track(
                previous.track,
                candidates[j],
                self.config,
            )

            self.active[previous.track.key] = _updated_record(
                previous,
                updated,
                candidates[j],
                receipt,
                current_view,
            )

            matched_keys.add(previous.track.key)

            decisions[j] = IdentityDecision(
                person_id=previous.track.key.person_id,
                state="MATCHED",
                confirmed=updated.confirmed,
                generation=updated.generation,
                score=edge.score,
            )

        for selected in assignment.ambiguous:
            edge = selected.evidence
            j = edge.candidate_index

            blocked.add(j)

            if not decisions[j].person_id:
                decisions[j] = IdentityDecision(
                    state="AMBIGUOUS",
                    score=edge.score,
                )

        for edge in pairs:
            if (
                edge.candidate_index in assignment.unassigned_candidates
                and self._admissible(edge)
            ):
                blocked.add(edge.candidate_index)

        # Actual missed observation: reset consecutive hits.
        # Do not renew track timestamps or receipts.
        for key, record in tuple(self.active.items()):
            if (
                key not in matched_keys
                and record.track.consecutive_hits != 0
            ):
                self.active[key] = replace(
                    record,
                    track=replace(
                        record.track,
                        consecutive_hits=0,
                    ),
                )

        free = [
            j
            for j in assignment.unassigned_candidates
            if j not in blocked
        ]

        # B. Dormant gallery.

        gallery_records, gallery, plausible = (
            self._dormant_assignment(
                candidates,
                free,
                stamp_ns,
                receipt,
            )
        )

        for selected in gallery.accepted:
            edge = selected.evidence
            j = edge.candidate_index

            if len(self.active) >= self.config.max_active_tracks:
                blocked.add(j)
                continue

            old = gallery_records[edge.track_index].track

            restored = create_track(
                candidates[j],
                old.key,
                generation=old.generation + 1,
                previous_count=old.observation_count,
                confirmed=(self.config.confirm_hits == 1),
            )

            self.active[old.key] = _new_record(
                restored,
                receipt,
                candidates[j],
                current_view,
            )

            del self.dormant[old.key]

            decisions[j] = IdentityDecision(
                person_id=old.key.person_id,
                state="RETURNING",
                confirmed=restored.confirmed,
                generation=restored.generation,
                score=edge.score,
            )

        for selected in gallery.ambiguous:
            edge = selected.evidence
            j = edge.candidate_index

            blocked.add(j)

            if not decisions[j].person_id:
                decisions[j] = IdentityDecision(
                    state="AMBIGUOUS",
                    score=edge.score,
                )

        blocked.update(plausible)

        # C. New tracks, including the first observation.

        for j, candidate in enumerate(candidates):
            if decisions[j].state != "INSUFFICIENT":
                continue

            if j in blocked:
                decisions[j] = IdentityDecision(
                    state="AMBIGUOUS"
                )
                continue

            if not initial_eligible(candidate, self.config):
                continue

            if len(self.active) >= self.config.max_active_tracks:
                continue

            key = self._new_key()

            track = create_track(
                candidate,
                key,
                confirmed=(self.config.confirm_hits == 1),
            )

            self.active[key] = _new_record(
                track,
                receipt,
                candidate,
                current_view,
            )

            decisions[j] = IdentityDecision(
                person_id=key.person_id,
                state="NEW",
                confirmed=track.confirmed,
                generation=track.generation,
            )

        # D. Output invariants.

        if len(decisions) != len(candidates):
            raise RuntimeError("IDENTITY_RESULT_COUNT_MISMATCH")

        assigned = [
            decision.person_id
            for decision in decisions
            if decision.person_id
        ]

        if len(assigned) != len(set(assigned)):
            raise RuntimeError(
                "DUPLICATE_PERSON_ID_IN_OBSERVATION"
            )

        return tuple(decisions)

    # ------------------------------------------------------------------------
    # PROCESSING / CONSISTENCY CHECKS
    # ------------------------------------------------------------------------

    def _validate_state(self):
        if set(self.active) & set(self.dormant):
            raise RuntimeError("ACTIVE_AND_DORMANT_IDENTITY")

        if len(self.active) > self.config.max_active_tracks:
            raise RuntimeError("ACTIVE_CAPACITY_EXCEEDED")

        if len(self.dormant) > self.config.max_gallery_entries:
            raise RuntimeError("GALLERY_CAPACITY_EXCEEDED")

        for collection in (self.active, self.dormant):
            for key, record in collection.items():
                if (
                    key.session_id != self.session_id
                    or record.track.key != key
                ):
                    raise RuntimeError("INVALID_TRACK_IDENTITY_KEY")

                if record.track.last_observation_ns > self.last_source_ns:
                    raise RuntimeError(
                        "TRACK_OBSERVATION_FROM_FUTURE"
                    )

                if (
                    record.last_observation_receipt
                    > self.last_receipt
                ):
                    raise RuntimeError(
                        "TRACK_RECEIPT_FROM_FUTURE"
                    )
