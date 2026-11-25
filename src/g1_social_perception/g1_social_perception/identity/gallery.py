"""Bounded dormant identity gallery; call only AFTER active-track association.

No long-horizon position extrapolation. No external/biometric identity.
An accepted return removes the record from the dormant gallery; the caller
reattaches its canonical key and reinitializes motion in the live tracker.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
from .appearance import Signature, compare

PersonKey = Tuple[str, str]


@dataclass(frozen=True)
class GalleryConfig:
    # Development defaults, not measured/calibrated operating thresholds.
    retention_sec: float = 30.0
    minimum_score: float = 0.90
    ambiguity_margin: float = 0.08
    min_region_quality: float = 0.70
    minimum_regions: int = 2
    max_entries: int = 64
    source_freshness_sec: float = 1.0
    future_tolerance_sec: float = 0.1
    weights: Tuple[float, float] = (0.5, 0.5)

    def __post_init__(self):
        for value in (self.retention_sec, self.source_freshness_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Durations must be finite and positive")
        for value in (self.minimum_score, self.ambiguity_margin,
                      self.min_region_quality, self.future_tolerance_sec):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid threshold")
        if self.minimum_regions not in (1, 2):
            raise ValueError("minimum_regions must be 1 or 2")
        if not isinstance(self.max_entries, int) or self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if len(self.weights) != 2 or any(
                not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Invalid weights")


@dataclass(frozen=True)
class Candidate:
    detection_index: int
    signature: Signature


@dataclass(frozen=True)
class Decision:
    detection_index: int
    status: str
    key: Optional[PersonKey] = None
    score: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class Record:
    key: PersonKey
    signature: Signature
    observed_ns: int
    received_monotonic: float


class IdentityGallery:
    def __init__(self, session_id, config=None):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("A non-empty identity-authority session is required")
        self.session_id = session_id
        self.config = config or GalleryConfig()
        self._records = {}
        self._last_source_ns = None
        self._last_now_ns = None
        self._last_monotonic = None

    @property
    def keys(self):
        return tuple(sorted(self._records))

    def remember(self, key, signature, *, last_observed_ns,
                 last_received_monotonic, confirmed):
        """Archive a retired confirmed track with its LAST REAL observation times.

        Retirement time must NOT replace these timestamps. The descriptor must
        come from a reliable confirmed association, never from a prediction.
        """
        if (not isinstance(key, tuple) or len(key) != 2 or
                key[0] != self.session_id or not isinstance(key[1], str) or
                not key[1].strip()):
            raise ValueError("Invalid or foreign-session key")
        if not confirmed:
            raise ValueError("Only confirmed identities enter the dormant gallery")
        if (not isinstance(last_observed_ns, int) or last_observed_ns <= 0 or
                not math.isfinite(last_received_monotonic)):
            raise ValueError("Invalid observation provenance")
        own = compare(signature, signature, weights=self.config.weights,
                      min_region_quality=self.config.min_region_quality)
        if own.common_regions < self.config.minimum_regions:
            raise ValueError("Insufficient reliable regions for a return template")
        old = self._records.get(key)
        if old and (last_observed_ns <= old.observed_ns or
                    last_received_monotonic < old.received_monotonic):
            raise ValueError("Duplicate/backward gallery observation")
        self._records[key] = Record(key, signature, last_observed_ns,
                                    last_received_monotonic)
        if len(self._records) > self.config.max_entries:
            oldest = min(self._records, key=lambda k: (
                self._records[k].received_monotonic, k))
            del self._records[oldest]

    def _validate_clocks(self, now_ros_ns, now_monotonic):
        if (not isinstance(now_ros_ns, int) or now_ros_ns <= 0 or
                not math.isfinite(now_monotonic)):
            raise ValueError("Invalid clocks")
        if ((self._last_now_ns is not None and now_ros_ns < self._last_now_ns) or
                (self._last_monotonic is not None and
                 now_monotonic < self._last_monotonic)):
            raise ValueError("Clock rollback: reset the identity session explicitly")
        if any(now_monotonic < r.received_monotonic for r in self._records.values()):
            raise ValueError("Receipt time is ahead of current monotonic clock")

    def _live_records(self, now_ros_ns, now_monotonic):
        return {
            key: record for key, record in self._records.items()
            if (-self.config.future_tolerance_sec <=
                (now_ros_ns - record.observed_ns) / 1e9 <= self.config.retention_sec)
            and (0 <= now_monotonic - record.received_monotonic <=
                 self.config.retention_sec)
        }

    def expire(self, *, now_ros_ns, now_monotonic):
        """Timer-only maintenance: never adds observations or extends retention."""
        self._validate_clocks(now_ros_ns, now_monotonic)
        self._records = self._live_records(now_ros_ns, now_monotonic)
        self._last_now_ns, self._last_monotonic = now_ros_ns, now_monotonic

    def resolve(self, candidates, *, source_stamp_ns, now_ros_ns, now_monotonic,
                active_keys=()):
        """Resolve one complete batch of genuinely new UNMATCHED detections.

        Statuses: RETURNING, NO_MATCH, AMBIGUOUS, INSUFFICIENT.
        NO_MATCH is not proof of a physically new person. No IDs are allocated.
        Validation/scoring complete before any gallery changes are committed.
        """
        candidates = tuple(candidates)
        active_keys = set(active_keys)
        self._validate_clocks(now_ros_ns, now_monotonic)
        if (not isinstance(source_stamp_ns, int) or source_stamp_ns <= 0 or
                (self._last_source_ns is not None and
                 source_stamp_ns <= self._last_source_ns)):
            raise ValueError("Not a genuinely new source frame")
        age = (now_ros_ns - source_stamp_ns) / 1e9
        if not -self.config.future_tolerance_sec <= age <= self.config.source_freshness_sec:
            raise ValueError("Stale/future source observation")
        indices = [c.detection_index for c in candidates]
        if (len(set(indices)) != len(indices) or
                any(not isinstance(i, int) or i < 0 for i in indices)):
            raise ValueError("Detection indices must be unique nonnegative integers")
        live = self._live_records(now_ros_ns, now_monotonic)
        eligible = {k: r for k, r in live.items() if k not in active_keys}
        scores, comparable, usable = {}, {}, {}
        for candidate in candidates:
            idx, signature = candidate.detection_index, candidate.signature
            own = compare(signature, signature, weights=self.config.weights,
                          min_region_quality=self.config.min_region_quality)
            usable[idx] = own.common_regions >= self.config.minimum_regions
            comparable[idx] = 0
            for key, record in eligible.items():
                if source_stamp_ns <= record.observed_ns:
                    continue
                if signature.schema != record.signature.schema:
                    continue
                similarity = compare(signature, record.signature,
                                     weights=self.config.weights,
                                     min_region_quality=self.config.min_region_quality)
                if usable[idx] and similarity.common_regions >= self.config.minimum_regions:
                    comparable[idx] += 1
                    scores[idx, key] = similarity.score
        decisions = []
        accepted = set()
        for candidate in candidates:
            idx = candidate.detection_index
            ranked = sorted(((score, key) for (i, key), score in scores.items() if i == idx),
                            key=lambda x: (-x[0], x[1]))
            if not usable[idx]:
                decisions.append(Decision(idx, "INSUFFICIENT", reason="candidate_regions"))
                continue
            if not ranked or ranked[0][0] < self.config.minimum_score:
                # Do not call absent/uncomparable reference evidence a mismatch.
                incomplete = comparable[idx] < len(eligible)
                status = "INSUFFICIENT" if incomplete else "NO_MATCH"
                decisions.append(Decision(idx, status,
                                          score=ranked[0][0] if ranked else None,
                                          reason="reference_unavailable" if incomplete
                                          else "no_compatible_retained_identity"))
                continue
            best, key = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else None
            competitors = [s for (i, k), s in scores.items() if k == key and i != idx]
            competitor = max(competitors) if competitors else None
            # A strong match cannot resolve another retained but uncomparable identity.
            uncertain_reference = comparable[idx] < len(eligible)
            row_ambiguous = second is not None and (
                best <= second or best - second < self.config.ambiguity_margin)
            col_ambiguous = competitor is not None and (
                best <= competitor or best - competitor < self.config.ambiguity_margin)
            if uncertain_reference or row_ambiguous or col_ambiguous:
                decisions.append(Decision(idx, "AMBIGUOUS", score=best,
                                          reason="unresolved_competing_identity"))
            else:
                decisions.append(Decision(idx, "RETURNING", key, best,
                                          "two_region_match" if self.config.minimum_regions == 2
                                          else "appearance_match"))
                accepted.add(key)
        # Atomic commit. Active identities cannot also remain dormant.
        self._records = {k: r for k, r in live.items()
                         if k not in accepted and k not in active_keys}
        self._last_source_ns = source_stamp_ns
        self._last_now_ns, self._last_monotonic = now_ros_ns, now_monotonic
        return tuple(decisions)
