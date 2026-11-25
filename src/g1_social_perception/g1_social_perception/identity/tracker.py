"""Transactional 2D association with two-region appearance and dormant returns.

This integration retains geometric IoU/centre matching. It does not implement
3D prediction, metric-size association, or a learned re-identification model.
"""
import copy
import math
from dataclasses import dataclass
from .appearance import compare, Signature
from .gallery import IdentityGallery, GalleryConfig, Candidate


@dataclass(frozen=True)
class TrackerConfig:
    active_timeout_sec: float = 0.75
    confirm_observations: int = 3
    minimum_iou: float = 0.20
    maximum_center_distance_px: float = 100.0
    iou_weight: float = 0.70
    center_weight: float = 0.30
    appearance_weight: float = 0.35
    minimum_match_score: float = 0.35
    minimum_geometry_only_score: float = 0.60
    minimum_appearance_score: float = 0.50
    ambiguity_margin: float = 0.08
    max_tracks: int = 128

    def __post_init__(self):
        for value in (self.active_timeout_sec, self.maximum_center_distance_px):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid positive tracker parameter")
        for value in (self.minimum_iou, self.iou_weight, self.center_weight,
                      self.appearance_weight, self.minimum_match_score,
                      self.minimum_geometry_only_score,
                      self.minimum_appearance_score, self.ambiguity_margin):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid tracker threshold/weight")
        if self.iou_weight + self.center_weight <= 0:
            raise ValueError("Geometry weights must have positive sum")
        if self.confirm_observations < 1 or self.max_tracks < 1:
            raise ValueError("Invalid tracker capacity/confirmation count")


@dataclass(frozen=True)
class Observation:
    box: tuple
    signature: Signature


@dataclass
class Track:
    person_id: str
    box: tuple
    stamp_ns: int
    receipt: float
    signature: Signature
    signature_ns: int
    signature_receipt: float
    consecutive_hits: int = 1
    confirmed: bool = False
    generation: int = 1


@dataclass(frozen=True)
class Result:
    person_id: str
    state: str
    confirmed: bool = False
    generation: int = 0
    score: float = 0.0


def geometry(a, b, config):
    def corners(box):
        x, y, w, h = box
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax, ay, ar, ab = corners(a)
    bx, by, br, bb = corners(b)
    area = max(0, min(ar, br)-max(ax, bx))*max(0, min(ab, bb)-max(ay, by))
    iou = area/(a[2]*a[3]+b[2]*b[3]-area)
    distance = math.hypot(a[0]-b[0], a[1]-b[1])
    if iou < config.minimum_iou and distance > config.maximum_center_distance_px:
        return None
    proximity = max(0.0, 1-distance/config.maximum_center_distance_px)
    return ((config.iou_weight*iou+config.center_weight*proximity)/
            (config.iou_weight+config.center_weight))


class IdentityTracker:
    def __init__(self, session_id, config=None, gallery_config=None):
        self.config = config or TrackerConfig()
        self.gallery = IdentityGallery(session_id, gallery_config)
        if self.gallery.config.retention_sec <= self.config.active_timeout_sec:
            raise ValueError("Gallery horizon must exceed active-track retention")
        self.session_id = session_id
        self.tracks = {}
        self.next_id = 1
        self.last_source_ns = None
        self.last_frame = None

    def _retire(self, now_ns, monotonic):
        for key, track in list(self.tracks.items()):
            if ((now_ns-track.stamp_ns)/1e9 > self.config.active_timeout_sec or
                    monotonic-track.receipt > self.config.active_timeout_sec):
                if track.confirmed:
                    try:
                        self.gallery.remember(
                            (self.session_id, key), track.signature,
                            last_observed_ns=track.signature_ns,
                            last_received_monotonic=track.signature_receipt,
                            confirmed=True)
                    except ValueError:
                        # An unusable template cannot support a future return.
                        pass
                del self.tracks[key]
        self.gallery.expire(now_ros_ns=now_ns, now_monotonic=monotonic)

    def expire(self, now_ns, monotonic):
        staged = copy.deepcopy(self)
        staged.gallery._validate_clocks(now_ns, monotonic)
        staged._retire(now_ns, monotonic)
        self.__dict__.update(staged.__dict__)

    def process(self, observations, *, stamp_ns, now_ns, monotonic, frame_id):
        """No caller-visible changes occur if validation/scoring fails."""
        observations = tuple(observations)
        if (not isinstance(stamp_ns, int) or stamp_ns <= 0 or
                self.last_source_ns is not None and stamp_ns <= self.last_source_ns):
            raise ValueError("Source frame is not strictly newer")
        if not frame_id or (self.last_frame and frame_id != self.last_frame):
            raise ValueError("Image frame changed: restart the identity authority")
        age = (now_ns-stamp_ns)/1e9
        cfg = self.gallery.config
        if not -cfg.future_tolerance_sec <= age <= cfg.source_freshness_sec:
            raise ValueError("Source frame is stale/future")
        for obs in observations:
            if (len(obs.box) != 4 or not all(math.isfinite(v) for v in obs.box)
                    or min(obs.box[2:]) <= 0):
                raise ValueError("Invalid person box")
            compare(obs.signature, obs.signature)
        staged = copy.deepcopy(self)
        staged.gallery._validate_clocks(now_ns, monotonic)
        results = staged._process(observations, stamp_ns, now_ns, monotonic)
        staged.last_source_ns, staged.last_frame = stamp_ns, frame_id
        self.__dict__.update(staged.__dict__)
        return results

    def _process(self, observations, stamp_ns, now_ns, monotonic):
        self._retire(now_ns, monotonic)
        scores = {}
        cfg = self.config
        for key, track in self.tracks.items():
            for j, obs in enumerate(observations):
                geo = geometry(track.box, obs.box, cfg)
                if geo is None:
                    continue
                sim = compare(track.signature, obs.signature)
                if sim.score is not None:
                    if sim.score < cfg.minimum_appearance_score:
                        continue
                    weight = cfg.appearance_weight * sim.evidence_quality
                    score = ((1-cfg.appearance_weight)*geo + weight*sim.score)/(
                        1-cfg.appearance_weight+weight)
                    threshold = cfg.minimum_match_score
                else:
                    score = geo
                    threshold = cfg.minimum_geometry_only_score
                if score >= threshold:
                    scores[key, j] = score
        output = {}
        # Mutual unique best: conservative, not a Hungarian/global optimum claim.
        for j, obs in enumerate(observations):
            ranked = sorted(((s, key) for (key, idx), s in scores.items() if idx == j),
                            reverse=True)
            if not ranked:
                continue
            best, key = ranked[0]
            others = [s for (k, idx), s in scores.items() if k == key and idx != j]
            if ((len(ranked)>1 and best-ranked[1][0] < cfg.ambiguity_margin) or
                    (len(ranked)>1 and best <= ranked[1][0]) or
                    (others and (best-max(others) < cfg.ambiguity_margin or best <= max(others)))):
                output[j] = Result("", "AMBIGUOUS", score=best)
                continue
            track = self.tracks[key]
            if track.stamp_ns != self.last_source_ns:
                track.consecutive_hits = 0
                track.generation += 1
            track.consecutive_hits += 1
            track.confirmed = track.consecutive_hits >= cfg.confirm_observations
            track.box, track.stamp_ns, track.receipt = obs.box, stamp_ns, monotonic
            own = compare(obs.signature, obs.signature,
                          min_region_quality=self.gallery.config.min_region_quality)
            if own.common_regions == 2:
                track.signature = obs.signature
                track.signature_ns, track.signature_receipt = stamp_ns, monotonic
            output[j] = Result(key, "MATCHED", track.confirmed, track.generation, best)
        pending = [Candidate(j, obs.signature) for j, obs in enumerate(observations)
                   if j not in output]
        if len(pending) > cfg.max_tracks - len(self.tracks):
            # Do not consume a dormant identity if no live slot can be reserved.
            for candidate in pending:
                output[candidate.detection_index] = Result("", "INSUFFICIENT")
            pending = []
        decisions = self.gallery.resolve(
            pending, source_stamp_ns=stamp_ns, now_ros_ns=now_ns,
            now_monotonic=monotonic,
            active_keys=[(self.session_id, key) for key in self.tracks])
        for decision in decisions:
            j, status = decision.detection_index, decision.status
            if status not in ("RETURNING", "NO_MATCH"):
                output[j] = Result("", status, score=decision.score or 0.0)
                continue
            if len(self.tracks) >= cfg.max_tracks:
                output[j] = Result("", "INSUFFICIENT")
                continue
            if status == "RETURNING":
                key = decision.key[1]
            else:
                key = "person_{}".format(self.next_id)
                self.next_id += 1
            obs = observations[j]
            # A fresh live segment always starts with unconfirmed geometry.
            track = Track(key, obs.box, stamp_ns, monotonic, obs.signature,
                          stamp_ns, monotonic,
                          confirmed=cfg.confirm_observations == 1)
            # Monotonic segment number changes even across gallery returns.
            track.generation = stamp_ns
            self.tracks[key] = track
            output[j] = Result(key, "RETURNING" if status == "RETURNING" else "NEW",
                               track.confirmed, track.generation, decision.score or 0.0)
        return tuple(output[j] for j in range(len(observations)))
