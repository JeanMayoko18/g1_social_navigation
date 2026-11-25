"""Keep actual observations separate from tracker reports."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math

@dataclass(frozen=True)
class Observation:
    stamp_ns: int
    frame_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    confidence: float
    observation_count: int


@dataclass(frozen=True)
class SocialZoneObservation:
    """Last validated social geometry, not necessarily current."""

    stamp_ns: int
    frame_id: str

    position: Tuple[float, float, float]
    heading: float

    radius_longitudinal: float
    radius_lateral: float

    zone_type: str
    person_speed: float


@dataclass
class PersonRecord:
    session_id: str
    person_id: str
    identity_generation: int

    last_observed: Optional[Observation] = None
    last_reported: Optional[Observation] = None
    last_social_zone: Optional[SocialZoneObservation] = None
    last_social_zone_map: Optional[SocialZoneObservation] = None

    tracking_state: str = "TENTATIVE"
    observed_in_current_array: bool = False

    last_received_monotonic: Optional[float] = None

    missing_since_monotonic: Optional[float] = None
    missing_since_source_stamp_ns: Optional[int] = None

    history: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )

@dataclass(frozen=True)
class SocialGeometrySnapshot:
    session_id: str
    person_id: str
    identity_generation: int

    status: str
    geometry: Optional[SocialZoneObservation]

    observation_stamp_ns: int
    observation_age_sec: float

    source_frame: str
    observed_now: bool

@dataclass(frozen=True)
class PersonSnapshot:
    session_id: str
    person_id: str
    identity_generation: int

    # Compatibility name: always the last actual observation.
    latest: Observation

    tracking_state: str
    observed_in_current_array: bool
    geometry_fresh: bool
    last_social_zone: Optional[SocialZoneObservation]
    last_social_zone_map: Optional[SocialZoneObservation]

    source_age_sec: float
    missing_since_monotonic: Optional[float]
    history: Tuple[Observation, ...]
    

    missing_since_source_stamp_ns: Optional[int]
    receipt_age_sec: float
    observation_expired: bool


class SocialContextMemory:
    def __init__(
        self,
        expected_frame,
        freshness_sec=1.0,
    ):
        if not expected_frame:
            raise ValueError(
                "expected_frame must be non-empty."
            )

        if (
            not math.isfinite(freshness_sec)
            or freshness_sec <= 0.0
        ):
            raise ValueError(
                "freshness_sec must be finite and positive."
            )

        self.expected_frame = expected_frame
        self.freshness_sec = freshness_sec

        self._persons = {}
        self.session_id = ""
        self.last_array_stamp_ns = None
        self._retired_sessions = set()

    @staticmethod
    def _stamp_ns(stamp):
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)

        value = (
            sec * 1_000_000_000
            + nanosec
        )

        if (
            sec < 0
            or not 0 <= nanosec < 1_000_000_000
            or value <= 0
        ):
            raise ValueError(
                "Invalid source timestamp."
            )

        return value

    def update(
        self,
        msg,
        now_ros_ns,
        now_monotonic,
    ):
        """
        Validate the complete array before committing changes.

        Only actual observations update last_observed and history.
        Tracker reports never renew actual observation freshness.
        """
        try:
            if not math.isfinite(now_monotonic):
                return False

            session = str(
                msg.tracking_session_id
            ).strip()

            if (
                not session
                or session in self._retired_sessions
            ):
                return False

            if msg.header.frame_id != self.expected_frame:
                return False

            array_ns = self._stamp_ns(
                msg.header.stamp
            )

            array_age = (
                now_ros_ns - array_ns
            ) / 1e9

            if not -0.1 <= array_age <= self.freshness_sec:
                return False

            if (
                session == self.session_id
                and self.last_array_stamp_ns is not None
                and array_ns <= self.last_array_stamp_ns
            ):
                return False

            staged = {}

            for person in msg.persons:
                person_id = str(person.id).strip()
                generation = person.identity_generation

                source_ns = self._stamp_ns(
                    person.last_observation_stamp
                )

                observed = bool(
                    person.observed_in_current_array
                )

                state = str(
                    person.tracking_state
                )

                if (
                    not person_id
                    or person_id in staged
                    or person.tracking_session_id != session
                    or type(generation) is not int
                    or generation <= 0
                    or source_ns > array_ns
                    or observed != (source_ns == array_ns)
                    or state not in {
                        "TENTATIVE",
                        "TRACKED",
                        "LOST",
                    }
                    or (
                        observed
                        and state == "LOST"
                    )
                ):
                    return False

                position = tuple(
                    float(getattr(person.position, axis))
                    for axis in "xyz"
                )

                velocity = tuple(
                    float(getattr(person.velocity, axis))
                    for axis in "xyz"
                )

                confidence = float(person.confidence)
                count = int(person.observation_count)

                if not all(
                    math.isfinite(value)
                    for value in (
                        position
                        + velocity
                        + (confidence,)
                    )
                ):
                    return False

                if (
                    not 0.0 <= confidence <= 1.0
                    or count < 0
                ):
                    return False

                sample = Observation(
                    stamp_ns=source_ns,
                    frame_id=msg.header.frame_id,
                    position=position,
                    velocity=velocity,
                    confidence=confidence,
                    observation_count=count,
                )

                old = self._persons.get(
                    (session, person_id)
                )

                # Never compare observation history across generations.
                same_identity = bool(
                    old is not None
                    and old.identity_generation == generation
                )

                real = (
                    old.last_observed
                    if same_identity
                    else None
                )

                if (
                    real is not None
                    and source_ns < real.stamp_ns
                ):
                    return False

                if (
                    observed
                    and real is not None
                    and source_ns <= real.stamp_ns
                ):
                    return False

                staged[person_id] = (
                    sample,
                    state,
                    observed,
                    generation,
                )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return False

        # Commit only after validating the complete array.
        if (
            self.session_id
            and session != self.session_id
        ):
            self._retired_sessions.add(
                self.session_id
            )

            # Observation records only: no social clearance is inferred.
            self._persons.clear()

        self.session_id = session
        self.last_array_stamp_ns = array_ns

        # Start an explicit absence interval for previously observed
        # persons. Actual observations below cancel that interval.
        for record in self._persons.values():
            record.observed_in_current_array = False

            if (
                record.last_observed is not None
                and record.missing_since_monotonic is None
            ):
                record.missing_since_monotonic = (
                    now_monotonic
                )

                record.missing_since_source_stamp_ns = (
                    array_ns
                )

        for person_id, values in staged.items():
            sample, state, observed, generation = values
            key = (session, person_id)

            record = self._persons.get(key)

            if (
                record is None
                or record.identity_generation != generation
            ):
                # A new generation must never inherit the old
                # generation's observation history or geometry.
                record = PersonRecord(
                    session_id=session,
                    person_id=person_id,
                    identity_generation=generation,
                )

                self._persons[key] = record

            record.last_reported = sample
            record.tracking_state = state
            record.observed_in_current_array = observed

            if observed:
                record.last_observed = sample
                record.last_received_monotonic = (
                    now_monotonic
                )

                record.history.append(sample)

                record.missing_since_monotonic = None
                record.missing_since_source_stamp_ns = None

        return True

    def remember_validated_zones(self, msg):
        """
        Commit zones belonging to already validated person observations.

        This method is intended for a staged memory copy after
        the source-pair validation has succeeded.

        Missing zones do not erase historical geometry.
        """
        try:
            session = str(msg.tracking_session_id).strip()
            frame = str(msg.header.frame_id).strip()
            stamp_ns = self._stamp_ns(msg.header.stamp)

            if (
                session != self.session_id
                or frame != self.expected_frame
                or stamp_ns != self.last_array_stamp_ns
            ):
                return False

            staged = {}
            seen = set()

            for zone in msg.zones:
                person_id = str(zone.person_id).strip()
                generation = int(zone.identity_generation)
                key = (session, person_id)

                if (
                    not person_id
                    or key in seen
                    or generation <= 0
                    or str(zone.tracking_session_id) != session
                    or str(zone.tracking_state) != "TRACKED"
                    or self._stamp_ns(
                        zone.observation_stamp
                    ) != stamp_ns
                ):
                    return False

                seen.add(key)

                record = self._persons.get(key)

                if (
                    record is None
                    or record.identity_generation != generation
                    or not record.observed_in_current_array
                    or record.tracking_state != "TRACKED"
                    or record.last_observed is None
                    or record.last_observed.stamp_ns != stamp_ns
                ):
                    return False

                values = (
                    float(zone.center.x),
                    float(zone.center.y),
                    float(zone.center.z),
                    float(zone.heading),
                    float(zone.radius_longitudinal),
                    float(zone.radius_lateral),
                    float(zone.person_speed),
                )

                if (
                    not all(math.isfinite(v) for v in values)
                    or values[4] <= 0.0
                    or values[5] <= 0.0
                    or values[6] < 0.0
                    or str(zone.zone_type) not in {
                        "STATIC", "DYNAMIC"
                    }
                    or values[:3] != record.last_observed.position
                ):
                    return False

                previous = record.last_social_zone

                if (
                    previous is not None
                    and stamp_ns <= previous.stamp_ns
                ):
                    return False

                staged[key] = SocialZoneObservation(
                    stamp_ns=stamp_ns,
                    frame_id=frame,
                    position=values[:3],
                    heading=values[3],
                    radius_longitudinal=values[4],
                    radius_lateral=values[5],
                    zone_type=str(zone.zone_type),
                    person_speed=values[6],
                )

            # No modification is made before every zone is valid.
            for key, geometry in staged.items():
                record = self._persons[key]

                # A new source observation invalidates the previous projection.
                # The projection must always match the current source stamp.
                if (
                    record.last_social_zone_map is not None
                    and record.last_social_zone_map.stamp_ns != geometry.stamp_ns
                ):
                    record.last_social_zone_map = None

                record.last_social_zone = geometry

            return True

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return False

    def snapshot(
        self,
        now_ros_ns,
        now_monotonic,
    ):
        """Expose actual measurements without refreshing their timestamps."""
        views = []

        for _, record in sorted(self._persons.items()):
            real = record.last_observed

            # A historical report received without an actual observation
            # does not establish an observed geometry.
            if real is None:
                continue

            source_age = (
                now_ros_ns - real.stamp_ns
            ) / 1e9

            receipt_age = (
                now_monotonic
                - record.last_received_monotonic
            )

            expired = not (
                -0.1 <= source_age <= self.freshness_sec
                and 0.0 <= receipt_age <= self.freshness_sec
            )

            fresh = bool(
                not expired
                and record.observed_in_current_array
                and record.tracking_state == "TRACKED"
            )

            views.append(
                PersonSnapshot(
                    session_id=record.session_id,
                    person_id=record.person_id,
                    identity_generation=record.identity_generation,
                    latest=real,
                    tracking_state=record.tracking_state,
                    observed_in_current_array=(
                        record.observed_in_current_array
                    ),
                    last_social_zone=record.last_social_zone,
                    last_social_zone_map=record.last_social_zone_map,
                    geometry_fresh=fresh,
                    source_age_sec=source_age,
                    missing_since_monotonic=(
                        record.missing_since_monotonic
                    ),
                    history=tuple(record.history),
                    missing_since_source_stamp_ns=(
                        record.missing_since_source_stamp_ns
                    ),
                    receipt_age_sec=receipt_age,
                    observation_expired=expired,
                )
            )

        return tuple(views)

    def remember_projected_zone(
        self,
        *,
        session_id,
        person_id,
        identity_generation,
        source_stamp_ns,
        target_frame,
        position,
        heading,
    ):
        """
        Store the map-frame projection of an already validated zone.

        Projection does not constitute a new observation.
        The original source timestamp is preserved.
        """
        key = (str(session_id), str(person_id))
        record = self._persons.get(key)

        if record is None:
            return False

        source = record.last_social_zone

        if (
            source is None
            or record.identity_generation != identity_generation
            or source.stamp_ns != source_stamp_ns
            or target_frame != "map"
        ):
            return False

        values = (*position, heading)

        if (
            len(position) != 3
            or not all(math.isfinite(float(v)) for v in values)
        ):
            return False

        record.last_social_zone_map = SocialZoneObservation(
            stamp_ns=source.stamp_ns,
            frame_id=target_frame,
            position=tuple(float(v) for v in position),
            heading=float(heading),
            radius_longitudinal=source.radius_longitudinal,
            radius_lateral=source.radius_lateral,
            zone_type=source.zone_type,
            person_speed=source.person_speed,
        )

        return True

    def social_geometry_snapshot(
        self,
        now_ros_ns,
        now_monotonic,
    ):
        """
        Describe stored social geometry without changing memory.

        OBSERVED:
            Zone and person belong to the same current observation.

        MEMORY:
            A previously validated zone is retained, but it is
            not a current observation.

        UNAVAILABLE:
            No validated social zone has been stored.

        This method does not authorize mask installation or motion.
        """
        persons = self.snapshot(
            now_ros_ns=now_ros_ns,
            now_monotonic=now_monotonic,
        )

        results = []

        for person in persons:
            zone = person.last_social_zone

            if zone is None:
                results.append(
                    SocialGeometrySnapshot(
                        session_id=person.session_id,
                        person_id=person.person_id,
                        identity_generation=person.identity_generation,
                        status="UNAVAILABLE",
                        geometry=None,
                        observation_stamp_ns=0,
                        observation_age_sec=float("inf"),
                        source_frame="",
                        observed_now=False,
                    )
                )
                continue

            zone_current = bool(
                person.geometry_fresh
                and person.observed_in_current_array
                and person.tracking_state == "TRACKED"
                and zone.stamp_ns == person.latest.stamp_ns
                and zone.stamp_ns == self.last_array_stamp_ns
            )

            status = (
                "OBSERVED"
                if zone_current
                else "MEMORY"
            )

            age = max(
                0.0,
                (now_ros_ns - zone.stamp_ns) / 1e9,
            )

            results.append(
                SocialGeometrySnapshot(
                    session_id=person.session_id,
                    person_id=person.person_id,
                    identity_generation=person.identity_generation,
                    status=status,
                    geometry=zone,
                    observation_stamp_ns=zone.stamp_ns,
                    observation_age_sec=age,
                    source_frame=zone.frame_id,
                    observed_now=zone_current,
                )
            )

        return tuple(results)

class SocialIdentityRegistry:
    """
    Stable social identities based on authoritative tracker keys.

    A tracker key keeps the same social identity throughout
    its source session, regardless of observation gaps.

    A different tracker key is NOT automatically treated as
    the same person. Such a merge would require explicit
    identity-association evidence from the tracker.
    """

    def __init__(self):
        import uuid

        self._uuid = uuid
        self.source_session_id = ""
        self.identities = {}

    def start_session(self, source_session_id):
        session = str(source_session_id).strip()

        if not session:
            raise ValueError("EMPTY_SOURCE_SESSION")

        if session != self.source_session_id:
            self.source_session_id = session
            self.identities.clear()

    def resolve(self, tracker_key, observation_stamp_ns):
        key = str(tracker_key).strip()
        stamp = int(observation_stamp_ns)

        if not self.source_session_id:
            raise ValueError("NO_ACTIVE_SOURCE_SESSION")

        if not key or stamp <= 0:
            raise ValueError("INVALID_TRACK_OBSERVATION")

        record = self.identities.get(key)

        if record is None:
            person_id = "yt_" + self._uuid.uuid4().hex

            self.identities[key] = {
                "person_id": person_id,
                "last_seen_ns": stamp,
                "generation": 1,
            }

            return person_id, "NEW", 1

        previous_stamp = record["last_seen_ns"]

        if stamp <= previous_stamp:
            raise ValueError("NON_MONOTONIC_TRACK_OBSERVATION")

        gap_sec = (stamp - previous_stamp) / 1e9

        record["last_seen_ns"] = stamp

        state = (
            "RETURNING"
            if gap_sec >= 0.5
            else "MATCHED"
        )

        return (
            record["person_id"],
            state,
            record["generation"],
        )

    def __len__(self):
        return len(self.identities)

