"""Temporal social interpretation without ROS or navigation commands."""

from dataclasses import dataclass
from typing import Tuple
import math
import uuid


PersonKey = Tuple[str, str]


@dataclass(frozen=True)
class SocialSituation:
    episode_id: str
    revision: int
    unresolved: bool
    observation_status: str
    clearance_status: str
    participants: Tuple[PersonKey, ...]
    missing: Tuple[PersonKey, ...]
    evidence_stamp_ns: int
    transition: str


class SocialSituationEvaluator:
    def __init__(
        self,
        entry_distance=1.50,
        release_distance=1.65,
        entry_duration=0.20,
        clear_duration=1.00,
        missing_grace=0.50,
        maximum_gap=1.00,
        reacquisition_duration=0.30,
    ):
        positive_parameters = {
            "entry_distance": entry_distance,
            "release_distance": release_distance,
            "entry_duration": entry_duration,
            "clear_duration": clear_duration,
            "maximum_gap": maximum_gap,
        }

        for name, value in positive_parameters.items():
            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive."
                )

        # Zero explicitly selects immediate loss reporting.
        if (
            not math.isfinite(missing_grace)
            or missing_grace < 0.0
        ):
            raise ValueError(
                "missing_grace must be finite and non-negative."
            )

        if release_distance <= entry_distance:
            raise ValueError(
                "release_distance must exceed entry_distance."
            )

        self.entry_distance = entry_distance
        self.release_distance = release_distance
        self.entry_duration = entry_duration
        self.clear_duration = clear_duration
        self.missing_grace = missing_grace
        self.maximum_gap = maximum_gap

        if (
            not math.isfinite(reacquisition_duration)
            or reacquisition_duration <= 0.0
        ):
            raise ValueError(
                "reacquisition_duration must be finite and positive."
            )

        self.reacquisition_duration = reacquisition_duration

        # Fresh consecutive evidence required after an actual loss.
        self._reacquire_since_stamp_ns = None
        self._reacquire_count = 0

        self.episode_id = ""
        self.unresolved = False

        self.retired = False
        self.retired_episodes = {}

        self.participants = set()

        self._entry = {}
        self._clear_start = None
        self._clear_count = 0

        self._last_frame = None
        self._last_zone_frame = None
        self._last_received = None

        self._missing_since = None
        self._was_observed = False
        self._loss_reported = False

        self._revision = 0
        self._last_signature = None

    def retire(self):
        """Retire an unresolved interaction without claiming clearance."""
        if self.unresolved:
            self.retired = True


    def evaluate(
        self,
        persons,
        source_stamp_ns,
        now_monotonic,
        eligible_keys=None,
        *,
        evidence_status="UNAVAILABLE",
        source_session_id="",
        evidence_received_monotonic=None,
        unavailable_since_monotonic=None,
        coverage_complete=False,
    ):
        """
        Evaluate factual social evidence.

        Only fresh, matched source frames can establish
        current social geometry.

        Tracking memory preserves participant identity.
        It never substitutes for current observations.
        """

        records = {
            (person.session_id, person.person_id): person
            for person in persons
        }

        frame = (source_session_id, source_stamp_ns)

        matched = bool(
            evidence_status == "MATCHED"
            and source_session_id
            and source_stamp_ns is not None
            and eligible_keys is not None
        )

        current = {
            key: person
            for key, person in records.items()
            if (
                matched
                and person.geometry_fresh
                and person.latest.stamp_ns == source_stamp_ns
                and person.session_id == source_session_id
            )
        }

        distances = {
            key: math.hypot(*person.latest.position[:2])
            for key, person in current.items()
        }

        eligible = set(eligible_keys or ())

        new_frame = (
            matched
            and frame != self._last_frame
        )

        received = (
            now_monotonic
            if evidence_received_monotonic is None
            else evidence_received_monotonic
        )

        transition = ""
        evidence_stamp = source_stamp_ns if matched else 0

        # ------------------------------------------------------------
        # 1. Source continuity
        # ------------------------------------------------------------

        if new_frame and self._last_frame is not None:
            previous_session, previous_stamp = self._last_frame

            if (
                source_session_id == previous_session
                and source_stamp_ns <= previous_stamp
            ):
                matched = False
                new_frame = False
                current = {}
                distances = {}
                eligible.clear()
                evidence_stamp = 0

            elif (
                source_session_id != previous_session
                or (
                    source_stamp_ns - previous_stamp
                ) / 1e9 > self.maximum_gap
                or (
                    received - self._last_received
                ) > self.maximum_gap
            ):
                self._entry.clear()
                self._clear_start = None
                self._clear_count = 0
                self._reacquire_since_stamp_ns = None
                self._reacquire_count = 0

        if not matched:
            self._entry.clear()
            self._clear_start = None
            self._clear_count = 0

        # ------------------------------------------------------------
        # 2. Intrusion confirmation
        # ------------------------------------------------------------

        if new_frame:
            self._last_frame = frame
            self._last_received = received

            intruders = {
                key
                for key in eligible
                if (
                    key in distances
                    and distances[key] <= self.entry_distance
                )
            }

            self._entry = {
                key: value
                for key, value in self._entry.items()
                if key in intruders
            }

            confirmed = set()

            for key in intruders:
                since, count = self._entry.get(
                    key, (source_stamp_ns, 0)
                )

                count += 1
                self._entry[key] = (since, count)

                if (
                    count >= 2
                    and (
                        source_stamp_ns - since
                    ) / 1e9 >= self.entry_duration
                ):
                    confirmed.add(key)

            if confirmed and (
                not self.unresolved or self.retired
            ):
                if self.retired:
                    self.retired_episodes[self.episode_id] = (
                        tuple(sorted(self.participants))
                    )

                self.episode_id = uuid.uuid4().hex
                self.unresolved = True
                self.retired = False
                self.participants = set(confirmed)

                self._clear_start = None
                self._clear_count = 0
                self._missing_since = None
                self._loss_reported = False
                self._reacquire_since_stamp_ns = None
                self._reacquire_count = 0

                transition = "INTRUSION_CONFIRMED"

            elif (
                self.unresolved
                and confirmed - self.participants
            ):
                self.participants.update(confirmed)
                transition = "PARTICIPANTS_CHANGED"

        # ------------------------------------------------------------
        # 3. Current participant observation
        # ------------------------------------------------------------

        # Both a fresh person observation and its corresponding
        # current social zone are required.
        observed_zone_keys = (
            set(current).intersection(eligible)
        )

        missing = (
            self.participants - observed_zone_keys
        )

        observed_now = bool(
            self.participants
            and not missing
            and matched
        )

        # ------------------------------------------------------------
        # 4. Temporal reacquisition
        # ------------------------------------------------------------

        observed = observed_now

        if (
            self.unresolved
            and self._loss_reported
            and not self.retired
        ):
            if observed_now:
                if new_frame:
                    if self._reacquire_since_stamp_ns is None:
                        self._reacquire_since_stamp_ns = (
                            source_stamp_ns
                        )
                        self._reacquire_count = 1
                    else:
                        self._reacquire_count += 1

                duration = (
                    (
                        source_stamp_ns
                        - self._reacquire_since_stamp_ns
                    ) / 1e9
                    if self._reacquire_since_stamp_ns is not None
                    else 0.0
                )

                observed = bool(
                    self._reacquire_count >= 2
                    and duration >= self.reacquisition_duration
                )

            else:
                self._reacquire_since_stamp_ns = None
                self._reacquire_count = 0

        else:
            self._reacquire_since_stamp_ns = None
            self._reacquire_count = 0

        # ------------------------------------------------------------
        # 5. Active unresolved episode
        # ------------------------------------------------------------

        if self.unresolved:
            clearance = "NOT_CONFIRMED"

            if observed:
                observation = "OBSERVED"
                self._missing_since = None

                if (
                    self._loss_reported
                    and not self.retired
                    and not transition
                ):
                    transition = "OBSERVATION_REACQUIRED"

                self._loss_reported = False

                # Clearance requires complete source coverage,
                # not merely the presence of historical participants.
                outside = bool(
                    coverage_complete
                    and all(
                        distance > self.release_distance
                        for distance in distances.values()
                    )
                )

                if outside:
                    if new_frame:
                        if self._clear_start is None:
                            self._clear_start = source_stamp_ns

                        self._clear_count += 1

                    clearance = "PENDING"

                    if (
                        new_frame
                        and self._clear_count >= 2
                        and (
                            source_stamp_ns - self._clear_start
                        ) / 1e9 >= self.clear_duration
                    ):
                        self.unresolved = False
                        self.retired = False
                        clearance = "CONFIRMED"
                        transition = "CLEAR"

                else:
                    self._clear_start = None
                    self._clear_count = 0

            else:
                self._clear_start = None
                self._clear_count = 0

                starts = [
                    person.missing_since_monotonic
                    for key, person in records.items()
                    if (
                        key in missing
                        and person.missing_since_monotonic is not None
                    )
                ]

                if unavailable_since_monotonic is not None:
                    starts.append(unavailable_since_monotonic)

                since = (
                    min(starts)
                    if starts
                    else now_monotonic
                )

                self._missing_since = (
                    since
                    if self._missing_since is None
                    else min(self._missing_since, since)
                )

                expired = bool(
                    evidence_status == "STALE"
                    or any(
                        key in records
                        and records[key].observation_expired
                        for key in missing
                    )
                )

                lost = bool(
                    expired
                    or (
                        now_monotonic - self._missing_since
                    ) >= self.missing_grace
                )

                observation = (
                    "LOST"
                    if lost
                    else "TEMPORARILY_MISSING"
                )

                if lost and not self._loss_reported:
                    if not transition:
                        transition = "OBSERVATION_LOST"

                    self._loss_reported = True

        # ------------------------------------------------------------
        # 6. No unresolved episode
        # ------------------------------------------------------------

        else:
            observation = (
                "OBSERVED"
                if current
                else "UNAVAILABLE"
            )

            clearance = (
                "CONFIRMED"
                if self.episode_id
                else "NOT_EVALUATED"
            )

        self._was_observed = observed

        # ------------------------------------------------------------
        # 7. Revision and immutable output
        # ------------------------------------------------------------

        signature = (
            self.episode_id,
            self.unresolved,
            observation,
            clearance,
            tuple(sorted(self.participants)),
            tuple(sorted(missing)),
        )

        if signature != self._last_signature:
            self._revision += 1
            self._last_signature = signature

        return SocialSituation(
            episode_id=self.episode_id,
            revision=self._revision,
            unresolved=self.unresolved,
            observation_status=observation,
            clearance_status=clearance,
            participants=tuple(sorted(self.participants)),
            missing=tuple(sorted(missing)),
            evidence_stamp_ns=evidence_stamp,
            transition=transition,
        )


    

    
    