"""
Interprétation sociale individuelle, indépendante de ROS.

Nouvelle interface : intégration explicite nécessaire.
Ce module ne remplace pas directement SocialSituationEvaluator.

Les observations entrantes doivent déjà être appariées et
validées géométriquement.

Une observation vide ne signifie jamais CLEAR pour une
personne historiquement présente mais actuellement absente.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import uuid

# (tracking_session_id, person_id, identity_generation)
PersonKey = Tuple[str, str, int]


@dataclass(frozen=True)
class ObservedPerson:
    key: PersonKey
    distance: float


@dataclass(frozen=True)
class ParticipantSituation:
    key: PersonKey
    episode_id: str
    revision: int
    unresolved: bool
    observation_status: str
    conflict_status: str
    transition: str
    observed_now: bool
    observation_stamp_ns: int
    last_distance: Optional[float]
    conflict_observed_sec: float
    persistent: bool
    retired: bool


@dataclass(frozen=True)
class IndividualSituation:
    participants: Tuple[ParticipantSituation, ...]
    source_fresh: bool
    source_coverage_complete: bool
    source_session: str
    source_stamp_ns: int


@dataclass
class _Record:
    episode: str = ""
    unresolved: bool = False
    retired: bool = False

    observed: bool = False
    observation: str = "UNAVAILABLE"
    conflict: str = "NOT_EVALUATED"

    last_stamp: int = 0
    last_received: Optional[float] = None
    last_distance: Optional[float] = None
    missing_since: Optional[float] = None

    entry_sec: float = 0.0
    entry_samples: int = 0

    clear_sec: float = 0.0
    clear_samples: int = 0

    reacquire_sec: float = 0.0
    reacquire_samples: int = 0

    loss_reported: bool = False

    conflict_sec: float = 0.0
    revision: int = 0
    signature: object = None


class IndividualSocialEvaluator:
    """
    Un cycle de vie et des compteurs par identité de suivi.

    update():
        Consomme une nouvelle observation source valide,
        éventuellement de couverture partielle.

    tick():
        Fait vieillir les preuves sans accumuler de durée
        d'observation.

    unavailable():
        Signale l'indisponibilité de la source entière.
        Ne pas l'utiliser pour la seule absence d'une personne.
    """

    def __init__(
        self,
        entry_distance=1.50,
        release_distance=1.65,
        entry_duration=0.20,
        clear_duration=1.00,
        missing_grace=0.50,
        maximum_gap=1.50,
        reacquisition_duration=0.30,
        persistence_duration=2.00,
        retirement_duration=20.00,
    ):
        values = locals().copy()
        values.pop("self")

        for name, value in values.items():
            if (
                not math.isfinite(value)
                or value < 0
                or (
                    value == 0
                    and name != "missing_grace"
                )
            ):
                raise ValueError(
                    "Invalid parameter: " + name
                )

            setattr(self, name, float(value))

        if release_distance <= entry_distance:
            raise ValueError(
                "release_distance must exceed entry_distance"
            )

        if retirement_duration <= missing_grace:
            raise ValueError(
                "retirement_duration must exceed missing_grace"
            )

        self._records: Dict[PersonKey, _Record] = {}

        self._session = ""
        self._retired_sessions = set()

        self._stamp = 0
        self._received = None
        self._coverage = False
        self._source_valid = False
        self._last_now = None

    def _check_time(self, now_ros_ns, now_monotonic):
        if (
            isinstance(now_ros_ns, bool)
            or not isinstance(now_ros_ns, int)
            or now_ros_ns <= 0
        ):
            raise ValueError("Invalid ROS time")

        if not math.isfinite(now_monotonic):
            raise ValueError("Invalid monotonic time")

        if (
            self._last_now is not None
            and now_monotonic < self._last_now
        ):
            raise ValueError(
                "Monotonic clock moved backwards"
            )

    def update(
        self,
        *,
        session,
        stamp_ns,
        received_monotonic,
        now_ros_ns,
        now_monotonic,
        persons,
        coverage_complete,
    ):
        """
        Process one new source frame atomically.

        Preserve individual identities and distinguish current
        observations from unresolved historical participants.
        """
        self._check_time(now_ros_ns, now_monotonic)

        session = str(session).strip()

        if (
            not session
            or session in self._retired_sessions
        ):
            raise ValueError(
                "Invalid or retired source session"
            )

        if (
            isinstance(stamp_ns, bool)
            or not isinstance(stamp_ns, int)
            or stamp_ns <= 0
        ):
            raise ValueError("Invalid source stamp")

        if not isinstance(coverage_complete, bool):
            raise ValueError(
                "coverage_complete must be boolean"
            )

        if (
            not math.isfinite(received_monotonic)
            or not (
                0
                <= now_monotonic - received_monotonic
                <= self.maximum_gap
            )
        ):
            raise ValueError(
                "Stale or invalid receipt time"
            )

        if not (
            -0.1
            <= (now_ros_ns - stamp_ns) / 1e9
            <= self.maximum_gap
        ):
            raise ValueError(
                "Stale or future source frame"
            )

        if (
            session == self._session
            and stamp_ns <= self._stamp
        ):
            raise ValueError(
                "Duplicate or reordered source frame"
            )

        staged = {}

        for person in persons:
            key = person.key

            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or any(
                    not isinstance(value, str)
                    or not value.strip()
                    or value != value.strip()
                    for value in key[:2]
                )
                or isinstance(key[2], bool)
                or not isinstance(key[2], int)
                or key[2] < 0
                or key[0] != session
                or key in staged
                or not math.isfinite(person.distance)
                or person.distance < 0
            ):
                raise ValueError(
                    "Invalid individual observation"
                )

            staged[key] = float(person.distance)

        # Commit only after every input observation is valid.
        if self._session and session != self._session:
            self._retired_sessions.add(self._session)

        self._session = session
        self._stamp = stamp_ns
        self._received = received_monotonic
        self._coverage = coverage_complete
        self._source_valid = True

        transitions = {}

        # Missing participants remain unresolved.
        # Their absence never establishes geometric clearance.
        for key, record in self._records.items():
            if key not in staged:
                transitions[key] = self._missing(
                    record,
                    now_monotonic,
                )

        for key, distance in staged.items():
            record = self._records.setdefault(
                key,
                _Record(),
            )

            transitions[key] = self._observe(
                record,
                distance,
                stamp_ns,
                received_monotonic,
            )

        return self._result(
            now_ros_ns,
            now_monotonic,
            transitions,
        )
    
    def _missing(self, record, now):
        if record.missing_since is None:
            record.missing_since = now

        record.observed = False

        # Ces preuves exigent des observations consécutives.
        record.entry_sec = 0.0
        record.entry_samples = 0

        record.clear_sec = 0.0
        record.clear_samples = 0

        record.reacquire_sec = 0.0
        record.reacquire_samples = 0

        # conflict_sec est volontairement conservé :
        # ni remise à zéro, ni ajout de temps absent.
        elapsed = now - record.missing_since
        transition = ""

        if elapsed >= self.missing_grace:
            record.observation = "LOST"

            if not record.loss_reported:
                transition = "OBSERVATION_LOST"

            record.loss_reported = True

        else:
            record.observation = "TEMPORARILY_MISSING"

        if (
            record.unresolved
            and elapsed >= self.retirement_duration
            and not record.retired
        ):
            record.retired = True
            transition = "RETIRED"

        if record.unresolved:
            record.conflict = "UNRESOLVED"

        return transition

    def _observe(
        self,
        record,
        distance,
        stamp,
        received,
    ):
        dt = (
            (stamp - record.last_stamp) / 1e9
            if record.last_stamp
            else 0.0
        )

        continuous = bool(
            record.observed
            and 0 < dt <= self.maximum_gap
            and record.last_received is not None
            and (
                0
                <= received - record.last_received
                <= self.maximum_gap
            )
        )

        previous_distance = record.last_distance

        # Une interruption de source casse aussi la continuité,
        # même sans message explicite indiquant une absence.
        if (
            record.last_stamp
            and not continuous
            and record.observed
        ):
            self._missing(record, received)
            record.loss_reported = True

        record.observed = True
        record.last_stamp = stamp
        record.last_received = received
        record.last_distance = distance
        record.missing_since = None

        transition = ""

        if (
            record.loss_reported
            and record.unresolved
            and not record.retired
        ):
            record.reacquire_samples = (
                record.reacquire_samples + 1
                if continuous
                else 1
            )

            record.reacquire_sec = (
                record.reacquire_sec + dt
                if continuous
                else 0.0
            )

            ready = bool(
                record.reacquire_samples >= 2
                and (
                    record.reacquire_sec + 1e-9
                    >= self.reacquisition_duration
                )
            )

            record.observation = (
                "OBSERVED"
                if ready
                else "REACQUIRING"
            )

            if ready:
                record.loss_reported = False
                transition = "OBSERVATION_REACQUIRED"

        else:
            record.observation = "OBSERVED"

        if distance <= self.entry_distance:
            was_inside = bool(
                previous_distance is not None
                and previous_distance <= self.entry_distance
            )

            record.entry_samples = (
                record.entry_samples + 1
                if continuous and was_inside
                else 1
            )

            record.entry_sec = (
                record.entry_sec + dt
                if continuous and was_inside
                else 0.0
            )

            if (
                (
                    not record.unresolved
                    or record.retired
                )
                and record.entry_samples >= 2
                and (
                    record.entry_sec + 1e-9
                    >= self.entry_duration
                )
            ):
                record.episode = uuid.uuid4().hex

                record.unresolved = True
                record.retired = False
                record.loss_reported = False

                record.conflict_sec = record.entry_sec
                record.observation = "OBSERVED"

                transition = "INTRUSION_CONFIRMED"

            elif (
                record.unresolved
                and not record.retired
                and continuous
                and was_inside
            ):
                record.conflict_sec += dt

        else:
            record.entry_sec = 0.0
            record.entry_samples = 0

        if record.unresolved:
            
            # Une réapparition ne suffit pas à confirmer
            # le dégagement de l'épisode précédent.
            if record.observation != "OBSERVED" or record.retired:
                record.clear_sec = 0.0
                record.clear_samples = 0
                record.conflict = "UNRESOLVED"
                return transition
            
            if distance > self.release_distance:
                was_outside = bool(
                    previous_distance is not None
                    and previous_distance > self.release_distance
                )

                record.clear_samples = (
                    record.clear_samples + 1
                    if continuous and was_outside
                    else 1
                )

                record.clear_sec = (
                    record.clear_sec + dt
                    if continuous and was_outside
                    else 0.0
                )

                record.conflict = "CLEAR_PENDING"

                if (
                    record.clear_samples >= 2
                    and (
                        record.clear_sec + 1e-9
                        >= self.clear_duration
                    )
                ):
                    record.unresolved = False
                    record.retired = False

                    record.conflict = "CLEAR"
                    record.conflict_sec = 0.0

                    transition = "CLEAR"

            else:
                record.clear_sec = 0.0
                record.clear_samples = 0

                record.conflict = (
                    "CONFLICT"
                    if not record.retired
                    else "UNRESOLVED"
                )

        else:
            record.conflict = (
                "CLEAR"
                if record.episode
                else "NOT_EVALUATED"
            )

        return transition

    def tick(self, *, now_ros_ns, now_monotonic):
        self._check_time(now_ros_ns, now_monotonic)

        transitions = {}

        for key, record in self._records.items():
            expired = bool(
                record.last_received is None
                or (
                    now_monotonic - record.last_received
                    > self.maximum_gap
                )
                or not (
                    -0.1
                    <= (now_ros_ns - record.last_stamp) / 1e9
                    <= self.maximum_gap
                )
            )

            if expired or not record.observed:
                transitions[key] = self._missing(
                    record,
                    now_monotonic,
                )

        return self._result(
            now_ros_ns,
            now_monotonic,
            transitions,
        )

    def unavailable(self, *, now_ros_ns, now_monotonic):
        """
        Signaler une source ou un appariement inutilisable.

        Une couverture partielle avec des observations valides
        doit passer par update(), pas par cette méthode.
        """
        self._check_time(now_ros_ns, now_monotonic)

        self._source_valid = False

        transitions = {
            key: self._missing(record, now_monotonic)
            for key, record in self._records.items()
        }

        return self._result(
            now_ros_ns,
            now_monotonic,
            transitions,
        )

    def _result(
        self,
        now_ros_ns,
        now_monotonic,
        transitions,
    ):
        self._last_now = now_monotonic
        rows = []

        for key, record in sorted(self._records.items()):
            signature = (
                record.episode,
                record.unresolved,
                record.retired,
                record.observation,
                record.conflict,
                record.observed,
                bool(
                    record.unresolved
                    and not record.retired
                    and record.conflict_sec + 1e-9
                    >= self.persistence_duration
                ),
            )
            if signature != record.signature:
                record.revision += 1
                record.signature = signature

            rows.append(
                ParticipantSituation(
                    key=key,
                    episode_id=record.episode,
                    revision=record.revision,
                    unresolved=record.unresolved,
                    observation_status=record.observation,
                    conflict_status=record.conflict,
                    transition=transitions.get(key, ""),
                    observed_now=record.observed,
                    observation_stamp_ns=record.last_stamp,
                    last_distance=record.last_distance,
                    conflict_observed_sec=record.conflict_sec,
                    persistent=bool(
                        record.unresolved
                        and not record.retired
                        and (
                            record.conflict_sec + 1e-9
                            >= self.persistence_duration
                        )
                    ),
                    retired=record.retired,
                )
            )

        fresh = bool(
            self._source_valid
            and self._received is not None
            and (
                0
                <= now_monotonic - self._received
                <= self.maximum_gap
            )
            and (
                -0.1
                <= (now_ros_ns - self._stamp) / 1e9
                <= self.maximum_gap
            )
        )

        return IndividualSituation(
            participants=tuple(rows),
            source_fresh=fresh,
            source_coverage_complete=bool(
                fresh and self._coverage
            ),
            source_session=self._session,
            source_stamp_ns=self._stamp,
        )