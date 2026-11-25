"""Associate actual odom observations using metric distance and appearance."""

import copy
import math
from dataclasses import dataclass

from .appearance import compare


@dataclass(frozen=True)
class Observation:
    position: object
    signature: object


@dataclass(frozen=True)
class Result:
    person_id: str = ""
    state: str = "INSUFFICIENT"
    confirmed: bool = False
    generation: int = 0


@dataclass
class Track:
    position: tuple
    signature: object
    stamp_ns: int
    receipt: float
    hits: int
    generation: int
    confirmed: bool
    signature_stamp_ns: int
    signature_receipt: float


class IdentityTracker3D:
    """Keep identities session-local; retained positions are never detections."""

    def __init__(
        self,
        session_id,
        active_sec=0.75,
        retention_sec=8.0,
        confirm_hits=3,
        max_tracks=128,
    ):
        if (
            not session_id
            or not math.isfinite(active_sec)
            or not math.isfinite(retention_sec)
            or not 0 < active_sec < retention_sec
            or type(confirm_hits) is not int
            or confirm_hits < 1
            or type(max_tracks) is not int
            or max_tracks < 1
        ):
            raise ValueError("Invalid identity configuration")

        self.session_id = session_id
        self.active_sec = active_sec
        self.retention_sec = retention_sec
        self.confirm_hits = confirm_hits
        self.max_tracks = max_tracks

        self.tracks = {}
        self.next_id = 1
        self.next_generation = 1
        self.last_stamp = 0
        self.last_receipt = None

    def process(self, observations, stamp_ns, receipt):
        """Validate and commit one source frame atomically."""
        if (
            type(stamp_ns) is not int
            or stamp_ns <= self.last_stamp
            or not math.isfinite(receipt)
            or (
                self.last_receipt is not None
                and receipt < self.last_receipt
            )
        ):
            raise ValueError(
                "Invalid source order or monotonic receipt time"
            )

        observations = tuple(observations)

        for observation in observations:
            if observation.position is not None and (
                len(observation.position) != 3
                or not all(
                    math.isfinite(value)
                    for value in observation.position
                )
            ):
                raise ValueError("Invalid odom position")

            compare(observation.signature, observation.signature)

        staged = copy.deepcopy(self)
        results = staged._process(observations, stamp_ns, receipt)

        staged.last_stamp = stamp_ns
        staged.last_receipt = receipt
        self.__dict__.update(staged.__dict__)

        return results

    def _process(self, observations, stamp, receipt):
        # Expired IDs are removed but never recycled within this session.
        self.tracks = {
            key: track
            for key, track in self.tracks.items()
            if (
                (stamp - track.stamp_ns) / 1e9 <= self.retention_sec
                and receipt - track.receipt <= self.retention_sec
            )
        }

        scores = {}

        for key, track in self.tracks.items():
            dt = (stamp - track.stamp_ns) / 1e9

            active = (
                dt <= self.active_sec
                and receipt - track.receipt <= self.active_sec
            )

            for index, observation in enumerate(observations):
                if observation.position is None:
                    continue

                distance = math.hypot(
                    observation.position[0] - track.position[0],
                    observation.position[1] - track.position[1],
                )

                similarity = compare(
                    track.signature,
                    observation.signature,
                    min_region_quality=0.50,
                )

                if active:
                    # Development gate: position uncertainty plus elapsed motion.
                    gate = min(1.50, 0.35 + 2.0 * dt)

                    if distance > gate:
                        continue

                    if similarity.score is None:
                        # Missing RGB supports only close geometric continuity.
                        if distance > 0.35:
                            continue

                        score = 1.0 - distance / gate

                    else:
                        if similarity.score < 0.60:
                            continue

                        score = (
                            0.65 * (1.0 - distance / gate)
                            + 0.35 * similarity.score
                        )

                    if score < 0.45:
                        continue

                else:
                    # Dormant returns require fresh evidence from both regions.
                    if (
                        not track.confirmed
                        or distance > 2.0
                        or (
                            stamp - track.signature_stamp_ns
                        ) / 1e9 > self.retention_sec
                        or (
                            receipt - track.signature_receipt
                        ) > self.retention_sec
                        or similarity.common_regions != 2
                        or similarity.score is None
                        or min(
                            similarity.upper_score,
                            similarity.lower_score,
                        ) < 0.90
                    ):
                        continue

                    score = (
                        0.85 * similarity.score
                        + 0.15 * (1.0 - distance / 2.0)
                    )

                scores[key, index] = (score, active)

        output = []

        for index, observation in enumerate(observations):
            if observation.position is None:
                output.append(Result())
                continue

            ranked = sorted(
                (
                    (value[0], key, value[1])
                    for (key, candidate_index), value in scores.items()
                    if candidate_index == index
                ),
                reverse=True,
            )

            if ranked:
                score, key, active = ranked[0]

                rivals = [
                    value[0]
                    for (candidate_key, candidate_index), value
                    in scores.items()
                    if candidate_key == key and candidate_index != index
                ]

                # Require a distinct best match in both directions.
                if (
                    (
                        len(ranked) > 1
                        and score - ranked[1][0] < 0.08
                    )
                    or (
                        rivals
                        and score - max(rivals) < 0.08
                    )
                ):
                    output.append(Result(state="AMBIGUOUS"))
                    continue

                track = self.tracks[key]

                continuous = (
                    active
                    and track.stamp_ns == self.last_stamp
                )

                hits = track.hits + 1 if continuous else 1
                generation = (
                    track.generation
                    if continuous
                    else self._generation()
                )

                state = "MATCHED" if continuous else "RETURNING"

                own = compare(
                    observation.signature,
                    observation.signature,
                    min_region_quality=0.50,
                )

                if own.common_regions == 2:
                    signature = observation.signature
                    signature_stamp = stamp
                    signature_receipt = receipt
                else:
                    signature = track.signature
                    signature_stamp = track.signature_stamp_ns
                    signature_receipt = track.signature_receipt

            else:
                if len(self.tracks) >= self.max_tracks:
                    output.append(Result())
                    continue

                key = "person_{}".format(self.next_id)
                self.next_id += 1

                hits = 1
                generation = self._generation()
                state = "NEW"

                signature = observation.signature
                signature_stamp = stamp
                signature_receipt = receipt

            confirmed = hits >= self.confirm_hits

            self.tracks[key] = Track(
                position=observation.position,
                signature=signature,
                stamp_ns=stamp,
                receipt=receipt,
                hits=hits,
                generation=generation,
                confirmed=confirmed,
                signature_stamp_ns=signature_stamp,
                signature_receipt=signature_receipt,
            )

            output.append(
                Result(key, state, confirmed, generation)
            )

        return tuple(output)

    def _generation(self):
        value = self.next_generation
        self.next_generation += 1
        return value