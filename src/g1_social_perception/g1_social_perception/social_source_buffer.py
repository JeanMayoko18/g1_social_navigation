"""Bounded source pairing without social or navigation policy."""

from copy import deepcopy
from dataclasses import dataclass
import math
from g1_social_perception.individual_social_evaluator import (
    IndividualSocialEvaluator,
    ObservedPerson,
)


@dataclass(frozen=True)
class SourcePair:
    key: tuple
    status: str

    persons: object
    zones: object

    first_received_monotonic: float
    persons_received_monotonic: object
    zones_received_monotonic: object


class SocialSourceBuffer:
    def __init__(
        self,
        expected_frame,
        freshness_sec=1.0,
        pairing_timeout_sec=0.20,
        max_pending=32,
    ):
        if not expected_frame:
            raise ValueError(
                "expected_frame must be non-empty."
            )

        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                freshness_sec,
                pairing_timeout_sec,
            )
        ):
            raise ValueError(
                "Buffer durations must be finite and positive."
            )

        if pairing_timeout_sec >= freshness_sec:
            raise ValueError(
                "pairing_timeout_sec must be smaller than freshness_sec."
            )

        if (
            not isinstance(max_pending, int)
            or isinstance(max_pending, bool)
            or max_pending < 1
        ):
            raise ValueError(
                "max_pending must be a positive integer."
            )

        self.expected_frame = expected_frame
        self.freshness_sec = freshness_sec
        self.pairing_timeout_sec = pairing_timeout_sec
        self.max_pending = max_pending

        self._pending = {}
        self._session = None
        self._retired_sessions = set()
        self._watermark = 0

        self.rejected_count = 0
        self.discarded_on_session_change = 0
        self.last_rejection = ""

    def _reject(self, reason):
        self.rejected_count += 1
        self.last_rejection = reason
        return False

    def push(
        self,
        kind,
        msg,
        now_ros_ns,
        now_monotonic,
    ):
        """
        Copy one source message.

        Duplicate arrivals never renew a pairing deadline.
        Only tracked-person arrays establish the active producer session.
        """
        if kind not in {"persons", "zones"}:
            raise ValueError(
                "kind must be 'persons' or 'zones'."
            )

        try:
            session = str(
                msg.tracking_session_id
            ).strip()

            frame = str(
                msg.header.frame_id
            )

            sec = int(msg.header.stamp.sec)
            nanosec = int(msg.header.stamp.nanosec)

            stamp = (
                sec * 1_000_000_000
                + nanosec
            )

            age = (
                now_ros_ns - stamp
            ) / 1e9

            if (
                not session
                or frame != self.expected_frame
                or sec < 0
                or not 0 <= nanosec < 1_000_000_000
                or stamp <= 0
                or not math.isfinite(now_monotonic)
                or not -0.1 <= age <= self.freshness_sec
            ):
                return self._reject(
                    "INVALID_SOURCE"
                )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return self._reject(
                "INVALID_SOURCE"
            )

        if session in self._retired_sessions:
            return self._reject(
                "RETIRED_SESSION"
            )

        if (
            kind == "persons"
            and session != self._session
        ):
            if self._session is not None:
                self._retired_sessions.add(
                    self._session
                )

            retained = {
                key: entry
                for key, entry in self._pending.items()
                if key[0] == session
            }

            self.discarded_on_session_change += (
                len(self._pending)
                - len(retained)
            )

            self._pending = retained
            self._session = session
            self._watermark = 0

        if (
            session == self._session
            and stamp <= self._watermark
        ):
            return self._reject(
                "ALREADY_PROCESSED_OR_LATE"
            )

        key = (
            session,
            stamp,
            frame,
        )

        entry = self._pending.get(key)

        if entry is None:
            if len(self._pending) >= self.max_pending:
                return self._reject(
                    "BUFFER_FULL"
                )

            entry = {
                "first": now_monotonic,
            }

            self._pending[key] = entry

        if kind in entry:
            return self._reject(
                "DUPLICATE_SOURCE"
            )

        if (
            now_monotonic - entry["first"]
            >= self.pairing_timeout_sec
        ):
            return self._reject(
                "PAIRING_DEADLINE_EXPIRED"
            )

        entry[kind] = (
            deepcopy(msg),
            now_monotonic,
        )

        return True

    def take_ready(
        self,
        now_ros_ns,
        now_monotonic,
    ):
        """
        Return completed or expired frames.

        An incomplete pair is never interpreted as an empty observation.
        Older known pending frames are processed before newer ones.
        """
        results = []

        ordered_keys = sorted(
            self._pending,
            key=lambda key: (
                key[1],
                key[0],
                key[2],
            ),
        )

        for key in ordered_keys:
            entry = self._pending[key]

            age = (
                now_ros_ns - key[1]
            ) / 1e9

            expired = (
                now_monotonic - entry["first"]
                >= self.pairing_timeout_sec
            )

            stale = not (
                -0.1 <= age <= self.freshness_sec
            )

            complete = (
                "persons" in entry
                and "zones" in entry
            )

            if key[0] != self._session:
                if expired or stale:
                    del self._pending[key]

                    self._reject(
                        "UNESTABLISHED_SESSION_EXPIRED"
                    )

                continue

            if (
                not complete
                and not expired
                and not stale
            ):
                # Do not advance past an older known pending frame.
                break

            if stale:
                status = "STALE"
            elif complete:
                status = "MATCHED"
            else:
                status = "INCOMPLETE"

            persons, persons_at = entry.get(
                "persons",
                (None, None),
            )

            zones, zones_at = entry.get(
                "zones",
                (None, None),
            )

            results.append(
                SourcePair(
                    key=key,
                    status=status,
                    persons=persons,
                    zones=zones,
                    first_received_monotonic=entry["first"],
                    persons_received_monotonic=persons_at,
                    zones_received_monotonic=zones_at,
                )
            )

            self._watermark = max(
                self._watermark,
                key[1],
            )

            del self._pending[key]

        return tuple(results)