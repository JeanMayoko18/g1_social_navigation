
#!/usr/bin/env python3
"""
Consent-based persistent registry of known persons.

Separates:
    - transient tracking identity: (session_id, person_id)
    - durable registered identity: known_person_id

No biometric templates, face images, appearance vectors,
or automated cross-session matching are stored.

Enrollment and linking require an explicit authorized action.
"""

import sqlite3
import threading
import time
import uuid

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnownPerson:
    known_person_id: str
    created_at_ns: int
    expires_at_ns: int


class KnownPersonRegistry:

    def __init__(
        self,
        database_path,
        retention_days=30,
    ):
        if (
            type(retention_days) is not int
            or not 1 <= retention_days <= 365
        ):
            raise ValueError("INVALID_RETENTION_DAYS")

        self.database_path = Path(
            database_path
        ).expanduser()

        self.retention_days = retention_days
        self.lock = threading.RLock()

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        self.connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )

        self.connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS known_persons (
                    known_person_id TEXT PRIMARY KEY,
                    created_at_ns INTEGER NOT NULL,
                    expires_at_ns INTEGER NOT NULL
                )
            """)

            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS tracking_links (
                    tracking_session_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    known_person_id TEXT NOT NULL,
                    linked_at_ns INTEGER NOT NULL,
                    PRIMARY KEY (
                        tracking_session_id,
                        person_id
                    ),
                    FOREIGN KEY (known_person_id)
                        REFERENCES known_persons(known_person_id)
                        ON DELETE CASCADE
                )
            """)

    def enroll(self, *, authorized=False):
        """
        Register one person after explicit authorization.

        Enrollment does not identify a currently visible person.
        """
        if authorized is not True:
            raise PermissionError("ENROLLMENT_NOT_AUTHORIZED")

        now = time.time_ns()

        expires = (
            now
            + self.retention_days
            * 24 * 60 * 60 * 1_000_000_000
        )

        person_id = "kp_" + uuid.uuid4().hex

        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO known_persons
                VALUES (?, ?, ?)
                """,
                (person_id, now, expires),
            )

        return person_id

    def link_tracking_identity(
        self,
        *,
        tracking_session_id,
        person_id,
        known_person_id,
        authorized=False,
    ):
        """
        Explicitly associate one tracking identity
        with an existing registered person.

        The caller must verify the association externally.
        """
        if authorized is not True:
            raise PermissionError("LINK_NOT_AUTHORIZED")

        if (
            not tracking_session_id
            or not person_id
            or not known_person_id
        ):
            raise ValueError("INVALID_IDENTITY_LINK")

        now = time.time_ns()

        with self.lock, self.connection:
            row = self.connection.execute(
                """
                SELECT expires_at_ns
                FROM known_persons
                WHERE known_person_id = ?
                """,
                (known_person_id,),
            ).fetchone()

            if row is None or row[0] <= now:
                raise ValueError("UNKNOWN_OR_EXPIRED_PERSON")

            existing = self.connection.execute(
                """
                SELECT known_person_id
                FROM tracking_links
                WHERE tracking_session_id = ?
                  AND person_id = ?
                """,
                (
                    tracking_session_id,
                    person_id,
                ),
            ).fetchone()

            if existing is not None:
                if existing[0] != known_person_id:
                    raise ValueError("CONFLICTING_IDENTITY_LINK")

                return

            self.connection.execute(
                """
                INSERT INTO tracking_links
                VALUES (?, ?, ?, ?)
                """,
                (
                    tracking_session_id,
                    person_id,
                    known_person_id,
                    now,
                ),
            )

    def resolve(
        self,
        tracking_session_id,
        person_id,
    ):
        """
        Resolve only an explicitly established link.
        Never infer identity from historical tracking IDs.
        """
        now = time.time_ns()

        with self.lock:
            row = self.connection.execute(
                """
                SELECT k.known_person_id
                FROM tracking_links AS t
                JOIN known_persons AS k
                  ON k.known_person_id = t.known_person_id
                WHERE t.tracking_session_id = ?
                  AND t.person_id = ?
                  AND k.expires_at_ns > ?
                """,
                (
                    tracking_session_id,
                    person_id,
                    now,
                ),
            ).fetchone()

        return row[0] if row else None

    def delete_person(self, known_person_id):
        with self.lock, self.connection:
            result = self.connection.execute(
                """
                DELETE FROM known_persons
                WHERE known_person_id = ?
                """,
                (known_person_id,),
            )

        return result.rowcount

    def purge_expired(self):
        now = time.time_ns()

        with self.lock, self.connection:
            result = self.connection.execute(
                """
                DELETE FROM known_persons
                WHERE expires_at_ns <= ?
                """,
                (now,),
            )

        return result.rowcount

    def close(self):
        with self.lock:
            self.connection.close()
