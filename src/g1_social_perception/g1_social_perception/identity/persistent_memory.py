
#!/usr/bin/env python3
"""
Persistent, non-biometric identity history for G1.

PURPOSE
    Save confirmed tracking records across process restarts.

IMPORTANT
    Persistent history is NOT persistent person recognition.
    A historical person_id must never be automatically assigned
    to a new observation just because it exists in this database.

STORAGE
    SQLite, local file, transaction-based writes.
    No image, facial embedding, appearance descriptor,
    position or raw depth is stored.

CLOCKS
    source_stamp_ns: original ROS observation timestamp.
    saved_at_ns: UTC system wall-clock timestamp.

POLICY
    Saving is disabled unless explicitly enabled.
    Records have a bounded retention period.
    Individual records and the entire database can be cleared.
"""

import os
import sqlite3
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistentMemoryConfig:
    database_path: str
    enabled: bool = False
    retention_days: int = 7
    maximum_records: int = 10000

    def __post_init__(self):
        if not self.database_path or not self.database_path.strip():
            raise ValueError("EMPTY_DATABASE_PATH")

        if type(self.enabled) is not bool:
            raise ValueError("INVALID_PERSISTENCE_ENABLED")

        if type(self.retention_days) is not int:
            raise ValueError("INVALID_RETENTION_DAYS")

        if not 1 <= self.retention_days <= 365:
            raise ValueError("RETENTION_DAYS_OUT_OF_RANGE")

        if type(self.maximum_records) is not int:
            raise ValueError("INVALID_MAXIMUM_RECORDS")

        if self.maximum_records < 1:
            raise ValueError("INVALID_MAXIMUM_RECORDS")


@dataclass(frozen=True)
class PersistentIdentity:
    tracking_session_id: str
    person_id: str
    generation: int
    observation_count: int
    last_observation_ns: int
    saved_at_ns: int


class PersistentIdentityMemory:
    """
    Thread-safe SQLite store for confirmed identity history.

    Each identity is uniquely scoped by:
        (tracking_session_id, person_id).

    There is deliberately no cross-session identity matching.
    """

    def __init__(self, config: PersistentMemoryConfig):
        if not isinstance(config, PersistentMemoryConfig):
            raise TypeError("INVALID_PERSISTENT_MEMORY_CONFIG")

        self.config = config
        self._lock = threading.RLock()
        self._connection = None

        if config.enabled:
            self._open()

    @property
    def enabled(self):
        return self.config.enabled

    def _open(self):
        path = Path(self.config.database_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Refuse a symbolic link to avoid redirecting the database
        # to an unexpected file.
        if path.is_symlink():
            raise ValueError("DATABASE_SYMLINK_NOT_ALLOWED")

        # Apply owner-only permissions when creating the database.
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        fd = os.open(str(path), flags, 0o600)

        try:
            if not path.is_file():
                raise ValueError("DATABASE_NOT_REGULAR_FILE")

            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)

        connection = sqlite3.connect(
            str(path),
            timeout=5.0,
            check_same_thread=False,
        )

        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                )
            """)

            row = connection.execute(
                "SELECT version FROM schema_info LIMIT 1"
            ).fetchone()

            if row is None:
                connection.execute(
                    "INSERT INTO schema_info(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif row[0] != SCHEMA_VERSION:
                raise ValueError("UNSUPPORTED_MEMORY_SCHEMA_VERSION")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS identity_history (
                    tracking_session_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    generation INTEGER NOT NULL
                        CHECK(generation >= 1),
                    observation_count INTEGER NOT NULL
                        CHECK(observation_count >= 1),
                    last_observation_ns INTEGER NOT NULL
                        CHECK(last_observation_ns > 0),
                    saved_at_ns INTEGER NOT NULL
                        CHECK(saved_at_ns > 0),
                    PRIMARY KEY (
                        tracking_session_id,
                        person_id
                    )
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    identity_history_saved_at_idx
                ON identity_history(saved_at_ns)
            """)

            connection.commit()

        except Exception:
            connection.close()
            raise

        self._connection = connection

        # Protect the database and its SQLite sidecar files
        # through owner-only directory permissions.
        self._purge_locked()

    def _require_enabled(self):
        if not self.enabled:
            raise RuntimeError("PERSISTENT_MEMORY_DISABLED")

        if self._connection is None:
            raise RuntimeError("PERSISTENT_MEMORY_CLOSED")

    def _purge_locked(self):
        if self._connection is None:
            return 0

        cutoff_ns = (
            time.time_ns()
            - self.config.retention_days
            * 24 * 60 * 60 * 1_000_000_000
        )

        with self._connection:
            result = self._connection.execute(
                """
                DELETE FROM identity_history
                WHERE saved_at_ns < ?
                """,
                (cutoff_ns,),
            )

        return result.rowcount

    def purge_expired(self):
        if not self.enabled:
            return 0

        with self._lock:
            self._require_enabled()
            return self._purge_locked()

    def save_confirmed(
        self,
        *,
        tracking_session_id: str,
        person_id: str,
        generation: int,
        observation_count: int,
        last_observation_ns: int,
    ) -> bool:
        """
        Persist one already-confirmed tracking identity.

        The caller must only invoke this method for a genuinely
        confirmed track. No association or confirmation is
        performed by this storage layer.

        Returns False when persistence is disabled.
        """

        if not self.enabled:
            return False

        if not isinstance(tracking_session_id, str):
            raise ValueError("INVALID_TRACKING_SESSION")

        if not tracking_session_id.strip():
            raise ValueError("EMPTY_TRACKING_SESSION")

        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError("INVALID_PERSON_ID")

        for name, value in (
            ("generation", generation),
            ("observation_count", observation_count),
            ("last_observation_ns", last_observation_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"INVALID_{name.upper()}")

        saved_at_ns = time.time_ns()

        with self._lock:
            self._require_enabled()

            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO identity_history (
                        tracking_session_id,
                        person_id,
                        generation,
                        observation_count,
                        last_observation_ns,
                        saved_at_ns
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        tracking_session_id, person_id
                    )
                    DO UPDATE SET
                        generation = excluded.generation,
                        observation_count =
                            excluded.observation_count,
                        last_observation_ns =
                            excluded.last_observation_ns,
                        saved_at_ns = excluded.saved_at_ns
                    WHERE
                        excluded.generation >
                            identity_history.generation
                        OR (
                            excluded.generation =
                                identity_history.generation
                            AND excluded.last_observation_ns >=
                                identity_history.last_observation_ns
                        )
                    """,
                    (
                        tracking_session_id,
                        person_id,
                        generation,
                        observation_count,
                        last_observation_ns,
                        saved_at_ns,
                    ),
                )

                # Retention is enforced on every successful write.
                cutoff_ns = (
                    saved_at_ns
                    - self.config.retention_days
                    * 24 * 60 * 60 * 1_000_000_000
                )

                self._connection.execute(
                    """
                    DELETE FROM identity_history
                    WHERE saved_at_ns < ?
                    """,
                    (cutoff_ns,),
                )

                self._connection.execute(
                    """
                    DELETE FROM identity_history
                    WHERE rowid IN (
                        SELECT rowid
                        FROM identity_history
                        ORDER BY saved_at_ns DESC,
                                 tracking_session_id DESC,
                                 person_id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (self.config.maximum_records,),
                )

        return True

    @staticmethod
    def _to_record(row):
        return PersistentIdentity(
            tracking_session_id=row[0],
            person_id=row[1],
            generation=row[2],
            observation_count=row[3],
            last_observation_ns=row[4],
            saved_at_ns=row[5],
        )

    def get(
        self,
        tracking_session_id: str,
        person_id: str,
    ) -> Optional[PersistentIdentity]:
        if not self.enabled:
            return None

        with self._lock:
            self._require_enabled()
            self._purge_locked()

            row = self._connection.execute(
                """
                SELECT
                    tracking_session_id,
                    person_id,
                    generation,
                    observation_count,
                    last_observation_ns,
                    saved_at_ns
                FROM identity_history
                WHERE tracking_session_id = ?
                  AND person_id = ?
                """,
                (tracking_session_id, person_id),
            ).fetchone()

            return None if row is None else self._to_record(row)

    def recent(self, limit=100) -> Tuple[PersistentIdentity, ...]:
        if not self.enabled:
            return ()

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("INVALID_HISTORY_LIMIT")

        with self._lock:
            self._require_enabled()
            self._purge_locked()

            rows = self._connection.execute(
                """
                SELECT
                    tracking_session_id,
                    person_id,
                    generation,
                    observation_count,
                    last_observation_ns,
                    saved_at_ns
                FROM identity_history
                ORDER BY saved_at_ns DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return tuple(self._to_record(row) for row in rows)

    def delete_identity(
        self,
        tracking_session_id: str,
        person_id: str,
    ) -> int:
        if not self.enabled:
            return 0

        with self._lock:
            self._require_enabled()

            with self._connection:
                cursor = self._connection.execute(
                    """
                    DELETE FROM identity_history
                    WHERE tracking_session_id = ?
                      AND person_id = ?
                    """,
                    (tracking_session_id, person_id),
                )

            return cursor.rowcount

    def clear(self) -> int:
        """Delete every stored identity-history row."""
        if not self.enabled:
            return 0

        with self._lock:
            self._require_enabled()

            with self._connection:
                result = self._connection.execute(
                    "DELETE FROM identity_history"
                )

            return result.rowcount

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
