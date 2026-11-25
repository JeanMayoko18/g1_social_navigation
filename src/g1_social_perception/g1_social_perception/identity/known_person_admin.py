
#!/usr/bin/env python3
"""
Local administration of known-person registrations.

A tracking identity can only be linked to a registered
person after:
    - a persisted, confirmed tracking record is found;
    - its generation is checked;
    - the registered identity is valid;
    - an operator explicitly confirms external verification.

Neither --confirm nor --verified-person is authentication.
No automatic cross-session recognition occurs here.
"""

import argparse
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

from g1_social_perception.identity.known_person_registry import (
    KnownPersonRegistry,
)

from g1_social_perception.identity.persistent_memory import (
    PersistentIdentityMemory,
    PersistentMemoryConfig,
)


DEFAULT_REGISTRY = (
    "~/.local/share/g1_social_perception/"
    "known_persons.sqlite3"
)

DEFAULT_HISTORY = (
    "~/.local/share/g1_social_perception/"
    "identity_history.sqlite3"
)


def format_time(timestamp_ns):
    return datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat()


def validate_database(path, *, must_exist=True):
    """
    Basic local file checks.

    These checks do not replace OS-level access control.
    """
    if path.is_symlink():
        raise ValueError("DATABASE_SYMLINK_NOT_ALLOWED")

    if not path.exists():
        if must_exist:
            raise FileNotFoundError(
                f"DATABASE_NOT_FOUND:{path}"
            )
        return

    info = path.stat()

    if not stat.S_ISREG(info.st_mode):
        raise ValueError("DATABASE_NOT_REGULAR_FILE")

    if info.st_uid != os.getuid():
        raise PermissionError("DATABASE_NOT_OWNED_BY_USER")

    if info.st_mode & 0o077:
        raise PermissionError(
            "DATABASE_PERMISSIONS_TOO_BROAD:"
            + str(path)
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manage known-person registrations."
    )

    parser.add_argument(
        "--database",
        default=DEFAULT_REGISTRY,
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser("list")

    enroll = commands.add_parser("enroll")
    enroll.add_argument(
        "--confirm",
        action="store_true",
    )

    delete = commands.add_parser("delete")
    delete.add_argument("known_person_id")
    delete.add_argument(
        "--confirm",
        action="store_true",
    )

    purge = commands.add_parser("purge")
    purge.add_argument(
        "--confirm",
        action="store_true",
    )

    link = commands.add_parser(
        "link",
        help="Link a confirmed track to a registered person.",
    )

    link.add_argument("tracking_session_id")
    link.add_argument("person_id")
    link.add_argument("generation", type=int)
    link.add_argument("known_person_id")

    link.add_argument(
        "--history-database",
        default=DEFAULT_HISTORY,
    )

    link.add_argument(
        "--history-retention-days",
        type=int,
        default=7,
    )

    link.add_argument(
        "--history-maximum-records",
        type=int,
        default=10000,
    )

    link.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly approve the link operation.",
    )

    link.add_argument(
        "--verified-person",
        action="store_true",
        help=(
            "Declare that the association was externally "
            "verified with the person's authorization."
        ),
    )

    return parser


def list_people(registry):
    registry.purge_expired()

    with registry.lock:
        rows = registry.connection.execute(
            """
            SELECT
                known_person_id,
                created_at_ns,
                expires_at_ns
            FROM known_persons
            ORDER BY created_at_ns DESC
            """
        ).fetchall()

    print(f"KNOWN_PERSON_COUNT={len(rows)}")

    for person_id, created, expires in rows:
        print(
            f"ID={person_id} | "
            f"CREATED={format_time(created)} | "
            f"EXPIRES={format_time(expires)}"
        )


def validate_persisted_track(args):
    """
    Check that the exact track and generation were
    previously saved as confirmed by person_tracking.

    This does NOT prove the real-world person's identity.
    """
    if (
        not args.tracking_session_id.strip()
        or not args.person_id.strip()
        or args.generation < 1
    ):
        raise ValueError("INVALID_TRACKING_IDENTITY")

    history_path = Path(
        args.history_database
    ).expanduser()

    validate_database(history_path)

    config = PersistentMemoryConfig(
        database_path=str(history_path),
        enabled=True,
        retention_days=args.history_retention_days,
        maximum_records=args.history_maximum_records,
    )

    with PersistentIdentityMemory(config) as history:
        record = history.get(
            args.tracking_session_id,
            args.person_id,
        )

    if record is None:
        raise ValueError("TRACK_NOT_FOUND_IN_PERSISTENT_HISTORY")

    if record.generation != args.generation:
        raise ValueError(
            "TRACK_GENERATION_MISMATCH: "
            f"stored={record.generation}, "
            f"requested={args.generation}"
        )

    if record.observation_count < 1:
        raise ValueError("TRACK_HAS_NO_CONFIRMED_OBSERVATIONS")

    if record.last_observation_ns <= 0:
        raise ValueError("INVALID_TRACK_SOURCE_TIMESTAMP")

    return record


def link_person(registry, args):
    """
    Associate an explicitly verified tracking identity
    with one registered person.

    The registry rejects links to expired identities and
    conflicting mappings for the same tracking identity.
    """
    record = validate_persisted_track(args)

    registry.link_tracking_identity(
        tracking_session_id=record.tracking_session_id,
        person_id=record.person_id,
        known_person_id=args.known_person_id,
        authorized=True,
    )

    resolved_id = registry.resolve(
        record.tracking_session_id,
        record.person_id,
    )

    if resolved_id != args.known_person_id:
        raise RuntimeError("IDENTITY_LINK_VERIFICATION_FAILED")

    print("LINK_CREATED_OR_ALREADY_PRESENT")
    print(f"TRACKING_SESSION={record.tracking_session_id}")
    print(f"PERSON_ID={record.person_id}")
    print(f"GENERATION={record.generation}")
    print(f"KNOWN_PERSON_ID={resolved_id}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"enroll", "delete", "purge", "link"}:
        if not args.confirm:
            parser.error(
                "Operation requires explicit --confirm."
            )

    if args.command == "link" and not args.verified_person:
        parser.error(
            "Link requires --verified-person and --confirm. "
            "These flags are declarations, not authentication."
        )

    database = Path(args.database).expanduser()

    validate_database(
        database,
        must_exist=(args.command != "enroll"),
    )

    # Only the enrollment command may create the database.
    # Restrict permissions on newly created files.
    old_umask = os.umask(0o077)
    try:
        registry = KnownPersonRegistry(
            database_path=str(database),
            retention_days=args.retention_days,
        )
    finally:
        os.umask(old_umask)

    try:
        if args.command == "list":
            list_people(registry)

        elif args.command == "enroll":
            known_person_id = registry.enroll(
                authorized=True
            )

            print("ENROLLMENT_CREATED")
            print(f"KNOWN_PERSON_ID={known_person_id}")

        elif args.command == "delete":
            deleted = registry.delete_person(
                args.known_person_id
            )

            print(f"DELETED={deleted}")

        elif args.command == "purge":
            deleted = registry.purge_expired()

            print(f"EXPIRED_DELETED={deleted}")

        elif args.command == "link":
            link_person(registry, args)

    finally:
        registry.close()


if __name__ == "__main__":
    main()
