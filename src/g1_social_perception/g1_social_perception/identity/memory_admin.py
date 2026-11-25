
#!/usr/bin/env python3
"""
Administrative interface for G1 persistent identity history.

This tool does not:
- perform person recognition;
- assign identity labels;
- start ROS nodes;
- write appearance or biometric descriptors.

Database modifications are explicit administrative actions.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from g1_social_perception.identity.persistent_memory import (
    PersistentIdentityMemory,
    PersistentMemoryConfig,
)


DEFAULT_PATH = (
    "~/.local/share/g1_social_perception/"
    "identity_history.sqlite3"
)


def format_time(ns):
    return datetime.fromtimestamp(
        ns / 1_000_000_000,
        tz=timezone.utc,
    ).isoformat()


def main():
    parser = argparse.ArgumentParser(
        description="Manage G1 persistent identity history."
    )

    parser.add_argument(
        "--database",
        default=DEFAULT_PATH,
        help="SQLite database location.",
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--maximum-records",
        type=int,
        default=10000,
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    recent = sub.add_parser("recent")
    recent.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    get = sub.add_parser("get")
    get.add_argument("session")
    get.add_argument("person")

    delete = sub.add_parser("delete")
    delete.add_argument("session")
    delete.add_argument("person")
    delete.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly approve deletion.",
    )

    clear = sub.add_parser("clear")
    clear.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly approve deletion of all records.",
    )

    sub.add_parser("purge")

    args = parser.parse_args()

    database_path = Path(
        args.database
    ).expanduser()

    # Administrative reads must not create a database that
    # has never been activated.
    if not database_path.is_file():
        parser.error(
            "Database does not exist: "
            + str(database_path)
            + ". Persistence may still be disabled."
        )

    if args.command in {"delete", "clear"}:
        if not args.confirm:
            parser.error(
                "Destructive operation requires --confirm."
            )

    config = PersistentMemoryConfig(
        database_path=str(database_path),
        enabled=True,
        retention_days=args.retention_days,
        maximum_records=args.maximum_records,
    )

    with PersistentIdentityMemory(config) as memory:
        if args.command == "recent":
            for record in memory.recent(
                limit=args.limit
            ):
                print(
                    record.tracking_session_id,
                    record.person_id,
                    f"generation={record.generation}",
                    f"observations={record.observation_count}",
                    f"saved={format_time(record.saved_at_ns)}",
                )

        elif args.command == "get":
            record = memory.get(
                args.session,
                args.person,
            )
            if record is None:
                print("NOT_FOUND")
            else:
                print(record)

        elif args.command == "delete":
            count = memory.delete_identity(
                args.session,
                args.person,
            )
            print(f"DELETED={count}")

        elif args.command == "clear":
            count = memory.clear()
            print(f"DELETED={count}")

        elif args.command == "purge":
            count = memory.purge_expired()
            print(f"EXPIRED_DELETED={count}")


if __name__ == "__main__":
    main()
