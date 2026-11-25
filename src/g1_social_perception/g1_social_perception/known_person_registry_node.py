
#!/usr/bin/env python3
"""
ROS 2 read-only status interface for known-person registry.

Responsibilities:
    - Open the local known-person registry.
    - Publish aggregate registry status.
    - Never publish durable person identifiers.
    - Never enroll or link persons automatically.

The node is administrative, not an identity engine.
"""

import json
import sqlite3
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from g1_social_perception.identity.known_person_registry import (
    KnownPersonRegistry,
)

from g1_qos_manager.profiles import reliable_qos


class KnownPersonRegistryNode(Node):

    def __init__(self):
        super().__init__("known_person_registry")

        self.declare_parameter(
            "database_path",
            "~/.local/share/g1_social_perception/"
            "known_persons.sqlite3",
        )

        self.declare_parameter(
            "retention_days",
            30,
        )

        self.declare_parameter(
            "status_period_sec",
            10.0,
        )

        self.declare_parameter(
            "status_topic",
            "/social_perception/known_person_registry/status",
        )

        database_path = str(
            Path(
                self.get_parameter("database_path").value
            ).expanduser()
        )

        retention_days = int(
            self.get_parameter("retention_days").value
        )

        status_period_sec = float(
            self.get_parameter("status_period_sec").value
        )

        if not 0.1 <= status_period_sec <= 3600.0:
            raise ValueError("INVALID_STATUS_PERIOD")

        self.registry = KnownPersonRegistry(
            database_path=database_path,
            retention_days=retention_days,
        )

        self.status_publisher = self.create_publisher(
            String,
            str(
                self.get_parameter("status_topic").value
            ),
            reliable_qos(depth=10),
        )

        self.status_timer = self.create_timer(
            status_period_sec,
            self.publish_status,
        )

        self.get_logger().info(
            "[KNOWN PERSON REGISTRY] "
            "Read-only ROS interface initialized"
        )

    def publish_status(self):
        """
        Publish aggregate counts only.

        Do not expose individual identities or tracking links.
        """
        try:
            # Apply the existing retention policy.
            self.registry.purge_expired()

            with self.registry.lock:
                known_count = self.registry.connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM known_persons
                    """
                ).fetchone()[0]

                linked_count = self.registry.connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tracking_links
                    """
                ).fetchone()[0]

            payload = {
                "schema_version": 1,
                "node": self.get_name(),
                "status": "READY",
                "known_person_count": known_count,
                "tracking_link_count": linked_count,
            }

            self.status_publisher.publish(
                String(
                    data=json.dumps(
                        payload,
                        allow_nan=False,
                    )
                )
            )

        except (
            sqlite3.Error,
            OSError,
            ValueError,
        ) as exc:
            self.get_logger().error(
                "[KNOWN PERSON REGISTRY] "
                + type(exc).__name__
                + ": "
                + str(exc),
                throttle_duration_sec=5.0,
            )

            self.status_publisher.publish(
                String(
                    data=json.dumps({
                        "schema_version": 1,
                        "node": self.get_name(),
                        "status": "UNAVAILABLE",
                    })
                )
            )

    def destroy_node(self):
        try:
            self.registry.close()
        finally:
            return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = KnownPersonRegistryNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
