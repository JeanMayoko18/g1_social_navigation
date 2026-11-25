
#!/usr/bin/env python3
"""YOLO track IDs -> short-term ROS identities, without score fusion."""

import copy
import math
import re
import uuid

import rclpy
from rclpy.node import Node
from g1_social_perception.social_context_memory import (
    SocialIdentityRegistry,
)
from g1_social_msgs.msg import PersonDetectionArray
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


TRACK_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{6,64}:[0-9]+$"
)


def stamp_ns(stamp):
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    value = sec * 1_000_000_000 + nanosec

    if value <= 0:
        raise ValueError("ZERO_TIMESTAMP")

    return value


class PersonYoloIdentityNode(Node):

    def __init__(self):
        super().__init__("person_yolo_identity")

        self.declare_parameter(
            "input_topic",
            "/social_perception/detections_3d_odom",
        )
        self.declare_parameter(
            "output_topic",
            "/social_perception/detections_identified_odom",
        )
        self.declare_parameter(
            "expected_frame",
            "odom",
        )
        self.declare_parameter(
            "track_retention_sec",
            3.0,
        )

        self.input_topic = str(
            self.get_parameter("input_topic").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )
        self.expected_frame = str(
            self.get_parameter("expected_frame").value
        )
        self.retention = float(
            self.get_parameter("track_retention_sec").value
        )

        if (
            not math.isfinite(self.retention)
            or self.retention <= 0
        ):
            raise ValueError("INVALID_TRACK_RETENTION")

        if (
            not self.input_topic
            or not self.output_topic
            or self.input_topic == self.output_topic
        ):
            raise ValueError("INVALID_TOPICS")

        self.source_session = ""
        self.retired_source_sessions = set()
        self.tracking_session_id = uuid.uuid4().hex

        self.last_source_sequence = 0
        self.last_stamp_ns = 0

        # The social memory owns canonical identity allocation.
        # This node only adapts BoT-SORT observations to ROS.
        self.identity_registry = SocialIdentityRegistry()

        self.received = 0
        self.published = 0
        self.rejected = 0
        self.unresolved = 0
        self.reused = 0
        self.new_tracks = 0

        qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        self.publisher = self.create_publisher(
            PersonDetectionArray,
            self.output_topic,
            reliable_qos(depth=10),
        )

        self.subscription = self.create_subscription(
            PersonDetectionArray,
            self.input_topic,
            self.callback,
            qos,
        )

        self.timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self.get_logger().info(
            "[YOLO IDENTITY] "
            f"{self.input_topic} -> {self.output_topic}"
        )

    def callback(self, msg):
        self.received += 1

        try:
            # ============================================================
            # 1. Validate incoming spatial detection array
            # ============================================================

            if (
                msg.processing_stage
                != PersonDetectionArray.SPATIAL
                or msg.identity_evaluated
                or msg.tracking_session_id
                or msg.header.frame_id != self.expected_frame
            ):
                raise ValueError(
                    "EXPECTED_UNIDENTIFIED_SPATIAL"
                )

            source = str(
                msg.source_session_id
            ).strip()

            if not source:
                raise ValueError(
                    "EMPTY_SOURCE_SESSION"
                )

            if source in self.retired_source_sessions:
                raise ValueError(
                    "RETIRED_SOURCE_SESSION"
                )

            source_ns = stamp_ns(
                msg.header.stamp
            )

            if source_ns != stamp_ns(
                msg.image_header.stamp
            ):
                raise ValueError(
                    "IMAGE_TIMESTAMP_MISMATCH"
                )

            sequence = int(
                msg.source_sequence
            )

            if (
                sequence < 1
                or int(msg.source_detection_count)
                != len(msg.candidates)
            ):
                raise ValueError(
                    "INVALID_SOURCE_ENVELOPE"
                )

            if source == self.source_session:
                if (
                    source_ns <= self.last_stamp_ns
                    or sequence <= self.last_source_sequence
                ):
                    raise ValueError(
                        "NON_MONOTONIC_SOURCE"
                    )

            # ============================================================
            # 2. Validate every candidate before modifying state
            # ============================================================

            indices = set()
            current_ids = set()

            for candidate in msg.candidates:

                index = int(
                    candidate.source_index
                )

                if index in indices:
                    raise ValueError(
                        "DUPLICATE_SOURCE_INDEX"
                    )

                indices.add(index)

                if (
                    candidate.person_id
                    or candidate.identity_confirmed
                    or int(candidate.identity_generation) != 0
                    or candidate.identity_state != "INSUFFICIENT"
                ):
                    raise ValueError(
                        "IDENTITY_ALREADY_ASSIGNED"
                    )

                raw = str(
                    candidate.source_detection.id
                ).strip()

                if TRACK_ID_PATTERN.fullmatch(raw):

                    if raw in current_ids:
                        raise ValueError(
                            "DUPLICATE_YOLO_TRACK_ID"
                        )

                    current_ids.add(raw)

            # ============================================================
            # 3. Start or continue tracking session
            # ============================================================

            if source != self.source_session:

                if self.source_session:
                    self.retired_source_sessions.add(
                        self.source_session
                    )

                self.source_session = source

                self.tracking_session_id = (
                    uuid.uuid4().hex
                )

                self.last_source_sequence = 0
                self.last_stamp_ns = 0

            self.identity_registry.start_session(
                source
            )

            # ============================================================
            # 4. Create output envelope
            # ============================================================

            output = copy.deepcopy(msg)

            output.processing_stage = (
                PersonDetectionArray.IDENTITY
            )

            output.identity_evaluated = True

            output.tracking_session_id = (
                self.tracking_session_id
            )

            all_resolved = True
            new_count = 0
            reused_count = 0
            unresolved_count = 0

            # ============================================================
            # 5. Resolve ALL candidates through social identity memory
            # ============================================================

            for candidate in output.candidates:

                raw = str(
                    candidate.source_detection.id
                ).strip()

                if not TRACK_ID_PATTERN.fullmatch(raw):

                    candidate.person_id = ""

                    candidate.identity_state = (
                        "INSUFFICIENT"
                    )

                    candidate.identity_confirmed = False
                    candidate.identity_generation = 0

                    unresolved_count += 1
                    all_resolved = False

                    continue

                # The tracker key is resolved by the social memory.
                # Same key, same source session -> same social ID,
                # even after an observation gap.

                person_id, state, generation = (
                    self.identity_registry.resolve(
                        tracker_key=raw,
                        observation_stamp_ns=source_ns,
                    )
                )

                candidate.person_id = person_id
                candidate.identity_state = state

                candidate.identity_confirmed = True

                candidate.identity_generation = (
                    generation
                )

                if state == "NEW":
                    new_count += 1
                else:
                    reused_count += 1

            # ============================================================
            # 6. Preserve upstream observation completeness
            # ============================================================

            output.source_observation_complete = bool(
                msg.source_observation_complete
                and all_resolved
            )

            # ============================================================
            # 7. Commit source progress and publish
            # ============================================================

            self.last_stamp_ns = source_ns
            self.last_source_sequence = sequence

            self.unresolved += unresolved_count
            self.new_tracks += new_count
            self.reused += reused_count

            self.publisher.publish(
                output
            )

            self.published += 1

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as exc:

            self.rejected += 1

            self.get_logger().warning(
                "[YOLO IDENTITY] Rejected: "
                + str(exc),
                throttle_duration_sec=2.0,
            )
            
    def log_diagnostics(self):
        self.get_logger().info(
            "[YOLO IDENTITY] "
            f"RX={self.received} "
            f"TX={self.published} "
            f"REJECTED={self.rejected} "
            f"NEW={self.new_tracks} "
            f"REUSED={self.reused} "
            f"UNRESOLVED={self.unresolved} "
            f"TRACKS={len(self.identity_registry)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonYoloIdentityNode()
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
