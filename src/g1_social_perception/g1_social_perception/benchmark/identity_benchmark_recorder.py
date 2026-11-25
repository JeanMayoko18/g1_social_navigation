
#!/usr/bin/env python3
"""
G1 identity benchmark recorder.

Records six independent tracking methods:
    M1, M2, M3, M4, BoT-SORT, BoT-SORT + ReID.

Inputs:
    M1-M4: g1_social_msgs/PersonDetectionArray
    BoT-SORT: std_msgs/String containing protocol JSON
    Timing: std_msgs/String containing protocol JSON

Output:
    JSONL files containing normalized observations and timing.

No identity comparison is performed here.
No ground-truth labels are inferred.
No image data, biometric descriptors or RGB crops are stored.

All method and session identifiers remain independent.
"""

import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from g1_social_msgs.msg import PersonDetectionArray

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


ABLATION_METHODS = ("M1", "M2", "M3", "M4")
BOTSORT_METHODS = ("botsort", "botsort_reid")

ALL_METHODS = ABLATION_METHODS + BOTSORT_METHODS


def timestamp_ns(stamp):
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    value = sec * 1_000_000_000 + nanosec

    if value <= 0:
        raise ValueError("ZERO_TIMESTAMP")

    return value


def finite_number(value):
    result = float(value)

    if not math.isfinite(result):
        raise ValueError("NON_FINITE_NUMBER")

    return result


def normalize_ablation(method, msg):
    """
    Normalize current identity decisions without interpreting
    confirmed, ambiguous or insufficient observations.
    """
    if not msg.identity_evaluated:
        raise ValueError("IDENTITY_NOT_EVALUATED")

    if not msg.tracking_session_id:
        raise ValueError("MISSING_TRACKING_SESSION")

    if (
        msg.source_detection_count
        != len(msg.candidates)
    ):
        raise ValueError("SOURCE_CANDIDATE_COUNT_MISMATCH")

    source_stamp_ns = timestamp_ns(
        msg.image_header.stamp
    )

    if source_stamp_ns != timestamp_ns(msg.header.stamp):
        raise ValueError("SOURCE_TIMESTAMP_MISMATCH")

    observations = []
    indices = set()

    for candidate in msg.candidates:
        index = int(candidate.source_index)

        if index in indices:
            raise ValueError("DUPLICATE_SOURCE_INDEX")

        indices.add(index)

        box = candidate.source_detection.bbox

        cx = finite_number(box.center.position.x)
        cy = finite_number(box.center.position.y)
        width = finite_number(box.size.x)
        height = finite_number(box.size.y)

        if width <= 0 or height <= 0:
            raise ValueError("INVALID_BOUNDING_BOX")

        observations.append({
            "source_index": index,
            "track_id": (
                str(candidate.person_id)
                if candidate.person_id else None
            ),
            "identity_state": str(
                candidate.identity_state
            ),
            "confirmed": bool(
                candidate.identity_confirmed
            ),
            "generation": int(
                candidate.identity_generation
            ),
            "confidence": finite_number(
                candidate.source_detection.score
            ),
            "bbox_xyxy": [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
        })

    return {
        "schema_version": 1,
        "record_type": "observation",
        "method": method,
        "tracking_session_id": str(
            msg.tracking_session_id
        ),
        "source_session_id": str(
            msg.source_session_id
        ),
        "source_sequence": int(
            msg.source_sequence
        ),
        "source_stamp_ns": source_stamp_ns,
        "source_frame_id": str(
            msg.image_header.frame_id
        ),
        "source_observation_complete": bool(
            msg.source_observation_complete
        ),
        "source_detection_count": int(
            msg.source_detection_count
        ),
        "detection_count": len(observations),
        "detections": observations,
    }


def normalize_botsort(expected_method, payload):
    """
    Normalize worker output without making up
    missing confirmations or identity generations.
    """
    if not isinstance(payload, dict):
        raise ValueError("INVALID_BOTSORT_PAYLOAD")

    if payload.get("schema_version") != 1:
        raise ValueError("UNSUPPORTED_BOTSORT_SCHEMA")

    if payload.get("method") != expected_method:
        raise ValueError("BOTSORT_METHOD_MISMATCH")

    if payload.get("status") != "OK":
        raise ValueError("BOTSORT_RESULT_NOT_OK")

    stamp = int(payload["source_stamp_ns"])

    if stamp <= 0:
        raise ValueError("INVALID_SOURCE_STAMP")

    observations = []

    for candidate in payload["detections"]:
        bbox = [
            finite_number(value)
            for value in candidate["bbox_xyxy"]
        ]

        if len(bbox) != 4:
            raise ValueError("INVALID_BBOX_DIMENSION")

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("INVALID_BBOX_GEOMETRY")

        track_id = candidate.get("track_id")

        observations.append({
            "source_index": None,
            "track_id": (
                None if track_id is None
                else str(track_id)
            ),
            "identity_state": (
                "TRACKED"
                if track_id is not None
                else "UNASSIGNED"
            ),
            "confirmed": None,
            "generation": None,
            "confidence": finite_number(
                candidate["confidence"]
            ),
            "bbox_xyxy": bbox,
        })

    if len(observations) != int(payload["detection_count"]):
        raise ValueError("BOTSORT_COUNT_MISMATCH")

    return {
        "schema_version": 1,
        "record_type": "observation",
        "method": expected_method,
        "tracking_session_id": str(
            payload["tracking_session_id"]
        ),
        "source_session_id": str(
            payload["source_session_id"]
        ),
        "source_sequence": int(
            payload["source_sequence"]
        ),
        "source_stamp_ns": stamp,
        "source_frame_id": str(
            payload["source_frame_id"]
        ),
        "source_observation_complete": None,
        "source_detection_count": None,
        "detection_count": len(observations),
        "detections": observations,
        "processing_ms": finite_number(
            payload["processing_ms"]
        ),
        "detection_included_in_timing": bool(
            payload["detection_included_in_timing"]
        ),
        "reid_enabled": bool(
            payload["reid_enabled"]
        ),
    }


class IdentityBenchmarkRecorder(Node):

    def __init__(self):
        super().__init__("identity_benchmark_recorder")

        self.declare_parameter(
            "output_directory",
            "~/g1_ws/benchmark_results",
        )

        self.declare_parameter(
            "log_period_sec",
            10.0,
        )

        self.declare_parameter(
            "maximum_file_size_mb",
            512,
        )

        self.output_directory = Path(
            str(
                self.get_parameter(
                    "output_directory"
                ).value
            )
        ).expanduser()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        self.maximum_bytes = int(
            self.get_parameter(
                "maximum_file_size_mb"
            ).value
        ) * 1024 * 1024

        if self.maximum_bytes <= 0:
            raise ValueError("INVALID_FILE_SIZE_LIMIT")

        # Use a unique run directory to avoid mixing
        # observations from different experiments.
        self.run_id = (
            time.strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(os.getpid())
        )

        self.run_directory = (
            self.output_directory / self.run_id
        )
        self.run_directory.mkdir(
            mode=0o700
        )

        self.observation_path = (
            self.run_directory
            / "identity_observations.jsonl"
        )
        self.timing_path = (
            self.run_directory
            / "identity_timing.jsonl"
        )

        self.observation_file = self.observation_path.open(
            "x",
            encoding="utf-8",
        )

        self.timing_file = self.timing_path.open(
            "x",
            encoding="utf-8",
        )

        os.chmod(self.observation_path, 0o600)
        os.chmod(self.timing_path, 0o600)

        self.observation_counts = {
            method: 0 for method in ALL_METHODS
        }

        self.timing_counts = {}
        self.rejected = 0
        self.file_limit_reached = False

        self.benchmark_subscriptions = []

        for method in ABLATION_METHODS:
            topic = (
                "/social_perception/benchmark/ablation/"
                + method.lower()
            )

            qos = resolve_subscription_qos(
                self,
                topic,
                fallback_depth=10,
                prefer_sensor_data=False,
            )

            subscription = self.create_subscription(
                PersonDetectionArray,
                topic,
                lambda msg, method=method:
                    self.ablation_callback(method, msg),
                qos,
            )

            self.benchmark_subscriptions.append(subscription)

        for method in BOTSORT_METHODS:
            topic = (
                "/social_perception/benchmark/"
                + method
            )

            qos = resolve_subscription_qos(
                self,
                topic,
                fallback_depth=10,
                prefer_sensor_data=False,
            )

            subscription = self.create_subscription(
                String,
                topic,
                lambda msg, method=method:
                    self.botsort_callback(method, msg),
                qos,
            )

            self.benchmark_subscriptions.append(subscription)

        for topic in (
            "/social_perception/benchmark/ablation/timing",
            "/social_perception/benchmark/bridge_timing",
        ):
            qos = resolve_subscription_qos(
                self,
                topic,
                fallback_depth=100,
                prefer_sensor_data=False,
            )

            self.benchmark_subscriptions.append(
                self.create_subscription(
                    String,
                    topic,
                    lambda msg, topic=topic:
                        self.timing_callback(topic, msg),
                    qos,
                )
            )

        self.diagnostic_timer = self.create_timer(
            float(
                self.get_parameter(
                    "log_period_sec"
                ).value
            ),
            self.log_diagnostics,
        )

        self.get_logger().info(
            "[BENCHMARK RECORDER] "
            f"run={self.run_directory}"
        )

    def append_record(self, handle, record):
        if self.file_limit_reached:
            return False

        line = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
        ) + "\n"

        encoded_length = len(
            line.encode("utf-8")
        )

        if (
            handle.tell() + encoded_length
            > self.maximum_bytes
        ):
            self.file_limit_reached = True

            self.get_logger().error(
                "[BENCHMARK RECORDER] "
                "File size limit reached; "
                "recording stopped."
            )
            return False

        handle.write(line)
        handle.flush()
        return True

    def save_observation(self, method, record):
        record["recorded_at_ns"] = time.time_ns()

        if self.append_record(
            self.observation_file,
            record,
        ):
            self.observation_counts[method] += 1

    def ablation_callback(self, method, msg):
        try:
            record = normalize_ablation(
                method,
                msg,
            )
            self.save_observation(method, record)

        except Exception as exc:
            self.rejected += 1
            self.get_logger().error(
                f"[RECORDER {method}] "
                f"{type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )

    def botsort_callback(self, method, msg):
        try:
            payload = json.loads(msg.data)

            record = normalize_botsort(
                method,
                payload,
            )
            self.save_observation(method, record)

        except Exception as exc:
            self.rejected += 1
            self.get_logger().error(
                f"[RECORDER {method}] "
                f"{type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )

    def timing_callback(self, topic, msg):
        try:
            payload = json.loads(msg.data)

            if not isinstance(payload, dict):
                raise ValueError("INVALID_TIMING_PAYLOAD")

            record = {
                "schema_version": 1,
                "record_type": "timing",
                "source_topic": topic,
                "recorded_at_ns": time.time_ns(),
                "data": payload,
            }

            if self.append_record(
                self.timing_file,
                record,
            ):
                self.timing_counts[topic] = (
                    self.timing_counts.get(topic, 0) + 1
                )

        except Exception as exc:
            self.rejected += 1
            self.get_logger().error(
                "[RECORDER TIMING] "
                f"{type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )

    def log_diagnostics(self):
        counts = " | ".join(
            f"{method}={count}"
            for method, count
            in self.observation_counts.items()
        )

        self.get_logger().info(
            "[BENCHMARK RECORDER] "
            + counts
            + f" | REJECTED={self.rejected}"
            + f" | LIMIT_REACHED={self.file_limit_reached}"
        )

    def destroy_node(self):
        try:
            self.observation_file.close()
            self.timing_file.close()
        finally:
            return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = IdentityBenchmarkRecorder()
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
