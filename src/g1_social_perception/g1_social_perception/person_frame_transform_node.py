#!/usr/bin/env python3

"""
Transform observed person positions into the tracking reference frame.

DATA FLOW
    PersonDepthFusionNode
      /social_perception/detections_camera [PersonDetectionArray]
        -> validate observation provenance and camera positions
        -> request target <- camera TF at the source observation timestamp
        -> retry briefly without blocking the executor
        -> transform available positions
        -> preserve all candidates and non-position features
      /social_perception/detections_odom [PersonDetectionArray]
        -> enriched person tracking

TRANSFORMATION
    p_target = R_target_source(t_source) * p_source + translation

PRESERVED
    Original YOLO detections and candidate order.
    Image header, source session and source sequence.
    Appearance, apparent dimensions, metric dimensions and depth evidence.
    Identity fields, which remain unevaluated at this stage.

CHANGED
    Array spatial frame.
    Available position values and their coordinate frames.
    Processing stage and observation completeness.

FAILURE
    Missing TF never produces an artificial empty observation.
    At the waiting deadline, available camera positions become UNAVAILABLE.
    Positions already unavailable retain their upstream failure reason.

TIMING
    Local durations and waiting deadlines use a monotonic performance clock.
    Retries never create new observations or renew source timestamps.
    Source ages require comparable source and ROS clocks.
"""


import copy
import json
import math
import time
import uuid

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import rclpy

from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from g1_social_msgs.msg import (
    FeatureMetadata,
    PersonDetectionArray,
    PositionFeature,
)

from g1_social_perception.mathematical_model.types import Vector3
from g1_social_perception.mathematical_model.geometry import (
    rotation_from_quaternion,
    transform_position,
)
from g1_social_perception.mathematical_model.validation import (
    integer,
    nonempty,
    positive,
    unit_interval,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# DATA / PENDING OBSERVATIONS AND TIMING
# =============================================================================


@dataclass
class PendingObservation:
    """
    One validated actual observation awaiting spatial processing.

    received_ns:
        Local callback-entry time, not source acquisition time.

    deadline_ns:
        Latest local time allowed for resolving the requested transform.

    trace:
        Processing evidence accumulated across nonblocking retries.
    """

    message: object
    received_ns: int
    deadline_ns: int
    trace: dict


def stamp_ns(stamp, allow_zero=False):
    """Validate and convert a ROS timestamp to integer nanoseconds."""
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    value = sec * 1_000_000_000 + nanosec

    if value == 0 and not allow_zero:
        raise ValueError("ZERO_TIMESTAMP")

    return value


@contextmanager
def measured_stage(durations, name):
    """Accumulate local stage duration, including repeated TF lookup attempts."""
    started_ns = time.perf_counter_ns()

    try:
        yield
    finally:
        durations[name] = durations.get(name, 0) + (
            time.perf_counter_ns() - started_ns
        )


# =============================================================================
# NODE
# =============================================================================


class PersonFrameTransformNode(Node):

    def __init__(self):
        super().__init__("person_frame_transform")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[PERSON TF] "
            f"input={self.input_topic} | "
            f"output={self.output_topic} | "
            f"target={self.target_frame} | "
            f"tf_wait_sec={self.tf_wait_timeout_sec}"
        )

    # =========================================================================
    # DATA / CONFIGURATION
    # =========================================================================

    def _configure(self):
        """Declare transport, bounded waiting and source-freshness settings."""
        defaults = {
            "input_topic": "/social_perception/detections_camera",
            "output_topic": "/social_perception/detections_odom",
            "timing_topic": "/social_perception/person_tf/timing",
            "target_frame": "odom",
            "tf_cache_sec": 10.0,
            "tf_wait_timeout_sec": 0.10,
            "tf_retry_period_sec": 0.02,
            "max_pending_observations": 32,
            "max_arrays_per_cycle": 4,
            "source_clock_comparable": False,
            "maximum_source_age_sec": 1.0,
            "future_tolerance_sec": 0.10,
            "log_period_sec": 5.0,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        for name in (
            "input_topic",
            "output_topic",
            "timing_topic",
            "target_frame",
        ):
            nonempty(name, getattr(self, name))

        if len({
            self.input_topic,
            self.output_topic,
            self.timing_topic,
        }) != 3:
            raise ValueError("Input, output and timing topics must differ")

        for name in (
            "tf_cache_sec",
            "tf_wait_timeout_sec",
            "tf_retry_period_sec",
            "maximum_source_age_sec",
            "log_period_sec",
        ):
            positive(name, getattr(self, name))

        if self.tf_retry_period_sec > self.tf_wait_timeout_sec:
            raise ValueError(
                "TF retry period must not exceed the TF waiting timeout"
            )

        for name in (
            "max_pending_observations",
            "max_arrays_per_cycle",
        ):
            integer(name, getattr(self, name), minimum=1)

        if (
            not math.isfinite(self.future_tolerance_sec)
            or self.future_tolerance_sec < 0.0
        ):
            raise ValueError("future_tolerance_sec must be non-negative")

        if type(self.source_clock_comparable) is not bool:
            raise ValueError("source_clock_comparable must be boolean")

    def _initialize_state(self):
        """Initialize ordered pending work, producer sessions and diagnostics."""
        self.pending = deque()

        self.active_source_session = ""
        self.retired_source_sessions = set()
        self.last_source_sequence = 0
        self.last_source_stamp_ns = None

        self.trace_session_id = uuid.uuid4().hex
        self.callback_sequence = 0

        self.arrays_received = 0
        self.arrays_rejected = 0
        self.arrays_discarded = 0
        self.arrays_published = 0
        self.tf_timeouts = 0
        self.transformed_positions = 0

        # Local retry deadlines must continue advancing if ROS time pauses.
        self.retry_clock = Clock(clock_type=ClockType.STEADY_TIME)

    # =========================================================================
    # INPUT / ROS AND TF INTERFACES
    # =========================================================================

    def _create_interfaces(self):
        """
        Create TF reception, perception endpoints and a nonblocking retry timer.

        Lookup calls use zero timeout.
        The executor remains available to receive additional TF messages.
        """
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=self.tf_cache_sec)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )
        log_qos_resolution(self, self.input_topic, input_qos)

        self.output_pub = self.create_publisher(
            PersonDetectionArray,
            self.output_topic,
            reliable_qos(depth=10),
        )
        self.timing_pub = self.create_publisher(
            String,
            self.timing_topic,
            reliable_qos(depth=100),
        )

        self.subscription = self.create_subscription(
            PersonDetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.retry_timer = self.create_timer(
            self.tf_retry_period_sec,
            self._drain_pending,
            clock=self.retry_clock,
        )

        self.diagnostics_timer = self.create_timer(
            self.log_period_sec,
            self.log_diagnostics,
            clock=self.retry_clock,
        )

    # =========================================================================
    # INPUT / OBSERVATION ADMISSION
    # =========================================================================

    def detections_callback(self, msg):
        """
        Validate and enqueue one observation without changing its source time.

        Source identity is preserved.
        Duplicate arrivals never renew a pending observation's deadline.
        """
        received_ns = time.perf_counter_ns()

        self.arrays_received += 1
        self.callback_sequence += 1

        trace = {
            "version": 1,
            "node": self.get_name(),
            "trace_session_id": self.trace_session_id,
            "callback_sequence": self.callback_sequence,
            "source_session_id": msg.source_session_id,
            "source_sequence": int(msg.source_sequence),
            "source_stamp_ns": None,
            "source_frame": msg.header.frame_id,
            "target_frame": self.target_frame,
            "callback_ros_ns": self.get_clock().now().nanoseconds,
            "source_clock_comparable": self.source_clock_comparable,
            "source_age_at_entry_sec": None,
            "source_age_before_publish_sec": None,
            "candidate_count": len(msg.candidates),
            "tf_attempts": 0,
            "last_tf_error": "",
            "tf_resolution_elapsed_ns": None,
            "stage_durations_ns": {},
            "status": "PENDING",
            "reason": "",
        }

        entry = PendingObservation(
            message=None,
            received_ns=received_ns,
            deadline_ns=(
                received_ns + int(self.tf_wait_timeout_sec * 1e9)
            ),
            trace=trace,
        )

        try:
            with measured_stage(trace["stage_durations_ns"], "validation"):
                source_ns = self._validate_input(msg)
                trace["source_stamp_ns"] = source_ns

                if self.source_clock_comparable:
                    age = (
                        trace["callback_ros_ns"] - source_ns
                    ) / 1e9

                    trace["source_age_at_entry_sec"] = age

                    if not self._age_allowed(age):
                        raise ValueError("SOURCE_AGE_OUT_OF_RANGE")

        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            self.arrays_rejected += 1
            self._finish_trace(entry, "REJECTED", str(exc))
            return

        self._commit_source_order(msg, source_ns)

        # Free ready work before checking the bounded queue capacity.
        self._drain_pending()

        if len(self.pending) >= self.max_pending_observations:
            self.arrays_discarded += 1
            self._finish_trace(entry, "DISCARDED", "PENDING_BUFFER_FULL")
            return

        with measured_stage(trace["stage_durations_ns"], "input_copy"):
            entry.message = copy.deepcopy(msg)

        self.pending.append(entry)
        self._drain_pending()

    def _validate_input(self, msg):
        """
        Validate DEPTH-stage provenance and position-feature contracts.

        AVAILABLE positions must share the array's spatial header.
        UNAVAILABLE positions must have zero quality and an empty header.
        NOT_PROCESSED positions are invalid after the depth stage.
        """
        nonempty("source_session_id", msg.source_session_id)
        nonempty("source_frame", msg.header.frame_id)
        nonempty("image_frame", msg.image_header.frame_id)

        source_ns = stamp_ns(msg.header.stamp)

        if source_ns != stamp_ns(msg.image_header.stamp):
            raise ValueError("SOURCE_TIMESTAMP_MISMATCH")

        if (
            msg.processing_stage != PersonDetectionArray.DEPTH
            or msg.identity_evaluated
            or msg.tracking_session_id
        ):
            raise ValueError("EXPECTED_UNIDENTIFIED_DEPTH_STAGE")

        if msg.source_sequence < 1:
            raise ValueError("INVALID_SOURCE_SEQUENCE")

        if msg.source_detection_count != len(msg.candidates):
            raise ValueError("SOURCE_COUNT_MISMATCH")

        if msg.source_session_id in self.retired_source_sessions:
            raise ValueError("RETIRED_SOURCE_SESSION")

        if msg.source_session_id == self.active_source_session:
            if (
                msg.source_sequence <= self.last_source_sequence
                or source_ns <= self.last_source_stamp_ns
            ):
                raise ValueError("DUPLICATE_OR_REVERSED_SOURCE")

        seen = set()

        for candidate in msg.candidates:
            if candidate.source_index in seen:
                raise ValueError("DUPLICATE_SOURCE_INDEX")

            seen.add(candidate.source_index)

            if (
                candidate.person_id
                or candidate.identity_confirmed
                or candidate.identity_generation != 0
                or candidate.identity_state != "INSUFFICIENT"
            ):
                raise ValueError("UNEXPECTED_IDENTITY_BEFORE_TRACKING")

            metadata = candidate.position.metadata
            unit_interval("position quality", metadata.quality)

            if metadata.status == FeatureMetadata.AVAILABLE:
                if metadata.header != msg.header:
                    raise ValueError("POSITION_HEADER_MISMATCH")

                point = candidate.position.value

                # Shared type validates finite geometry.
                Vector3(
                    float(point.x),
                    float(point.y),
                    float(point.z),
                )

            elif metadata.status == FeatureMetadata.UNAVAILABLE:
                if (
                    metadata.quality != 0.0
                    or metadata.header.frame_id
                    or stamp_ns(
                        metadata.header.stamp,
                        allow_zero=True,
                    ) != 0
                ):
                    raise ValueError("INVALID_UNAVAILABLE_POSITION")

            else:
                raise ValueError("POSITION_NOT_PROCESSED_BY_DEPTH_STAGE")

            if msg.source_observation_complete and not (
                metadata.status == FeatureMetadata.AVAILABLE
                and metadata.quality > 0.0
            ):
                raise ValueError("INCONSISTENT_DEPTH_COMPLETENESS")

        return source_ns

    def _commit_source_order(self, msg, source_ns):
        """
        Advance admission order and retire pending work from an old producer.

        Retired observations are traced as discarded, never as empty detections.
        """
        if msg.source_session_id != self.active_source_session:
            if self.active_source_session:
                self.retired_source_sessions.add(
                    self.active_source_session
                )

                while self.pending:
                    old = self.pending.popleft()
                    self.arrays_discarded += 1
                    self._finish_trace(
                        old,
                        "DISCARDED",
                        "SOURCE_SESSION_CHANGED",
                    )

            self.active_source_session = msg.source_session_id

        self.last_source_sequence = msg.source_sequence
        self.last_source_stamp_ns = source_ns

    def _age_allowed(self, age):
        return (
            -self.future_tolerance_sec
            <= age
            <= self.maximum_source_age_sec
        )

    # =========================================================================
    # PROCESSING / ORDERED NONBLOCKING TF RESOLUTION
    # =========================================================================

    def _drain_pending(self):
        """
        Resolve pending observations in source order.

        A newer observation never overtakes an older pending observation.
        Each cycle handles a bounded number of arrays.

        Timers retry existing observations; they create no new source evidence.
        """
        for _ in range(self.max_arrays_per_cycle):
            if not self.pending:
                return

            entry = self.pending[0]
            msg = entry.message
            trace = entry.trace

            if self.source_clock_comparable:
                age = (
                    self.get_clock().now().nanoseconds
                    - trace["source_stamp_ns"]
                ) / 1e9

                if not self._age_allowed(age):
                    self.pending.popleft()
                    self.arrays_discarded += 1
                    self._finish_trace(
                        entry,
                        "DISCARDED",
                        "SOURCE_EXPIRED_WHILE_PENDING",
                    )
                    continue

            needs_tf = any(
                candidate.position.metadata.status
                == FeatureMetadata.AVAILABLE
                for candidate in msg.candidates
            )

            transform = None
            failure = ""

            if needs_tf:
                if time.perf_counter_ns() >= entry.deadline_ns:
                    failure = "TF_DEADLINE_EXPIRED"
                    self.tf_timeouts += 1

                else:
                    trace["tf_attempts"] += 1

                    with measured_stage(
                        trace["stage_durations_ns"],
                        "tf_lookup",
                    ):
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                self.target_frame,
                                msg.header.frame_id,
                                Time.from_msg(msg.header.stamp),
                                timeout=Duration(seconds=0.0),
                            )

                        except TransformException as exc:
                            trace["last_tf_error"] = str(exc)

                    if transform is None:
                        # Return control so TF subscriptions can run.
                        return

                    # Do not accept a lookup that finished after the deadline.
                    if time.perf_counter_ns() > entry.deadline_ns:
                        transform = None
                        failure = "TF_DEADLINE_EXPIRED"
                        self.tf_timeouts += 1

            trace["tf_resolution_elapsed_ns"] = (
                time.perf_counter_ns() - entry.received_ns
            )

            self.pending.popleft()

            try:
                self._process_resolved(entry, transform, failure)

            except Exception as exc:
                self._finish_trace(
                    entry,
                    "ERROR",
                    type(exc).__name__,
                )
                raise

    # =========================================================================
    # PROCESSING / SPATIAL TRANSFORMATION
    # =========================================================================

    def _process_resolved(self, entry, transform, failure):
        """
        Transform positions or expose TF unavailability, then publish.

        A valid empty source array needs no TF and remains an empty observation.
        Missing positions are never replaced by predicted or zero-distance data.
        """
        trace = entry.trace

        with measured_stage(trace["stage_durations_ns"], "transformation"):
            output = copy.deepcopy(entry.message)

            output.header.frame_id = self.target_frame
            output.processing_stage = PersonDetectionArray.SPATIAL

            translation = None
            quaternion = None

            if transform is not None:
                try:
                    if (
                        transform.header.frame_id != self.target_frame
                        or transform.child_frame_id
                        != entry.message.header.frame_id
                    ):
                        raise ValueError("TF_FRAME_MISMATCH")

                    tf_translation = transform.transform.translation
                    tf_rotation = transform.transform.rotation

                    translation = Vector3(
                        float(tf_translation.x),
                        float(tf_translation.y),
                        float(tf_translation.z),
                    )

                    quaternion = (
                        float(tf_rotation.x),
                        float(tf_rotation.y),
                        float(tf_rotation.z),
                        float(tf_rotation.w),
                    )

                    # Validate the shared transform before applying it.
                    rotation_from_quaternion(*quaternion)

                    trace["tf_returned_stamp_ns"] = stamp_ns(
                        transform.header.stamp,
                        allow_zero=True,
                    )
                    trace["tf_translation"] = [
                        translation.x,
                        translation.y,
                        translation.z,
                    ]
                    trace["tf_quaternion_xyzw"] = list(quaternion)

                except (ValueError, TypeError, OverflowError):
                    failure = "INVALID_TF_GEOMETRY"

            transformed = 0

            for candidate in output.candidates:
                metadata = candidate.position.metadata

                # Preserve unavailable upstream evidence and its failure reason.
                if metadata.status != FeatureMetadata.AVAILABLE:
                    continue

                if failure:
                    self._invalidate_position(candidate, failure)
                    continue

                if translation is None or quaternion is None:
                    self._invalidate_position(
                        candidate,
                        "TF_UNAVAILABLE",
                    )
                    continue

                point = candidate.position.value

                try:
                    result = transform_position(
                        position=Vector3(
                            float(point.x),
                            float(point.y),
                            float(point.z),
                        ),
                        translation=translation,
                        quaternion_xyzw=quaternion,
                    )

                except (ValueError, OverflowError):
                    self._invalidate_position(
                        candidate,
                        "INVALID_TRANSFORMED_POSITION",
                    )
                    continue

                candidate.position.value.x = result.x
                candidate.position.value.y = result.y
                candidate.position.value.z = result.z

                # Preserve source timestamp and upstream measurement quality.
                metadata.header = copy.deepcopy(output.header)
                transformed += 1

            usable = sum(
                candidate.position.metadata.status
                == FeatureMetadata.AVAILABLE
                and candidate.position.metadata.quality > 0.0
                for candidate in output.candidates
            )

            output.source_observation_complete = bool(
                entry.message.source_observation_complete
                and usable == len(output.candidates)
            )

            trace["transformed_positions"] = transformed
            trace["usable_positions"] = usable
            trace["coverage_complete"] = output.source_observation_complete
            trace["candidate_results"] = [
                {
                    "source_index": int(candidate.source_index),
                    "position_status": int(candidate.position.metadata.status),
                    "position_reason": candidate.position.metadata.reason,
                }
                for candidate in output.candidates
            ]

        if self.source_clock_comparable:
            age = (
                self.get_clock().now().nanoseconds
                - trace["source_stamp_ns"]
            ) / 1e9

            trace["source_age_before_publish_sec"] = age

            if not self._age_allowed(age):
                self.arrays_discarded += 1
                self._finish_trace(
                    entry,
                    "DISCARDED",
                    "SOURCE_EXPIRED_BEFORE_PUBLICATION",
                )
                return

        with measured_stage(trace["stage_durations_ns"], "publication"):
            self.output_pub.publish(output)

        self.arrays_published += 1
        self.transformed_positions += transformed

        self._finish_trace(entry, "PUBLISHED", failure)

    @staticmethod
    def _invalidate_position(candidate, reason):
        """
        Replace only the position feature with explicit unavailability.

        Metric extent and depth evidence retain their camera-frame provenance.
        """
        candidate.position = PositionFeature()
        candidate.position.metadata.status = FeatureMetadata.UNAVAILABLE
        candidate.position.metadata.quality = 0.0
        candidate.position.metadata.reason = reason

    # =========================================================================
    # OUTPUT / CORRELATED TIMING TRACES
    # =========================================================================

    def _finish_trace(self, entry, status, reason):
        """
        Publish the terminal processing record for one admitted or rejected input.

        local_residence_ns includes queueing, retries and computation.
        Stage durations describe selected active processing sections.
        These durations overlap residence time and must not be added to it.

        Timing-message serialization and publication are excluded.
        """
        trace = entry.trace
        trace["status"] = status
        trace["reason"] = reason
        trace["local_residence_ns"] = (
            time.perf_counter_ns() - entry.received_ns
        )

        self.timing_pub.publish(
            String(data=json.dumps(trace, allow_nan=False))
        )

    def log_diagnostics(self):
        """Report queue pressure, transform failures and published observations."""
        self.get_logger().info(
            "[PERSON TF] "
            f"RX={self.arrays_received} | "
            f"REJECTED={self.arrays_rejected} | "
            f"DISCARDED={self.arrays_discarded} | "
            f"PUBLISHED={self.arrays_published} | "
            f"PENDING={len(self.pending)} | "
            f"TF_TIMEOUTS={self.tf_timeouts} | "
            f"TRANSFORMED_POSITIONS={self.transformed_positions}"
        )

    def discard_pending_on_shutdown(self):
        """Close pending traces without publishing incomplete shutdown geometry."""
        while self.pending:
            entry = self.pending.popleft()
            self.arrays_discarded += 1
            self._finish_trace(
                entry,
                "DISCARDED",
                "NODE_SHUTDOWN",
            )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonFrameTransformNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            if rclpy.ok():
                node.discard_pending_on_shutdown()

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()