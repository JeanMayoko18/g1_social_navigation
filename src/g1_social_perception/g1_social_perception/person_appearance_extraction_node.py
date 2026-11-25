
#!/usr/bin/env python3

"""
SPATIAL -> APPEARANCE

DATA
    Configuration, RGB cache, pending observations, source ordering.

INPUT
    /social_perception/detections_odom
    /camera/color/image_raw

PROCESSING
    Validate source provenance and upstream feature contracts.
    Wait non-blockingly for the matching original RGB image.
    Extract upper/lower appearance descriptors.
    Preserve every upstream measurement and its quality.
    Distinguish invalid observations from missing measurements.

OUTPUT
    /social_perception/detections_enriched
    /social_perception/person_appearance/timing

No association, identity assignment or memory update occurs here.
"""

import copy
import json
import math
import time
import uuid

from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String

from g1_social_msgs.msg import (
    AppearanceFeature,
    DepthEvidence,
    ExtentFeature,
    FeatureMetadata,
    PersonDetectionArray,
)

from g1_social_perception.mathematical_model.types import (
    BoundingBox2D,
)

from g1_social_perception.mathematical_model.appearance import (
    AppearancePolicy,
    RelativeRegion,
    extract_appearance_detailed,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# DATA / STRUCTURES
# =============================================================================


@dataclass
class PendingObservation:
    message: object
    admitted_ns: int
    deadline_ns: int
    trace: dict


def stamp_ns(stamp, allow_zero=False):
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    result = sec * 1_000_000_000 + nanosec

    if result == 0 and not allow_zero:
        raise ValueError("ZERO_TIMESTAMP")

    return result


def valid_metadata(metadata, name, expected_header=None):
    """Validate an upstream feature without modifying it."""

    if metadata.status not in (
        FeatureMetadata.AVAILABLE,
        FeatureMetadata.UNAVAILABLE,
    ):
        raise ValueError(f"{name}:NOT_PROCESSED")

    q = float(metadata.quality)

    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"{name}:INVALID_QUALITY")

    if metadata.status == FeatureMetadata.AVAILABLE:
        if q <= 0.0:
            raise ValueError(f"{name}:ZERO_AVAILABLE_QUALITY")

        if (
            expected_header is not None
            and metadata.header != expected_header
        ):
            raise ValueError(f"{name}:HEADER_MISMATCH")

    else:
        if q != 0.0:
            raise ValueError(f"{name}:UNAVAILABLE_NONZERO_QUALITY")

        if (
            metadata.header.frame_id
            or stamp_ns(
                metadata.header.stamp, allow_zero=True
            ) != 0
        ):
            raise ValueError(f"{name}:INVALID_UNAVAILABLE_HEADER")


def feature_available(feature):
    return (
        feature.metadata.status == FeatureMetadata.AVAILABLE
        and feature.metadata.quality > 0.0
    )


def availability(candidate):
    return {
        name: feature_available(getattr(candidate, name))
        for name in (
            "position",
            "apparent_extent",
            "metric_extent",
            "appearance_upper",
            "appearance_lower",
        )
    }


# =============================================================================
# NODE
# =============================================================================


class PersonAppearanceExtractionNode(Node):

    def __init__(self):
        super().__init__("person_appearance_extraction")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[PERSON APPEARANCE] "
            f"{self.input_topic} -> {self.output_topic}"
        )

    # =========================================================================
    # DATA / CONFIGURATION
    # =========================================================================

    def _configure(self):
        defaults = {
            "input_topic": "/social_perception/detections_odom",
            "rgb_topic": "/camera/color/image_raw",
            "output_topic": "/social_perception/detections_enriched",
            "timing_topic": "/social_perception/person_appearance/timing",

            "rgb_buffer_size": 30,
            "max_pending_observations": 32,
            "max_arrays_per_cycle": 4,

            "rgb_wait_timeout_sec": 0.15,
            "retry_period_sec": 0.01,
            "max_rgb_receipt_age_sec": 1.0,
            "log_period_sec": 5.0,

            "upper_region": [0.20, 0.15, 0.80, 0.48],
            "lower_region": [0.20, 0.52, 0.80, 0.90],

            "hue_bins": 12,
            "saturation_bins": 4,
            "value_bins": 4,
            "achromatic_bins": 8,

            "minimum_saturation": 0.15,
            "minimum_value": 0.10,
            "minimum_pixels": 30,
            "minimum_support": 0.30,

            "upper_external_reliability": 1.0,
            "lower_external_reliability": 1.0,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        topics = (
            self.input_topic,
            self.rgb_topic,
            self.output_topic,
            self.timing_topic,
        )

        if (
            any(not isinstance(x, str) or not x.strip()
                for x in topics)
            or len(set(topics)) != len(topics)
        ):
            raise ValueError("INVALID_TOPIC_CONFIGURATION")

        for name in (
            "rgb_wait_timeout_sec",
            "retry_period_sec",
            "max_rgb_receipt_age_sec",
            "log_period_sec",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"INVALID_PARAMETER:{name}")

        if self.retry_period_sec > self.rgb_wait_timeout_sec:
            raise ValueError("RETRY_PERIOD_EXCEEDS_WAIT_TIMEOUT")

        for name in (
            "rgb_buffer_size",
            "max_pending_observations",
            "max_arrays_per_cycle",
        ):
            value = getattr(self, name)

            if type(value) is not int or value < 1:
                raise ValueError(f"INVALID_PARAMETER:{name}")

        for name in (
            "upper_external_reliability",
            "lower_external_reliability",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"INVALID_PARAMETER:{name}")

        self.policy = AppearancePolicy(
            upper=RelativeRegion(*self.upper_region),
            lower=RelativeRegion(*self.lower_region),
            hue_bins=self.hue_bins,
            saturation_bins=self.saturation_bins,
            value_bins=self.value_bins,
            achromatic_bins=self.achromatic_bins,
            minimum_saturation=self.minimum_saturation,
            minimum_value=self.minimum_value,
            minimum_pixels=self.minimum_pixels,
            minimum_support=self.minimum_support,
        )

    def _initialize_state(self):
        # (source stamp ns, receipt monotonic ns, ROS image)
        self.rgb_buffer = deque(maxlen=self.rgb_buffer_size)

        self.pending = deque()

        self.active_session = ""
        self.retired_sessions = set()
        self.last_sequence = 0
        self.last_stamp_ns = 0

        self.trace_session = uuid.uuid4().hex

        self.counters = {
            "received": 0,
            "rejected": 0,
            "discarded": 0,
            "published": 0,
            "rgb_received": 0,
            "rgb_timeouts": 0,
            "appearance_available": 0,
            "appearance_unavailable": 0,
            "processing_errors": 0,
        }

        self.steady_clock = Clock(
            clock_type=ClockType.STEADY_TIME
        )

    # =========================================================================
    # INPUT / ROS INTERFACES
    # =========================================================================

    def _create_interfaces(self):
        detection_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self, self.input_topic, detection_qos
        )

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

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self._rgb_callback,
            qos_profile_sensor_data,
        )

        self.detection_sub = self.create_subscription(
            PersonDetectionArray,
            self.input_topic,
            self._detections_callback,
            detection_qos,
        )

        self.retry_timer = self.create_timer(
            self.retry_period_sec,
            self._drain_pending,
            clock=self.steady_clock,
        )

        self.diagnostics_timer = self.create_timer(
            self.log_period_sec,
            self._diagnostics,
            clock=self.steady_clock,
        )

    # =========================================================================
    # INPUT / RGB BUFFER
    # =========================================================================

    @staticmethod
    def _validate_rgb(msg):
        stamp = stamp_ns(msg.header.stamp)

        if (
            not msg.header.frame_id
            or msg.header.frame_id != msg.header.frame_id.strip()
        ):
            raise ValueError("INVALID_RGB_FRAME")

        if msg.encoding not in ("rgb8", "bgr8"):
            raise ValueError("UNSUPPORTED_RGB_ENCODING")

        if (
            msg.width <= 0
            or msg.height <= 0
            or msg.step < msg.width * 3
            or len(msg.data) != msg.height * msg.step
        ):
            raise ValueError("INVALID_RGB_LAYOUT")

        return stamp

    def _rgb_callback(self, msg):
        self.counters["rgb_received"] += 1

        try:
            stamp = self._validate_rgb(msg)
        except (ValueError, TypeError, OverflowError):
            return

        key = (stamp, msg.header.frame_id)

        if any(
            (entry[0], entry[2].header.frame_id) == key
            for entry in self.rgb_buffer
        ):
            return

        self.rgb_buffer.append(
            (stamp, time.perf_counter_ns(), msg)
        )

        self._drain_pending()

    # =========================================================================
    # INPUT / SPATIAL OBSERVATION VALIDATION
    # =========================================================================

    @staticmethod
    def _box(candidate):
        box = candidate.source_detection.bbox

        return BoundingBox2D(
            center_u=float(box.center.position.x),
            center_v=float(box.center.position.y),
            width=float(box.size.x),
            height=float(box.size.y),
        )

    def _validate_observation(self, msg):
        source_ns = stamp_ns(msg.header.stamp)

        if stamp_ns(msg.image_header.stamp) != source_ns:
            raise ValueError("SOURCE_TIMESTAMP_MISMATCH")

        if (
            not msg.header.frame_id
            or not msg.image_header.frame_id
            or msg.header.frame_id != msg.header.frame_id.strip()
            or msg.image_header.frame_id
            != msg.image_header.frame_id.strip()
        ):
            raise ValueError("INVALID_FRAME")

        if msg.processing_stage != PersonDetectionArray.SPATIAL:
            raise ValueError("EXPECTED_SPATIAL_STAGE")

        if (
            not msg.source_session_id
            or msg.source_session_id != msg.source_session_id.strip()
            or msg.source_sequence < 1
        ):
            raise ValueError("INVALID_SOURCE_IDENTITY")

        if msg.source_session_id in self.retired_sessions:
            raise ValueError("RETIRED_SOURCE_SESSION")

        if msg.source_session_id == self.active_session:
            if (
                msg.source_sequence <= self.last_sequence
                or source_ns <= self.last_stamp_ns
            ):
                raise ValueError("DUPLICATE_OR_REVERSED_SOURCE")

        if (
            msg.identity_evaluated
            or msg.tracking_session_id
        ):
            raise ValueError("IDENTITY_EVALUATED_UPSTREAM")

        if msg.source_detection_count != len(msg.candidates):
            raise ValueError("SOURCE_COUNT_MISMATCH")

        indices = set()

        for candidate in msg.candidates:
            index = int(candidate.source_index)

            if index in indices:
                raise ValueError("DUPLICATE_SOURCE_INDEX")

            indices.add(index)

            if (
                candidate.person_id
                or candidate.identity_confirmed
                or candidate.identity_generation != 0
                or candidate.identity_state != "INSUFFICIENT"
            ):
                raise ValueError("PREMATURE_IDENTITY")

            if candidate.source_detection.class_id != 0:
                raise ValueError("NON_PERSON_DETECTION")

            confidence = float(candidate.source_detection.score)

            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("INVALID_DETECTOR_CONFIDENCE")

            box = self._box(candidate)

            valid_metadata(
                candidate.position.metadata,
                "position",
                msg.header,
            )

            valid_metadata(
                candidate.apparent_extent.metadata,
                "apparent_extent",
                msg.image_header,
            )

            valid_metadata(
                candidate.metric_extent.metadata,
                "metric_extent",
            )

            if (
                candidate.position.metadata.status
                == FeatureMetadata.AVAILABLE
            ):
                p = candidate.position.value

                if not all(
                    math.isfinite(float(v))
                    for v in (p.x, p.y, p.z)
                ):
                    raise ValueError("INVALID_ODOM_POSITION")

            if (
                candidate.apparent_extent.unit
                != ExtentFeature.PIXELS
                or candidate.metric_extent.unit
                != ExtentFeature.METRES
            ):
                raise ValueError("INVALID_EXTENT_UNITS")

            if (
                candidate.apparent_extent.metadata.status
                != FeatureMetadata.AVAILABLE
            ):
                raise ValueError("APPARENT_EXTENT_MISSING")

            if (
                candidate.apparent_extent.width != box.width
                or candidate.apparent_extent.height != box.height
            ):
                raise ValueError("APPARENT_EXTENT_MISMATCH")

            if (
                candidate.metric_extent.metadata.status
                == FeatureMetadata.AVAILABLE
            ):
                e = candidate.metric_extent

                if (
                    not math.isfinite(e.width)
                    or not math.isfinite(e.height)
                    or e.width <= 0
                    or e.height <= 0
                ):
                    raise ValueError("INVALID_METRIC_EXTENT")

                if (
                    e.metadata.header.stamp != msg.header.stamp
                    or not e.metadata.header.frame_id
                ):
                    raise ValueError("INVALID_METRIC_PROVENANCE")

            if candidate.depth_evidence.status not in (
                DepthEvidence.AVAILABLE,
                DepthEvidence.UNAVAILABLE,
            ):
                raise ValueError("DEPTH_NOT_PROCESSED")

            if candidate.depth_evidence.status == DepthEvidence.AVAILABLE:
                if (
                    candidate.metric_extent.metadata.status
                    != FeatureMetadata.AVAILABLE
                ):
                    raise ValueError("DEPTH_METRIC_CONTRADICTION")

            for feature in (
                candidate.appearance_upper,
                candidate.appearance_lower,
            ):
                if (
                    feature.metadata.status
                    != FeatureMetadata.NOT_PROCESSED
                ):
                    raise ValueError("APPEARANCE_ALREADY_PROCESSED")

                if feature.bins:
                    raise ValueError("PREMATURE_APPEARANCE_BINS")

        if msg.source_observation_complete:
            for candidate in msg.candidates:
                if not feature_available(candidate.position):
                    raise ValueError(
                        "SPATIAL_COMPLETENESS_CONTRADICTION"
                    )

        return source_ns

    # =========================================================================
    # INPUT / ADMISSION
    # =========================================================================

    def _detections_callback(self, msg):
        self.counters["received"] += 1

        now = time.perf_counter_ns()

        trace = {
            "version": 1,
            "node": self.get_name(),
            "trace_session_id": self.trace_session,
            "source_session_id": msg.source_session_id,
            "source_sequence": int(msg.source_sequence),
            "observation_valid": False,
            "enrichment_processed": False,
            "all_features_available": False,
            "status": "ERROR",
            "reason": "",
        }

        try:
            source_ns = self._validate_observation(msg)

        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            self.counters["rejected"] += 1
            trace["status"] = "REJECTED"
            trace["reason"] = str(exc)
            self._publish_trace(trace, now)
            return

        trace["observation_valid"] = True

        if self.active_session != msg.source_session_id:
            if self.active_session:
                self.retired_sessions.add(self.active_session)

                while self.pending:
                    old = self.pending.popleft()
                    self.counters["discarded"] += 1
                    self._finish(
                        old, "DISCARDED", "SOURCE_SESSION_CHANGED"
                    )

            self.active_session = msg.source_session_id

        self.last_sequence = int(msg.source_sequence)
        self.last_stamp_ns = source_ns

        # Complete any already ready work before testing queue capacity.
        self._drain_pending()

        if len(self.pending) >= self.max_pending_observations:
            self.counters["discarded"] += 1

            trace["status"] = "DISCARDED"
            trace["reason"] = "PENDING_QUEUE_FULL"
            self._publish_trace(trace, now)
            return

        entry = PendingObservation(
            message=copy.deepcopy(msg),
            admitted_ns=now,
            deadline_ns=now + int(
                self.rgb_wait_timeout_sec * 1_000_000_000
            ),
            trace=trace,
        )

        self.pending.append(entry)
        self._drain_pending()

    # =========================================================================
    # PROCESSING / RGB SELECTION
    # =========================================================================

    def _select_rgb(self, msg):
        """
        Exact source timestamp + frame match.

        No nearest-frame substitution.
        """

        expected_stamp = stamp_ns(msg.image_header.stamp)
        expected_frame = msg.image_header.frame_id

        now = time.perf_counter_ns()

        for stamp, received, image in reversed(self.rgb_buffer):
            if (
                stamp == expected_stamp
                and image.header.frame_id == expected_frame
                and 0 <= now - received
                <= int(self.max_rgb_receipt_age_sec * 1e9)
            ):
                return image

        return None

    @staticmethod
    def _decode_rgb(msg):
        if (
            msg.encoding not in ("rgb8", "bgr8")
            or msg.height <= 0
            or msg.width <= 0
            or msg.step < msg.width * 3
            or len(msg.data) != msg.height * msg.step
        ):
            raise ValueError("INVALID_RGB_LAYOUT")

        rows = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        ).reshape((msg.height, msg.step))

        pixels = rows[
            :, :msg.width * 3
        ].reshape((msg.height, msg.width, 3))

        if msg.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]

        return np.ascontiguousarray(pixels)

    # =========================================================================
    # PROCESSING / APPEARANCE SERIALIZATION
    # =========================================================================

    @staticmethod
    def _unavailable(feature, reason):
        feature.metadata = FeatureMetadata()
        feature.metadata.status = FeatureMetadata.UNAVAILABLE
        feature.metadata.quality = 0.0
        feature.metadata.reason = reason
        feature.bins = []

    @staticmethod
    def _assign(feature, result, image_header):
        feature.schema_id = result.schema_id
        feature.relative_region = list(result.relative_region)

        feature.intended_pixel_count = (
            result.intended_pixel_count
        )
        feature.selected_pixel_count = (
            result.selected_pixel_count
        )
        feature.support_fraction = result.support_fraction

        feature.foreground_mask_used = (
            result.foreground_mask_used
        )

        measurement = result.measurement

        if not measurement.usable:
            PersonAppearanceExtractionNode._unavailable(
                feature,
                measurement.reason or "REGION_UNAVAILABLE",
            )
            return

        bins = list(measurement.value.bins)

        if (
            not bins
            or not all(math.isfinite(v) and v >= 0 for v in bins)
            or not math.isclose(
                math.fsum(bins),
                1.0,
                rel_tol=0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("INVALID_NORMALIZED_HISTOGRAM")

        feature.bins = bins

        feature.metadata.status = FeatureMetadata.AVAILABLE
        feature.metadata.header = copy.deepcopy(image_header)
        feature.metadata.quality = measurement.quality
        feature.metadata.reason = measurement.reason

    # =========================================================================
    # PROCESSING / ORDERED PENDING QUEUE
    # =========================================================================

    def _drain_pending(self):
        for _ in range(self.max_arrays_per_cycle):

            if not self.pending:
                return

            entry = self.pending[0]
            msg = entry.message

            # A true empty source observation requires no RGB.
            if msg.candidates:
                rgb_msg = self._select_rgb(msg)

                if rgb_msg is None:
                    if time.perf_counter_ns() < entry.deadline_ns:
                        return

                    failure = "RGB_WAIT_TIMEOUT"
                    self.counters["rgb_timeouts"] += 1

                else:
                    failure = ""

            else:
                rgb_msg = None
                failure = ""

            self.pending.popleft()

            try:
                self._process(entry, rgb_msg, failure)

            except Exception as exc:
                self.counters["processing_errors"] += 1

                self.get_logger().error(
                    "[PERSON APPEARANCE] "
                    f"PROCESSING_ERROR: {repr(exc)}"
                )

                self._finish(
                    entry,
                    "ERROR",
                    type(exc).__name__,
                )

    # =========================================================================
    # PROCESSING / FINAL ENRICHMENT
    # =========================================================================

    def _process(self, entry, rgb_msg, failure):
        original = entry.message
        output = copy.deepcopy(original)

        output.processing_stage = PersonDetectionArray.APPEARANCE

        # No change to source_observation_complete.
        # Appearance is an optional association cue.

        rgb = None

        if rgb_msg is not None:
            try:
                rgb = self._decode_rgb(rgb_msg)

            except (ValueError, TypeError, OverflowError):
                failure = "INVALID_RGB_LAYOUT"

        for candidate in output.candidates:

            candidate.appearance_upper = AppearanceFeature()
            candidate.appearance_lower = AppearanceFeature()

            if rgb is None:
                self._unavailable(
                    candidate.appearance_upper, failure
                )
                self._unavailable(
                    candidate.appearance_lower, failure
                )
                continue

            upper, lower = extract_appearance_detailed(
                image_rgb=rgb,
                bbox=self._box(candidate),
                stamp_ns=stamp_ns(output.image_header.stamp),
                image_frame=output.image_header.frame_id,
                policy=self.policy,
                upper_reliability=self.upper_external_reliability,
                lower_reliability=self.lower_external_reliability,
            )

            self._assign(
                candidate.appearance_upper,
                upper,
                output.image_header,
            )

            self._assign(
                candidate.appearance_lower,
                lower,
                output.image_header,
            )

        # Verify all results reached a terminal feature state.
        for candidate in output.candidates:
            for feature in (
                candidate.appearance_upper,
                candidate.appearance_lower,
            ):
                if feature.metadata.status not in (
                    FeatureMetadata.AVAILABLE,
                    FeatureMetadata.UNAVAILABLE,
                ):
                    raise ValueError("INCOMPLETE_APPEARANCE_PROCESSING")

        # Verify preservation of the upstream evidence.
        for before, after in zip(
            original.candidates,
            output.candidates,
        ):
            if (
                before.source_detection != after.source_detection
                or before.position != after.position
                or before.apparent_extent != after.apparent_extent
                or before.metric_extent != after.metric_extent
                or before.depth_evidence != after.depth_evidence
            ):
                raise ValueError("UPSTREAM_EVIDENCE_MODIFIED")

        if (
            output.header != original.header
            or output.image_header != original.image_header
            or output.source_session_id != original.source_session_id
            or output.source_sequence != original.source_sequence
            or output.source_observation_complete
            != original.source_observation_complete
        ):
            raise ValueError("SOURCE_CONTRACT_MODIFIED")

        results = [
            availability(candidate)
            for candidate in output.candidates
        ]

        entry.trace["feature_availability"] = [
            {
                "source_index": int(candidate.source_index),
                **state,
            }
            for candidate, state in zip(
                output.candidates, results
            )
        ]

        entry.trace["enrichment_processed"] = True

        entry.trace["all_features_available"] = all(
            all(state.values()) for state in results
        )

        entry.trace["appearance_available_count"] = sum(
            feature_available(feature)
            for candidate in output.candidates
            for feature in (
                candidate.appearance_upper,
                candidate.appearance_lower,
            )
        )

        total = 2 * len(output.candidates)
        available = entry.trace["appearance_available_count"]

        self.output_pub.publish(output)

        self.counters["published"] += 1
        self.counters["appearance_available"] += available
        self.counters["appearance_unavailable"] += (
            total - available
        )

        self._finish(entry, "PUBLISHED", failure)

    # =========================================================================
    # OUTPUT / TIMING AND DIAGNOSTICS
    # =========================================================================

    def _publish_trace(self, trace, started_ns):
        trace["local_residence_ns"] = (
            time.perf_counter_ns() - started_ns
        )

        self.timing_pub.publish(
            String(
                data=json.dumps(trace, allow_nan=False)
            )
        )

    def _finish(self, entry, status, reason):
        entry.trace["status"] = status
        entry.trace["reason"] = reason

        self._publish_trace(
            entry.trace, entry.admitted_ns
        )

    def _diagnostics(self):
        self.get_logger().info(
            "[PERSON APPEARANCE] "
            + " | ".join(
                f"{key.upper()}={value}"
                for key, value in self.counters.items()
            )
            + f" | PENDING={len(self.pending)}"
            + f" | RGB_BUFFER={len(self.rgb_buffer)}"
        )

    def discard_pending_on_shutdown(self):
        while self.pending:
            entry = self.pending.popleft()
            self.counters["discarded"] += 1
            self._finish(
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
        node = PersonAppearanceExtractionNode()
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
