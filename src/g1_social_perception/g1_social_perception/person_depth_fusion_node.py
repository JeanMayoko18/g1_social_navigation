#!/usr/bin/env python3

"""
Enrich source person candidates with camera-frame depth geometry.

DATA FLOW
    PersonDetectionSourceNode
      /social_perception/detections_source [PersonDetectionArray]
        |
        + aligned depth image buffer
        + compatible CameraInfo
        |
        -> validate source envelope
        -> select nearest buffered depth image
        -> decode depth into metres
        -> estimate central-ROI depth and reliability
        -> compute camera position and visible metric extent
        -> preserve every source candidate
      /social_perception/detections_camera [PersonDetectionArray]
        |
        -> spatial TF transformation node

PRESERVED
    Original YOLO detection, source index, image header, source session,
    source sequence, appearance and apparent dimensions.

PRODUCED
    Camera-frame position.
    Visible metric width/height.
    Depth-image provenance, intrinsics, sample counts, median and MAD.
    Explicit availability and quality.

FAILURE RULE
    Missing depth or calibration never becomes an empty observation.
    Candidates remain present with UNAVAILABLE spatial features.

CAMERA MODEL
    Aligned depth and YOLO boxes must share image coordinates.
    This implementation uses K with a zero-distortion pinhole model.
    Nonzero distortion, cropped or binned calibration is rejected explicitly.

TIMING
    Per-observation stages use a local monotonic performance clock.
    Source ages require explicitly comparable ROS/source clocks.
    Publication duration does not measure downstream delivery.
"""


import copy
import json
import math
import time
import uuid

from collections import deque
from contextlib import contextmanager

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from launch.conditions import IfCondition
from launch.substitutions import PythonExpression

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from g1_social_msgs.msg import (
    DepthEvidence,
    ExtentFeature,
    FeatureMetadata,
    PersonDetectionArray,
    PositionFeature,
)

from g1_social_perception.mathematical_model.types import BoundingBox2D
from g1_social_perception.mathematical_model.geometry import (
    CameraIntrinsics,
    deproject_pixel,
    metric_extent_from_bbox,
)
from g1_social_perception.mathematical_model.depth import (
    central_depth_statistics,
    depth_reliability,
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
# DATA / TIME UTILITIES
# =============================================================================


def stamp_ns(stamp, allow_zero=False):
    """Read a ROS timestamp without floating-point ordering loss."""
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
    """Measure local elapsed time for one processing stage."""
    started = time.perf_counter_ns()

    try:
        yield
    finally:
        durations[name] = time.perf_counter_ns() - started


# =============================================================================
# NODE
# =============================================================================


class PersonDepthFusionNode(Node):

    def __init__(self):
        super().__init__("person_depth_fusion")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[PERSON DEPTH] "
            f"input={self.detections_topic} | "
            f"output={self.output_topic} | "
            f"optical_frame={self.camera_optical_frame}"
        )

    # =========================================================================
    # DATA / CONFIGURATION
    # =========================================================================

    def _configure(self):
        """Configure transport, synchronization and explicit quality policy."""
        defaults = {
            "detections_topic": "/social_perception/detections_source",
            "output_topic": "/social_perception/detections_camera",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/aligned_depth_to_color/camera_info",
            "timing_topic": "/social_perception/person_depth/timing",
            "camera_optical_frame": "camera_color_optical_frame",
            "person_class_id": 0,
            "min_score": 0.50,
            "roi_scale": 0.30,
            "depth_scale": 0.001,
            "min_depth_m": 0.20,
            "max_depth_m": 8.0,
            "max_sync_error_s": 0.10,
            "depth_buffer_size": 15,
            "minimum_valid_depth_pixels": 9,
            "maximum_depth_receipt_age_sec": 1.0,
            "depth_dispersion_scale_m": 0.10,
            "depth_sync_quality_scale_sec": 0.10,
            "source_clock_comparable": False,
            "maximum_source_age_sec": 1.0,
            "future_tolerance_sec": 0.10,
            "log_period_s": 5.0,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        topic_names = (
            "detections_topic",
            "output_topic",
            "depth_topic",
            "camera_info_topic",
            "timing_topic",
        )

        for name in topic_names + ("camera_optical_frame",):
            nonempty(name, getattr(self, name))

        if len({getattr(self, name) for name in topic_names}) != len(topic_names):
            raise ValueError("Node topics must be distinct")

        integer("person_class_id", self.person_class_id)
        integer("depth_buffer_size", self.depth_buffer_size, minimum=1)
        integer(
            "minimum_valid_depth_pixels",
            self.minimum_valid_depth_pixels,
            minimum=1,
        )

        unit_interval("min_score", self.min_score)
        unit_interval("roi_scale", self.roi_scale)

        for name in (
            "roi_scale",
            "depth_scale",
            "min_depth_m",
            "max_depth_m",
            "max_sync_error_s",
            "maximum_depth_receipt_age_sec",
            "depth_dispersion_scale_m",
            "depth_sync_quality_scale_sec",
            "maximum_source_age_sec",
            "log_period_s",
        ):
            positive(name, getattr(self, name))

        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be smaller than max_depth_m")

        if (
            not math.isfinite(self.future_tolerance_sec)
            or self.future_tolerance_sec < 0.0
        ):
            raise ValueError("future_tolerance_sec must be non-negative")

        if type(self.source_clock_comparable) is not bool:
            raise ValueError("source_clock_comparable must be boolean")

    def _initialize_state(self):
        """Initialize bounded sensor caches and source ordering."""
        # Depth entry: (source timestamp, local reception time, message).
        self.depth_buffer = deque(maxlen=self.depth_buffer_size)

        # Calibration entry: (calibration timestamp, message).
        self.camera_info_buffer = deque(maxlen=self.depth_buffer_size)

        self.active_source_session = ""
        self.retired_source_sessions = set()
        self.last_source_sequence = 0
        self.last_source_stamp_ns = None

        self.trace_session_id = uuid.uuid4().hex
        self.callback_sequence = 0

        self.depth_received = 0
        self.arrays_received = 0
        self.arrays_rejected = 0
        self.arrays_published = 0
        self.valid_positions = 0
        self.unavailable_positions = 0

        self.last_depth_received_at = None

    # =========================================================================
    # INPUT / ROS INTERFACES
    # =========================================================================

    def _create_interfaces(self):
        """Connect source candidates, aligned depth and calibration."""
        source_qos = resolve_subscription_qos(
            self, self.detections_topic, fallback_depth=10
        )
        info_qos = resolve_subscription_qos(
            self, self.camera_info_topic, fallback_depth=10
        )

        log_qos_resolution(self, self.detections_topic, source_qos)
        log_qos_resolution(self, self.camera_info_topic, info_qos)
        log_qos_resolution(self, self.depth_topic, qos_profile_sensor_data)

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

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            info_qos,
        )
        self.detections_sub = self.create_subscription(
            PersonDetectionArray,
            self.detections_topic,
            self.detections_callback,
            source_qos,
        )

        self.diagnostics_timer = self.create_timer(
            self.log_period_s,
            self.log_diagnostics,
        )

    # =========================================================================
    # INPUT / SENSOR CACHES
    # =========================================================================

    def depth_callback(self, msg):
        """
        Cache a depth frame with its original local reception time.

        Duplicate timestamps do not renew cached evidence.
        Decoding is deferred until an observation actually selects the image.
        """
        self.depth_received += 1
        self.last_depth_received_at = time.monotonic()

        try:
            source_ns = stamp_ns(msg.header.stamp)

            if msg.header.frame_id != self.camera_optical_frame:
                raise ValueError("DEPTH_FRAME_MISMATCH")

        except ValueError as exc:
            self._warn(str(exc))
            return

        if any(entry[0] == source_ns for entry in self.depth_buffer):
            return

        self.depth_buffer.append(
            (source_ns, time.monotonic(), msg)
        )

    def camera_info_callback(self, msg):
        """Cache calibration messages; zero timestamps may denote static data."""
        try:
            source_ns = stamp_ns(msg.header.stamp, allow_zero=True)

            if msg.header.frame_id != self.camera_optical_frame:
                raise ValueError("CAMERA_INFO_FRAME_MISMATCH")

        except ValueError as exc:
            self._warn(str(exc))
            return

        self.camera_info_buffer.append((source_ns, msg))

    # =========================================================================
    # INPUT / OBSERVATION ORCHESTRATION
    # =========================================================================

    def detections_callback(self, msg):
        """
        Process one source observation without dropping unresolved candidates.

        A source observation is consumed at most once after validation.
        Missing sensor evidence produces unavailable features, not empty arrays.
        """
        started = time.perf_counter_ns()
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
            "callback_ros_ns": self.get_clock().now().nanoseconds,
            "source_clock_comparable": self.source_clock_comparable,
            "source_age_at_entry_sec": None,
            "source_age_before_publish_sec": None,
            "depth_stamp_ns": None,
            "depth_offset_ns": None,
            "depth_buffer_residence_sec": None,
            "candidate_count": len(msg.candidates),
            "status": "ERROR",
            "reason": "",
            "stage_durations_ns": {},
        }

        try:
            with measured_stage(trace["stage_durations_ns"], "validation"):
                try:
                    source_ns = self._validate_source(msg)
                except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                    self.arrays_rejected += 1
                    trace["status"] = "REJECTED"
                    trace["reason"] = str(exc)
                    return

                trace["source_stamp_ns"] = source_ns

                if self.source_clock_comparable:
                    age = (trace["callback_ros_ns"] - source_ns) / 1e9
                    trace["source_age_at_entry_sec"] = age

                    if not self._age_allowed(age):
                        self.arrays_rejected += 1
                        trace["status"] = "REJECTED"
                        trace["reason"] = "SOURCE_AGE_OUT_OF_RANGE"
                        return

                self._commit_source_order(msg, source_ns)

            with measured_stage(trace["stage_durations_ns"], "sensor_selection"):
                selected = (
                    self._select_depth(source_ns)
                    if msg.candidates else None
                )

                depth_msg = selected[2] if selected else None

                info = (
                    self._select_calibration(depth_msg, source_ns)
                    if depth_msg is not None else None
                )

                if selected is not None:
                    trace["depth_stamp_ns"] = selected[0]
                    trace["depth_offset_ns"] = selected[0] - source_ns
                    trace["depth_buffer_residence_sec"] = (
                        time.monotonic() - selected[1]
                    )

            depth_m = None
            intrinsics = None
            failure = ""

            with measured_stage(trace["stage_durations_ns"], "sensor_preparation"):
                if msg.candidates:
                    try:
                        if depth_msg is None:
                            raise ValueError("NO_MATCHING_DEPTH")

                        depth_m = self._decode_depth(depth_msg)

                        if info is None:
                            raise ValueError("NO_COMPATIBLE_CALIBRATION")

                        intrinsics = self._camera_intrinsics(info)

                    except (ValueError, TypeError, OverflowError) as exc:
                        failure = str(exc)

            with measured_stage(trace["stage_durations_ns"], "candidate_fusion"):
                output = copy.deepcopy(msg)
                output.header.frame_id = self.camera_optical_frame
                output.processing_stage = PersonDetectionArray.DEPTH

                for candidate in output.candidates:
                    self._fuse_candidate(
                        candidate,
                        output.header,
                        source_ns,
                        depth_msg,
                        info,
                        intrinsics,
                        depth_m,
                        failure,
                    )

                usable = sum(
                    candidate.position.metadata.status
                    == FeatureMetadata.AVAILABLE
                    and candidate.position.metadata.quality > 0.0
                    for candidate in output.candidates
                )

                output.source_observation_complete = bool(
                    msg.source_observation_complete
                    and usable == len(output.candidates)
                )

                trace["usable_positions"] = usable
                trace["coverage_complete"] = output.source_observation_complete
                trace["candidate_results"] = [
                    {
                        "source_index": int(candidate.source_index),
                        "reason": candidate.position.metadata.reason,
                        "quality": candidate.position.metadata.quality,
                    }
                    for candidate in output.candidates
                ]

            if self.source_clock_comparable:
                age = (
                    self.get_clock().now().nanoseconds - source_ns
                ) / 1e9

                trace["source_age_before_publish_sec"] = age

                if not self._age_allowed(age):
                    trace["status"] = "OUTPUT_EXPIRED"
                    trace["reason"] = "SOURCE_EXPIRED_BEFORE_PUBLICATION"
                    return

            with measured_stage(trace["stage_durations_ns"], "publication"):
                self.output_pub.publish(output)

            self.arrays_published += 1
            self.valid_positions += usable
            self.unavailable_positions += len(output.candidates) - usable
            trace["status"] = "PUBLISHED"
            trace["reason"] = failure

        except Exception as exc:
            trace["reason"] = type(exc).__name__
            raise

        finally:
            trace["callback_before_trace_ns"] = (
                time.perf_counter_ns() - started
            )
            self.timing_pub.publish(
                String(data=json.dumps(trace, allow_nan=False))
            )

    # =========================================================================
    # PROCESSING / SOURCE CONTRACT
    # =========================================================================

    def _validate_source(self, msg):
        """Validate the SOURCE-stage fields consumed by this node."""
        nonempty("source_session_id", msg.source_session_id)

        source_ns = stamp_ns(msg.header.stamp)

        if (
            source_ns != stamp_ns(msg.image_header.stamp)
            or msg.header.frame_id != msg.image_header.frame_id
            or msg.image_header.frame_id != self.camera_optical_frame
        ):
            raise ValueError("SOURCE_IMAGE_GEOMETRY_MISMATCH")

        if (
            msg.processing_stage != PersonDetectionArray.SOURCE
            or msg.identity_evaluated
            or msg.tracking_session_id
        ):
            raise ValueError("EXPECTED_UNIDENTIFIED_SOURCE_STAGE")

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

            detection = candidate.source_detection

            if detection.class_id != self.person_class_id:
                raise ValueError("NON_PERSON_CANDIDATE")

            box = self._box(detection)
            unit_interval("detector score", float(detection.score))

            if (
                candidate.person_id
                or candidate.identity_confirmed
                or candidate.identity_generation != 0
                or candidate.identity_state != "INSUFFICIENT"
            ):
                raise ValueError("UNEXPECTED_SOURCE_IDENTITY")

            if (
                candidate.position.metadata.status
                != FeatureMetadata.NOT_PROCESSED
                or candidate.metric_extent.metadata.status
                != FeatureMetadata.NOT_PROCESSED
                or candidate.depth_evidence.status
                != DepthEvidence.NOT_PROCESSED
            ):
                raise ValueError("DEPTH_FEATURES_ALREADY_PROCESSED")

            extent = candidate.apparent_extent

            if (
                extent.unit != ExtentFeature.PIXELS
                or extent.metadata.status != FeatureMetadata.AVAILABLE
                or extent.metadata.header != msg.image_header
                or extent.width != box.width
                or extent.height != box.height
            ):
                raise ValueError("INVALID_APPARENT_EXTENT")

            unit_interval("apparent extent quality", extent.metadata.quality)

        return source_ns

    def _commit_source_order(self, msg, source_ns):
        """Advance source ordering without changing the source identity."""
        if msg.source_session_id != self.active_source_session:
            if self.active_source_session:
                self.retired_source_sessions.add(self.active_source_session)

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
    # PROCESSING / SENSOR ASSOCIATION AND DECODING
    # =========================================================================

    def _select_depth(self, source_ns):
        """
        Select the closest admissible depth image already in the buffer.

        Source timestamps establish RGB/depth synchronization.
        Local reception age prevents indefinite reuse of cached data.
        """
        now = time.monotonic()

        eligible = [
            entry
            for entry in self.depth_buffer
            if (
                0.0 <= now - entry[1]
                <= self.maximum_depth_receipt_age_sec
                and abs(entry[0] - source_ns) / 1e9
                <= self.max_sync_error_s
            )
        ]

        if not eligible:
            return None

        return min(
            eligible,
            key=lambda entry: (abs(entry[0] - source_ns), entry[0]),
        )

    def _select_calibration(self, depth_msg, source_ns):
        """
        Select the latest compatible calibration no newer than the paired data.

        A zero timestamp is accepted for static calibration.
        Resolution and optical frame must match the aligned depth image.
        """
        reference_ns = max(source_ns, stamp_ns(depth_msg.header.stamp))

        eligible = [
            entry
            for entry in self.camera_info_buffer
            if (
                entry[0] <= reference_ns
                and entry[1].width == depth_msg.width
                and entry[1].height == depth_msg.height
                and entry[1].header.frame_id == depth_msg.header.frame_id
            )
        ]

        if not eligible:
            return None

        return max(eligible, key=lambda entry: entry[0])[1]

    def _decode_depth(self, msg):
        """
        Decode depth while respecting encoding, byte order and row stride.

        16UC1 / mono16:
            Integer values multiplied by configured depth_scale.

        32FC1:
            Floating-point depths already expressed in metres.
        """
        if msg.encoding in ("16UC1", "mono16"):
            scalar, multiplier = "u2", self.depth_scale
        elif msg.encoding == "32FC1":
            scalar, multiplier = "f4", 1.0
        else:
            raise ValueError("UNSUPPORTED_DEPTH_ENCODING")

        dtype = np.dtype(
            (">" if msg.is_bigendian else "<") + scalar
        )

        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)

        if (
            height <= 0
            or width <= 0
            or step < width * dtype.itemsize
            or len(msg.data) != height * step
        ):
            raise ValueError("INVALID_DEPTH_LAYOUT")

        pixels = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=msg.data,
            strides=(step, dtype.itemsize),
        )

        return pixels.astype(np.float64) * multiplier

    @staticmethod
    def _camera_intrinsics(info):
        """
        Validate the pinhole calibration supported by the shared equations.

        Nonzero distortion needs an explicit distortion-aware or rectified
        implementation; it must not be silently ignored.
        """
        if not all(math.isfinite(float(v)) for v in info.k):
            raise ValueError("INVALID_CAMERA_MATRIX")

        if any(
            not math.isfinite(float(v)) or float(v) != 0.0
            for v in info.d
        ):
            raise ValueError("UNSUPPORTED_NONZERO_DISTORTION")

        if (
            info.binning_x not in (0, 1)
            or info.binning_y not in (0, 1)
            or info.roi.x_offset != 0
            or info.roi.y_offset != 0
            or info.roi.width not in (0, info.width)
            or info.roi.height not in (0, info.height)
        ):
            raise ValueError("UNSUPPORTED_CALIBRATION_ROI_OR_BINNING")

        if (
            info.k[1] != 0.0
            or info.k[3] != 0.0
            or info.k[6] != 0.0
            or info.k[7] != 0.0
            or info.k[8] != 1.0
        ):
            raise ValueError("UNSUPPORTED_CAMERA_MATRIX")

        return CameraIntrinsics(
            fx=float(info.k[0]),
            fy=float(info.k[4]),
            cx=float(info.k[2]),
            cy=float(info.k[5]),
        )

    # =========================================================================
    # PROCESSING / CANDIDATE ENRICHMENT
    # =========================================================================

    @staticmethod
    def _box(detection):
        return BoundingBox2D(
            center_u=float(detection.bbox.center.position.x),
            center_v=float(detection.bbox.center.position.y),
            width=float(detection.bbox.size.x),
            height=float(detection.bbox.size.y),
        )

    @staticmethod
    def _unavailable(candidate, reason):
        """Clear spatial values and mark an attempted but failed computation."""
        candidate.position = PositionFeature()
        candidate.position.metadata.status = FeatureMetadata.UNAVAILABLE
        candidate.position.metadata.reason = reason

        candidate.metric_extent = ExtentFeature()
        candidate.metric_extent.unit = ExtentFeature.METRES
        candidate.metric_extent.metadata.status = FeatureMetadata.UNAVAILABLE
        candidate.metric_extent.metadata.reason = reason

        candidate.depth_evidence.status = DepthEvidence.UNAVAILABLE
        candidate.depth_evidence.reason = reason

    @staticmethod
    def _available_metadata(header, quality):
        metadata = FeatureMetadata()
        metadata.status = FeatureMetadata.AVAILABLE
        metadata.header = copy.deepcopy(header)
        metadata.quality = quality
        metadata.reason = "VALID"
        return metadata

    def _fuse_candidate(
        self,
        candidate,
        spatial_header,
        source_ns,
        depth_msg,
        info,
        intrinsics,
        depth_m,
        failure,
    ):
        """
        Fill depth provenance and spatial features for one retained candidate.

        The original YOLO detection is never modified.
        Both spatial features inherit depth reliability in this initial policy.
        Metric-size quality does not certify posture or full-body visibility.
        """
        evidence = DepthEvidence()
        candidate.depth_evidence = evidence

        if depth_msg is not None:
            evidence.image_selected = True
            evidence.image_header = copy.deepcopy(depth_msg.header)
            evidence.image_offset_ns = (
                stamp_ns(depth_msg.header.stamp) - source_ns
            )

        if intrinsics is not None:
            evidence.calibration_available = True
            evidence.camera_info_header = copy.deepcopy(info.header)
            evidence.fx = intrinsics.fx
            evidence.fy = intrinsics.fy
            evidence.cx = intrinsics.cx
            evidence.cy = intrinsics.cy

        if failure:
            self._unavailable(candidate, failure)
            return

        detection = candidate.source_detection
        confidence = float(detection.score)

        if confidence < self.min_score:
            self._unavailable(candidate, "DETECTOR_CONFIDENCE_TOO_LOW")
            return

        box = self._box(detection)

        stats = central_depth_statistics(
            depth_m=depth_m,
            box=box,
            roi_scale=self.roi_scale,
            minimum_m=self.min_depth_m,
            maximum_m=self.max_depth_m,
        )

        evidence.roi_pixel_count = stats.roi_pixels
        evidence.valid_pixel_count = stats.valid_pixels

        if stats.valid_pixels < self.minimum_valid_depth_pixels:
            self._unavailable(candidate, "INSUFFICIENT_VALID_DEPTH")
            return

        quality = depth_reliability(
            stats=stats,
            confidence=confidence,
            offset_sec=evidence.image_offset_ns / 1e9,
            dispersion_scale_m=self.depth_dispersion_scale_m,
            sync_scale_sec=self.depth_sync_quality_scale_sec,
        )

        if quality <= 0.0:
            self._unavailable(candidate, "ZERO_DEPTH_QUALITY")
            return

        try:
            position = deproject_pixel(
                u=box.center_u,
                v=box.center_v,
                depth_m=stats.median_m,
                intrinsics=intrinsics,
            )

            extent = metric_extent_from_bbox(
                bbox=box,
                depth_m=stats.median_m,
                intrinsics=intrinsics,
            )

        except (ValueError, OverflowError):
            self._unavailable(candidate, "INVALID_PROJECTED_GEOMETRY")
            return

        candidate.position = PositionFeature()
        candidate.position.metadata = self._available_metadata(
            spatial_header, quality
        )
        candidate.position.value.x = position.x
        candidate.position.value.y = position.y
        candidate.position.value.z = position.z

        candidate.metric_extent = ExtentFeature()
        candidate.metric_extent.metadata = self._available_metadata(
            spatial_header, quality
        )
        candidate.metric_extent.unit = ExtentFeature.METRES
        candidate.metric_extent.width = extent.width
        candidate.metric_extent.height = extent.height

        evidence.status = DepthEvidence.AVAILABLE
        evidence.median_depth_m = stats.median_m
        evidence.depth_mad_m = stats.mad_m
        evidence.reason = "VALID"

    # =========================================================================
    # OUTPUT / DIAGNOSTICS
    # =========================================================================

    def _warn(self, reason):
        self.get_logger().warning(
            "[PERSON DEPTH] " + reason,
            throttle_duration_sec=2.0,
        )

    def log_diagnostics(self):
        """Report transport activity separately from successful spatial fusion."""
        silence = (
            None
            if self.last_depth_received_at is None
            else time.monotonic() - self.last_depth_received_at
        )

        self.get_logger().info(
            "[PERSON DEPTH] "
            f"ARRAYS_RX={self.arrays_received} | "
            f"REJECTED={self.arrays_rejected} | "
            f"PUBLISHED={self.arrays_published} | "
            f"VALID_POSITIONS={self.valid_positions} | "
            f"UNAVAILABLE_POSITIONS={self.unavailable_positions} | "
            f"DEPTH_RX={self.depth_received} | "
            f"DEPTH_BUFFER={len(self.depth_buffer)} | "
            f"DEPTH_SILENCE_SEC={silence}"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonDepthFusionNode()
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