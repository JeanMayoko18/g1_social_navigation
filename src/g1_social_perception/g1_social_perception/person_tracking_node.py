
#!/usr/bin/env python3
"""
ROS 2 orchestration for multimodal person tracking.

INPUT
    /social_perception/detections_enriched
    PersonDetectionArray, APPEARANCE stage.

OUTPUT
    /social_perception/detections_identified
    PersonDetectionArray, IDENTITY stage.

PIPELINE
    Validate observation and source order.
    Retain an ordered, bounded pending queue.
    Prepare timestamp-correct RGB camera geometry.
    Validate RGB/depth metric provenance.
    Decode all candidates without dropping any.
    Invoke one transactional identity engine.
    Publish identity decisions.

BOUNDARIES
    The node does not calculate association scores.
    It does not invent observations, timestamps, or geometry.
    GeometryAdapter is an auxiliary class, not another ROS node.
"""

import copy
import math
import sqlite3
import time
import uuid

from pathlib import Path
from collections import deque
from dataclasses import dataclass

import rclpy

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String

from g1_social_msgs.msg import (
    DepthEvidence,
    ExtentFeature,
    FeatureMetadata,
    PersonDetection,
    PersonDetectionArray,
)

from g1_social_perception.geometry_adapter import (
    GeometryAdapter,
    GeometryAdapterConfig,
)

from g1_social_perception.identity.persistent_memory import (
    PersistentIdentityMemory,
    PersistentMemoryConfig,
)

from g1_social_perception.identity.multimodal_engine import (
    ACTIVE_MODES,
    EngineConfig,
    MultimodalIdentityEngine,
)

from g1_social_perception.mathematical_model.assignment import (
    AssignmentThresholds,
)

from g1_social_perception.mathematical_model.types import (
    AssociationWeights,
    BoundingBox2D,
    EnrichedDetection,
    Extent2D,
    Histogram,
    Measurement,
    SourceDetection,
    Vector3,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# ============================================================================
# DATA / UTILITIES
# ============================================================================


def stamp_ns(stamp):
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)

    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    result = seconds * 1_000_000_000 + nanoseconds

    if result <= 0:
        raise ValueError("ZERO_TIMESTAMP")

    return result


def positive(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"INVALID_POSITIVE_PARAMETER:{name}")


def unit(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"INVALID_UNIT_PARAMETER:{name}")


@dataclass
class PendingObservation:
    message: object
    received_at: float
    deadline_at: float
    source_ns: int
    source_session_id: str
    source_sequence: int


# ============================================================================
# PROCESSING / MESSAGE DECODING
# ============================================================================


def decode_feature(feature, value, *, reason="UNAVAILABLE"):
    metadata = feature.metadata

    if (
        metadata.status != FeatureMetadata.AVAILABLE
        or metadata.quality <= 0.0
    ):
        return Measurement.unavailable(
            metadata.reason or reason
        )

    return Measurement(
        value=value,
        quality=float(metadata.quality),
        stamp_ns=stamp_ns(metadata.header.stamp),
        frame_id=metadata.header.frame_id,
        reason=metadata.reason,
    )


def decode_position(feature):
    if (
        feature.metadata.status != FeatureMetadata.AVAILABLE
        or feature.metadata.quality <= 0.0
    ):
        return Measurement.unavailable(
            feature.metadata.reason or "POSITION_UNAVAILABLE"
        )

    return decode_feature(
        feature,
        Vector3(
            float(feature.value.x),
            float(feature.value.y),
            float(feature.value.z),
        ),
    )


def decode_extent(feature, expected_unit):
    if (
        feature.metadata.status != FeatureMetadata.AVAILABLE
        or feature.metadata.quality <= 0.0
    ):
        return Measurement.unavailable(
            feature.metadata.reason or "EXTENT_UNAVAILABLE"
        )

    if feature.unit != expected_unit:
        raise ValueError("EXTENT_UNIT_MISMATCH")

    return decode_feature(
        feature,
        Extent2D(
            float(feature.width),
            float(feature.height),
        ),
    )


def decode_histogram(feature):
    if (
        feature.metadata.status != FeatureMetadata.AVAILABLE
        or feature.metadata.quality <= 0.0
    ):
        return Measurement.unavailable(
            feature.metadata.reason or "APPEARANCE_UNAVAILABLE"
        )

    return decode_feature(
        feature,
        Histogram(
            bins=tuple(float(value) for value in feature.bins),
            schema_id=str(feature.schema_id),
        ),
    )

def decode_candidate(msg, candidate, *, metric_allowed):
    """
    Decode one real candidate.

    Invalid RGB-D provenance excludes BOTH depth-derived
    spatial measurements: odom position and metric extent.

    Original YOLO and appearance evidence remain available.
    """
    detection = candidate.source_detection
    bbox_msg = detection.bbox

    bbox = BoundingBox2D(
        center_u=float(bbox_msg.center.position.x),
        center_v=float(bbox_msg.center.position.y),
        width=float(bbox_msg.size.x),
        height=float(bbox_msg.size.y),
    )

    source = SourceDetection(
        stamp_ns=stamp_ns(msg.image_header.stamp),
        image_frame=msg.image_header.frame_id,
        candidate_index=int(candidate.source_index),
        bbox=bbox,
        confidence=float(detection.score),
        class_id=int(detection.class_id),
        source_session_id=msg.source_session_id,
    )

    if metric_allowed:
        position = decode_position(candidate.position)
        metric = decode_extent(
            candidate.metric_extent,
            ExtentFeature.METRES,
        )
    else:
        position = Measurement.unavailable(
            "INVALID_RGB_DEPTH_PROVENANCE"
        )
        metric = Measurement.unavailable(
            "INVALID_RGB_DEPTH_PROVENANCE"
        )

    upper = decode_histogram(candidate.appearance_upper)
    lower = decode_histogram(candidate.appearance_lower)

    apparent = decode_extent(
        candidate.apparent_extent,
        ExtentFeature.PIXELS,
    )

    return EnrichedDetection(
        source=source,
        position_odom=position,
        appearance_upper=upper,
        appearance_lower=lower,
        apparent_extent=apparent,
        metric_extent=metric,
    )

# ============================================================================
# ROS NODE
# ============================================================================


class PersonTrackingNode(Node):

    def __init__(self):
        super().__init__("person_tracking")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[PERSON TRACKING] "
            f"input={self.input_topic} "
            f"output={self.output_topic} "
            f"session={self.tracking_session_id}"
        )

    # ========================================================================
    # DATA / CONFIGURATION
    # ========================================================================

    def _configure(self):
        defaults = {
            "input_topic":
                "/social_perception/detections_enriched",
            "output_topic":
                "/social_perception/detections_identified",
            "timing_topic":
                "/social_perception/person_tracking/timing",
            "camera_info_topic":
                "/camera/aligned_depth_to_color/camera_info",

            "odom_frame": "odom",
            "person_class_id": 0,

            "geometry_wait_sec": 0.15,
            "retry_period_sec": 0.02,
            "max_pending_observations": 32,

            "maximum_depth_rgb_offset_sec": 0.10,
            "maximum_camera_info_age_sec": 2.0,
            "tf_cache_sec": 10.0,

            "weight_g": 0.25,
            "weight_m": 0.35,
            "weight_a": 0.30,
            "weight_h": 0.10,

            "eta_upper": 0.50,
            "eta_lower": 0.50,
            "alpha": 0.50,
            "beta": 0.50,

            "motion_tolerance_m": 1.0,
            "maximum_velocity_age_sec": 1.0,

            "metric_sigma_w": 0.45,
            "metric_sigma_h": 0.45,
            "apparent_sigma_w": 0.45,
            "apparent_sigma_h": 0.45,

            "maximum_view_angle_rad": 0.35,
            "minimum_projection_depth_m": 0.20,
            "maximum_projection_depth_m": 8.0,

            "active_timeout_sec": 2.0,
            "gallery_retention_sec": 30.0,

            "confirm_hits": 3,
            "max_active_tracks": 64,
            "max_gallery_entries": 128,

            "minimum_initial_confidence": 0.50,
            "minimum_initial_region_quality": 0.20,

            "minimum_return_score": 0.75,
            "minimum_return_support": 0.50,
            "minimum_return_margin": 0.10,

            "stationary_fallback_quality": 0.35,

            "active_minimum_score": 0.55,
            "active_minimum_support": 0.15,
            "active_minimum_margin": 0.10,

            "log_period_sec": 5.0,

            "persistent_memory_enabled": False,
            "persistent_memory_path":
                "~/.local/share/g1_social_perception/"
                "identity_history.sqlite3",
            "persistent_memory_retention_days": 7,
            "persistent_memory_maximum_records": 10000,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        if len({
            self.input_topic,
            self.output_topic,
            self.timing_topic,
        }) != 3:
            raise ValueError("TOPIC_COLLISION")

        for name in (
            "geometry_wait_sec",
            "retry_period_sec",
            "tf_cache_sec",
            "log_period_sec",
        ):
            positive(name, getattr(self, name))

        if self.retry_period_sec > self.geometry_wait_sec:
            raise ValueError("INVALID_GEOMETRY_RETRY_PERIOD")

        if (
            type(self.max_pending_observations) is not int
            or self.max_pending_observations < 1
        ):
            raise ValueError("INVALID_PENDING_CAPACITY")

        for name in (
            "weight_g", "weight_m", "weight_a", "weight_h",
            "eta_upper", "eta_lower", "alpha", "beta",
            "minimum_initial_confidence",
            "minimum_initial_region_quality",
            "minimum_return_score",
            "minimum_return_support",
            "minimum_return_margin",
            "active_minimum_score",
            "active_minimum_support",
            "active_minimum_margin",
            "stationary_fallback_quality",
        ):
            unit(name, getattr(self, name))

        if (
            self.weight_g + self.weight_m
            + self.weight_a + self.weight_h <= 0.0
        ):
            raise ValueError("ALL_ASSOCIATION_WEIGHTS_ZERO")

        if self.eta_upper + self.eta_lower <= 0.0:
            raise ValueError("APPEARANCE_WEIGHTS_ZERO")

        if self.alpha + self.beta <= 0.0:
            raise ValueError("SIZE_WEIGHTS_ZERO")

    def _engine_config(self):
        weights = AssociationWeights(
            g=float(self.weight_g),
            m=float(self.weight_m),
            a=float(self.weight_a),
            h=float(self.weight_h),
            eta_upper=float(self.eta_upper),
            eta_lower=float(self.eta_lower),
            alpha=float(self.alpha),
            beta=float(self.beta),
        )

        # One explicitly configured initial policy for each mode.
        # Per-mode thresholds should be calibrated offline.
        thresholds = {
            mode: AssignmentThresholds(
                minimum_score=float(self.active_minimum_score),
                minimum_support=float(self.active_minimum_support),
                minimum_margin=float(self.active_minimum_margin),
            )
            for mode in ACTIVE_MODES
        }

        return EngineConfig(
            weights=weights,
            active_thresholds=thresholds,
            motion_tolerance_m=float(self.motion_tolerance_m),
            maximum_velocity_age_sec=float(
                self.maximum_velocity_age_sec
            ),
            metric_sigma=(
                float(self.metric_sigma_w),
                float(self.metric_sigma_h),
            ),
            apparent_sigma=(
                float(self.apparent_sigma_w),
                float(self.apparent_sigma_h),
            ),
            maximum_view_angle_rad=float(
                self.maximum_view_angle_rad
            ),
            minimum_projection_depth_m=float(
                self.minimum_projection_depth_m
            ),
            maximum_projection_depth_m=float(
                self.maximum_projection_depth_m
            ),
            active_timeout_sec=float(self.active_timeout_sec),
            gallery_retention_sec=float(
                self.gallery_retention_sec
            ),
            confirm_hits=int(self.confirm_hits),
            max_active_tracks=int(self.max_active_tracks),
            max_gallery_entries=int(self.max_gallery_entries),
            minimum_initial_confidence=float(
                self.minimum_initial_confidence
            ),
            minimum_initial_region_quality=float(
                self.minimum_initial_region_quality
            ),
            minimum_return_score=float(
                self.minimum_return_score
            ),
            minimum_return_support=float(
                self.minimum_return_support
            ),
            minimum_return_margin=float(
                self.minimum_return_margin
            ),
            stationary_fallback_quality=float(
                self.stationary_fallback_quality
            ),
        )

    # ========================================================================
    # DATA / STATE
    # ========================================================================

    def _initialize_state(self):
        self.tracking_session_id = uuid.uuid4().hex

        self.engine = MultimodalIdentityEngine(
            session_id=self.tracking_session_id,
            config=self._engine_config(),
        )

        if type(self.persistent_memory_enabled) is not bool:
            raise ValueError("INVALID_PERSISTENCE_ENABLED")

        self.persistent_memory = PersistentIdentityMemory(
            PersistentMemoryConfig(
                database_path=str(
                    Path(self.persistent_memory_path).expanduser()
                ),
                enabled=self.persistent_memory_enabled,
                retention_days=int(
                    self.persistent_memory_retention_days
                ),
                maximum_records=int(
                    self.persistent_memory_maximum_records
                ),
            )
        )

        self._persistent_observation_counts = {}
        self.persistent_saved = 0
        self.persistent_errors = 0

        self.pending = deque()
        self.active_source_session = None
        self.retired_source_sessions = set()

        self.last_source_sequence = 0
        self.last_source_stamp_ns = 0

        self.arrays_received = 0
        self.arrays_rejected = 0
        self.arrays_published = 0
        self.arrays_discarded = 0
        self.geometry_unavailable = 0

        self.steady_clock = Clock(
            clock_type=ClockType.STEADY_TIME
        )

    # ========================================================================
    # ROS INTERFACES
    # ========================================================================

    def _create_interfaces(self):
        self.geometry_adapter = GeometryAdapter(
            node=self,
            config=GeometryAdapterConfig(
                camera_info_topic=self.camera_info_topic,
                odom_frame=self.odom_frame,
                tf_cache_sec=float(self.tf_cache_sec),
                maximum_camera_info_age_sec=float(
                    self.maximum_camera_info_age_sec
                ),
                maximum_depth_rgb_offset_sec=float(
                    self.maximum_depth_rgb_offset_sec
                ),
            ),
        )

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self, self.input_topic, input_qos
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

        self.subscription = self.create_subscription(
            PersonDetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.retry_timer = self.create_timer(
            self.retry_period_sec,
            self._drain_pending,
            clock=self.steady_clock,
        )

        self.diagnostics_timer = self.create_timer(
            self.log_period_sec,
            self.log_diagnostics,
            clock=self.steady_clock,
        )

    # ========================================================================
    # INPUT / VALIDATION
    # ========================================================================

    def _validate_input(self, msg):
        if (
            msg.processing_stage != PersonDetectionArray.APPEARANCE
            or msg.identity_evaluated
            or msg.tracking_session_id
        ):
            raise ValueError("EXPECTED_UNIDENTIFIED_APPEARANCE_STAGE")

        if (
            not msg.source_session_id
            or not msg.image_header.frame_id
            or msg.header.frame_id != self.odom_frame
        ):
            raise ValueError("INVALID_SOURCE_OR_SPATIAL_FRAME")

        source_ns = stamp_ns(msg.header.stamp)

        if source_ns != stamp_ns(msg.image_header.stamp):
            raise ValueError("RGB_SOURCE_TIMESTAMP_MISMATCH")

        if msg.source_sequence < 1:
            raise ValueError("INVALID_SOURCE_SEQUENCE")

        if msg.source_detection_count != len(msg.candidates):
            raise ValueError("SOURCE_CANDIDATE_COUNT_MISMATCH")

        if msg.source_session_id in self.retired_source_sessions:
            raise ValueError("RETIRED_SOURCE_SESSION")

        if msg.source_session_id == self.active_source_session:
            if (
                msg.source_sequence <= self.last_source_sequence
                or source_ns <= self.last_source_stamp_ns
            ):
                raise ValueError("DUPLICATE_OR_REVERSED_OBSERVATION")

        seen = set()

        for candidate in msg.candidates:
            index = int(candidate.source_index)

            if index in seen:
                raise ValueError("DUPLICATE_SOURCE_INDEX")

            seen.add(index)

            if (
                candidate.person_id
                or candidate.identity_confirmed
                or candidate.identity_generation
                or candidate.identity_state != "INSUFFICIENT"
            ):
                raise ValueError("UNEXPECTED_UPSTREAM_IDENTITY")

            if int(candidate.source_detection.class_id) != int(
                self.person_class_id
            ):
                raise ValueError("NON_PERSON_SOURCE_DETECTION")

            confidence = float(candidate.source_detection.score)
            unit("source_confidence", confidence)

        return source_ns

    # ========================================================================
    # INPUT / ORDERED QUEUE
    # ========================================================================

    def detections_callback(self, msg):
        self.arrays_received += 1

        try:
            source_ns = self._validate_input(msg)

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as exc:
            self.arrays_rejected += 1
            self.get_logger().warning(
                "[TRACKING] Rejected observation: " + str(exc)
            )
            return

        if msg.source_session_id != self.active_source_session:
            if self.active_source_session is not None:
                self.retired_source_sessions.add(
                    self.active_source_session
                )

                # A new source producer starts a new identity domain.
                # Pending observations from its predecessor cannot be
                # carried into that domain.
                while self.pending:
                    self.pending.popleft()
                    self.arrays_discarded += 1

                self.tracking_session_id = uuid.uuid4().hex
                self.engine = MultimodalIdentityEngine(
                    session_id=self.tracking_session_id,
                    config=self._engine_config(),
                )

            self.active_source_session = msg.source_session_id

        self.last_source_sequence = int(msg.source_sequence)
        self.last_source_stamp_ns = source_ns

        # Preserve source order even while TF is missing.
        self._drain_pending()

        if len(self.pending) >= self.max_pending_observations:
            self.arrays_discarded += 1
            self.get_logger().warning(
                "[TRACKING] Pending buffer full"
            )
            return

        now = time.monotonic()

        self.pending.append(
            PendingObservation(
                message=copy.deepcopy(msg),
                received_at=now,
                deadline_at=now + self.geometry_wait_sec,
                source_ns=source_ns,
                source_session_id=msg.source_session_id,
                source_sequence=int(msg.source_sequence),
            )
        )

        self._drain_pending()

    # ========================================================================
    # PROCESSING / CANDIDATE CONVERSION
    # ========================================================================

    def _decode_candidates(self, msg):
        source_ns = stamp_ns(msg.image_header.stamp)
        frame = msg.image_header.frame_id

        decoded = []

        for candidate in msg.candidates:
            provenance = (
                self.geometry_adapter.validate_metric_provenance(
                    candidate,
                    rgb_stamp_ns=source_ns,
                    camera_frame=frame,
                )
            )

            decoded.append(
                decode_candidate(
                    msg,
                    candidate,
                    metric_allowed=provenance.usable,
                )
            )

        return tuple(decoded)

    # ========================================================================
    # PROCESSING / ORDERED GEOMETRY AND IDENTITY
    # ========================================================================

    def _drain_pending(self):
        while self.pending:
            entry = self.pending[0]
            msg = entry.message

            try:
                geometry = self.geometry_adapter.prepare(msg)

            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                # Invalid observation contracts are not repaired by
                # substituting an unrelated camera transformation.
                self.pending.popleft()
                self.arrays_rejected += 1
                self.get_logger().error(
                    "[TRACKING] Geometry contract: " + str(exc)
                )
                continue

            if (
                not geometry.usable
                and time.monotonic() < entry.deadline_at
            ):
                # Keep this observation at the head of the queue.
                # CameraInfo and TF subscriptions can receive updates.
                return

            self.pending.popleft()

            if not geometry.usable:
                self.geometry_unavailable += 1

            try:
                self._process_observation(entry, geometry)

            except Exception as exc:
                self.arrays_rejected += 1
                self.get_logger().error(
                    "[TRACKING] Processing failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                )

    def _process_observation(self, entry, geometry):
        msg = entry.message

        candidates = self._decode_candidates(msg)

        receipt = time.monotonic()

        decisions = self.engine.process(
            candidates,
            stamp_ns=entry.source_ns,
            receipt=receipt,
            current_view=geometry.view,
        )

        if len(decisions) != len(msg.candidates):
            raise RuntimeError("DECISION_COUNT_MISMATCH")

        output = copy.deepcopy(msg)

        output.processing_stage = PersonDetectionArray.IDENTITY
        output.identity_evaluated = True
        output.tracking_session_id = self.tracking_session_id

        # source_observation_complete remains unchanged:
        # it is not the same as identity confirmation.

        for candidate, decision in zip(
            output.candidates,
            decisions,
        ):
            candidate.person_id = decision.person_id
            candidate.identity_state = decision.state
            candidate.identity_confirmed = decision.confirmed
            candidate.identity_generation = decision.generation

        self.output_pub.publish(output)
        self.arrays_published += 1

        # Persistence is independent from identity decisions.
        self._persist_confirmed_observations(
            output=output,
            source_ns=entry.source_ns,
        )

        trace = {
            "node": self.get_name(),
            "tracking_session_id": self.tracking_session_id,
            "source_session_id": entry.source_session_id,
            "source_sequence": entry.source_sequence,
            "source_stamp_ns": entry.source_ns,
            "geometry_status": geometry.reason,
            "candidate_count": len(candidates),
            "identity_states": [
                decision.state for decision in decisions
            ],
            "processing_receipt_monotonic": receipt,
        }

        import json
        self.timing_pub.publish(
            String(data=json.dumps(trace, allow_nan=False))
        )

    # ========================================================================
    # PERSISTENT MEMORY
    # ========================================================================

    def _persist_confirmed_observations(
        self,
        *,
        output,
        source_ns,
    ):
        """
        Persist non-biometric history for confirmed identities.

        Never store tentative or ambiguous identities.
        Never reassign an identity from historical database rows.
        A database failure must not invalidate a published result.
        """
        if not self.persistent_memory.enabled:
            return

        session = str(output.tracking_session_id).strip()

        if not session or source_ns <= 0:
            return

        for candidate in output.candidates:
            person_id = str(candidate.person_id).strip()
            generation = int(candidate.identity_generation)

            if (
                not person_id
                or generation < 1
                or not candidate.identity_confirmed
                or candidate.identity_state in {
                    "AMBIGUOUS",
                    "INSUFFICIENT",
                }
            ):
                continue

            key = (session, person_id, generation)

            count = (
                self._persistent_observation_counts.get(key, 0)
                + 1
            )

            try:
                saved = self.persistent_memory.save_confirmed(
                    tracking_session_id=session,
                    person_id=person_id,
                    generation=generation,
                    observation_count=count,
                    last_observation_ns=source_ns,
                )

                if saved:
                    self._persistent_observation_counts[key] = count
                    self.persistent_saved += 1

            except (
                OSError,
                sqlite3.Error,
                RuntimeError,
                ValueError,
                TypeError,
            ) as exc:
                self.persistent_errors += 1
                self.get_logger().error(
                    "[PERSISTENT MEMORY] Save failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc),
                    throttle_duration_sec=5.0,
                )

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    def log_diagnostics(self):
        self.get_logger().info(
            "[PERSON TRACKING] "
            f"RX={self.arrays_received} | "
            f"REJECTED={self.arrays_rejected} | "
            f"PUBLISHED={self.arrays_published} | "
            f"DISCARDED={self.arrays_discarded} | "
            f"PENDING={len(self.pending)} | "
            f"GEOMETRY_UNAVAILABLE={self.geometry_unavailable} | "
            f"ACTIVE_TRACKS={len(self.engine.active)} | "
            f"DORMANT_TRACKS={len(self.engine.dormant)} | "
            f"PERSISTENCE_ENABLED={self.persistent_memory.enabled} | "
            f"PERSISTENT_SAVED={self.persistent_saved} | "
            f"PERSISTENT_ERRORS={self.persistent_errors}"
        )

    def discard_pending_on_shutdown(self):
        while self.pending:
            self.pending.popleft()
            self.arrays_discarded += 1


# ============================================================================
# ORCHESTRATION
# ============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonTrackingNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.discard_pending_on_shutdown()

            try:
                node.persistent_memory.close()
            finally:
                node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
