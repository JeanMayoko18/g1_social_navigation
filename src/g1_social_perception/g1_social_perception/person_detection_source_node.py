"""
Create source candidates from actual YOLO observations.

DATA FLOW
    YOLO
      /yolo/detections [yolo_msgs/DetectionArray]
        -> validate source header and person detections
        -> preserve original detections and source indices
        -> initialize apparent dimensions
        -> mark depth, position and appearance as NOT_PROCESSED
        -> allocate source observation identity
      /social_perception/detections_source [PersonDetectionArray]
        -> person_depth_fusion

OBSERVATION IDENTITY
    (source_session_id, source_sequence)

    The session changes when this adapter restarts.
    The sequence increases for each observation prepared for publication.
    Downstream nodes preserve both fields.

OUTPUT RULES
    All selected person candidates remain present.
    A valid empty person observation is published.
    Malformed input is rejected without publishing an empty observation.
    No persistent person identity is assigned here.

TIMING
    Local stage durations use perf_counter_ns.
    Source age is meaningful only with comparable source and ROS clocks.
    Timing records are published separately from perception messages.
"""


import copy
import json
import math
import time
import uuid

from contextlib import contextmanager

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray

# Import the message class generated from PersonDetectionArray.msg.
from g1_social_msgs.msg import (
    AppearanceFeature,
    DepthEvidence,
    ExtentFeature,
    FeatureMetadata,
    PersonDetection,
    PersonDetectionArray,
    PositionFeature,
)

from g1_social_perception.mathematical_model.types import BoundingBox2D
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
# DATA / LOCAL TIMING
# =============================================================================


@contextmanager
def measured_stage(durations, name):
    """Measure one stage without modifying source observation timestamps."""
    started_ns = time.perf_counter_ns()

    try:
        yield
    finally:
        durations[name] = time.perf_counter_ns() - started_ns


# =============================================================================
# NODE
# =============================================================================


class PersonDetectionSourceNode(Node):

    def __init__(self):
        super().__init__("person_detection_source")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[PERSON SOURCE] "
            f"input={self.input_topic} | "
            f"output={self.output_topic} | "
            f"session={self.source_session_id} | "
            f"source_clock_comparable={self.source_clock_comparable}"
        )

    # =========================================================================
    # DATA / CONFIGURATION
    # =========================================================================

    def _configure(self):
        """Declare the adapter's transport and observation-validity settings."""
        defaults = {
            "input_topic": "/yolo/detections",
            "output_topic": "/social_perception/detections_source",
            "timing_topic": "/social_perception/person_source/timing",
            "person_class_id": 0,
            "source_clock_comparable": False,
            "maximum_source_age_sec": 1.0,
            "future_tolerance_sec": 0.10,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        for name in ("input_topic", "output_topic", "timing_topic"):
            nonempty(name, getattr(self, name))

        if len({
            self.input_topic,
            self.output_topic,
            self.timing_topic,
        }) != 3:
            raise ValueError("Input, output and timing topics must differ")

        integer("person_class_id", self.person_class_id)
        positive("maximum_source_age_sec", self.maximum_source_age_sec)

        if (
            not math.isfinite(self.future_tolerance_sec)
            or self.future_tolerance_sec < 0.0
        ):
            raise ValueError(
                "future_tolerance_sec must be finite and non-negative"
            )

        if type(self.source_clock_comparable) is not bool:
            raise ValueError("source_clock_comparable must be boolean")

    def _initialize_state(self):
        """Initialize producer identity, source ordering and counters."""
        self.source_session_id = uuid.uuid4().hex
        self.source_sequence = 0
        self.callback_sequence = 0

        self.last_published_stamp_ns = None

        self.received_arrays = 0
        self.rejected_arrays = 0
        self.published_arrays = 0
        self.published_candidates = 0
        self.published_empty_arrays = 0
        self.failed_callbacks = 0

    # =========================================================================
    # INPUT / ROS INTERFACES
    # =========================================================================

    def _create_interfaces(self):
        """Connect raw YOLO observations to the internal perception pipeline."""
        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(self, self.input_topic, input_qos)

        self.publisher = self.create_publisher(
            PersonDetectionArray,
            self.output_topic,
            reliable_qos(depth=10),
        )

        self.timing_publisher = self.create_publisher(
            String,
            self.timing_topic,
            reliable_qos(depth=100),
        )

        self.subscription = self.create_subscription(
            DetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

    # =========================================================================
    # INPUT / OBSERVATION ORCHESTRATION
    # =========================================================================

    def detections_callback(self, msg):
        """
        Validate, construct and publish one actual source observation.

        No confidence threshold removes selected person candidates here.
        Their detector confidence remains available to downstream policies.
        """
        started_ns = time.perf_counter_ns()
        callback_ros_ns = self.get_clock().now().nanoseconds

        self.received_arrays += 1
        self.callback_sequence += 1

        trace = {
            "version": 1,
            "node": self.get_name(),
            "source_session_id": self.source_session_id,
            "source_sequence": None,
            "callback_sequence": self.callback_sequence,
            "source_stamp_ns": None,
            "source_frame": msg.header.frame_id,
            "callback_ros_ns": callback_ros_ns,
            "source_clock_comparable": self.source_clock_comparable,
            "source_age_at_entry_sec": None,
            "source_age_before_publish_sec": None,
            "detections_received": len(msg.detections),
            "person_candidates": 0,
            "status": "ERROR",
            "reason": "",
            "stage_durations_ns": {},
        }

        try:
            with measured_stage(trace["stage_durations_ns"], "validation"):
                validated, reason = self._validate_input(
                    msg,
                    callback_ros_ns,
                    trace,
                )

            if validated is None:
                self.rejected_arrays += 1
                trace["status"] = "REJECTED"
                trace["reason"] = reason
                return

            stamp_ns, persons = validated
            trace["person_candidates"] = len(persons)

            with measured_stage(trace["stage_durations_ns"], "construction"):
                output = self._build_source_array(msg, persons)
                trace["source_sequence"] = output.source_sequence

            # Do not emit an observation that expired during construction.
            if self.source_clock_comparable:
                age = (
                    self.get_clock().now().nanoseconds - stamp_ns
                ) / 1e9

                trace["source_age_before_publish_sec"] = age

                if not self._source_age_allowed(age):
                    self.rejected_arrays += 1
                    trace["status"] = "REJECTED"
                    trace["reason"] = "SOURCE_EXPIRED_BEFORE_PUBLICATION"
                    return

            with measured_stage(trace["stage_durations_ns"], "publication"):
                self._publish_source_array(output)

            self.last_published_stamp_ns = stamp_ns
            trace["status"] = "PUBLISHED"

        except Exception as exc:
            self.failed_callbacks += 1
            trace["status"] = "ERROR"
            trace["reason"] = type(exc).__name__
            raise

        finally:
            trace["callback_before_trace_ns"] = (
                time.perf_counter_ns() - started_ns
            )

            self._publish_timing(trace)

    def _validate_input(self, msg, now_ros_ns, trace):
        """
        Validate source provenance and all selected person detections.

        Returns:
            (source_stamp_ns, selected_persons), or None with a reason.

        Each selected entry retains its index in the original YOLO array.
        Invalid selected detections reject the entire observation.
        """
        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)

        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            return None, "INVALID_TIMESTAMP"

        stamp_ns = sec * 1_000_000_000 + nanosec
        trace["source_stamp_ns"] = stamp_ns

        if stamp_ns <= 0:
            return None, "INVALID_TIMESTAMP"

        if (
            self.last_published_stamp_ns is not None
            and stamp_ns <= self.last_published_stamp_ns
        ):
            return None, "DUPLICATE_OR_REVERSED_TIMESTAMP"

        frame_id = msg.header.frame_id

        if not frame_id or frame_id != frame_id.strip():
            return None, "INVALID_IMAGE_FRAME"

        if self.source_clock_comparable:
            age = (now_ros_ns - stamp_ns) / 1e9
            trace["source_age_at_entry_sec"] = age

            if not self._source_age_allowed(age):
                return None, "SOURCE_AGE_OUT_OF_RANGE"

        persons = []

        try:
            for source_index, detection in enumerate(msg.detections):
                if int(detection.class_id) != self.person_class_id:
                    continue

                box = BoundingBox2D(
                    center_u=float(detection.bbox.center.position.x),
                    center_v=float(detection.bbox.center.position.y),
                    width=float(detection.bbox.size.x),
                    height=float(detection.bbox.size.y),
                )

                confidence = float(detection.score)
                unit_interval("detector confidence", confidence)

                persons.append(
                    (source_index, detection, box, confidence)
                )

        except (ValueError, TypeError, AttributeError, OverflowError):
            return None, "INVALID_PERSON_DETECTION"

        return (stamp_ns, persons), ""

    def _source_age_allowed(self, age):
        """Check source age only after clock comparability is established."""
        return (
            -self.future_tolerance_sec
            <= age
            <= self.maximum_source_age_sec
        )

    # =========================================================================
    # PROCESSING / FEATURE INITIALIZATION
    # =========================================================================

    @staticmethod
    def _pending_metadata():
        """
        Represent a feature whose responsible stage has not run yet.

        The default header remains zero/empty.
        No measurement timestamp or geometry is invented.
        """
        metadata = FeatureMetadata()
        metadata.status = FeatureMetadata.NOT_PROCESSED
        metadata.quality = 0.0
        metadata.reason = "NOT_PROCESSED"

        return metadata

    def _build_candidate(
        self,
        source_header,
        source_index,
        detection,
        box,
        confidence,
    ):
        """
        Preserve one YOLO candidate and initialize its enrichment fields.

        Available now:
            Original YOLO data and apparent width/height.

        Not processed yet:
            Depth, spatial position, metric extent and appearance.

        Identity:
            Unassigned until the identity authority evaluates this candidate.
        """
        candidate = PersonDetection()

        candidate.source_index = source_index
        candidate.source_detection = copy.deepcopy(detection)

        candidate.position = PositionFeature()
        candidate.position.metadata = self._pending_metadata()

        candidate.appearance_upper = AppearanceFeature()
        candidate.appearance_upper.metadata = self._pending_metadata()

        candidate.appearance_lower = AppearanceFeature()
        candidate.appearance_lower.metadata = self._pending_metadata()

        # Region definitions and histogram schemas are filled by extraction.
        # Their default values must not be read while NOT_PROCESSED.

        candidate.apparent_extent = ExtentFeature()
        candidate.apparent_extent.unit = ExtentFeature.PIXELS
        candidate.apparent_extent.width = box.width
        candidate.apparent_extent.height = box.height

        metadata = FeatureMetadata()
        metadata.status = FeatureMetadata.AVAILABLE
        metadata.header = copy.deepcopy(source_header)
        metadata.quality = confidence
        metadata.reason = "OBSERVED_BBOX"

        candidate.apparent_extent.metadata = metadata

        candidate.metric_extent = ExtentFeature()
        candidate.metric_extent.unit = ExtentFeature.METRES
        candidate.metric_extent.metadata = self._pending_metadata()

        candidate.depth_evidence = DepthEvidence()
        candidate.depth_evidence.status = DepthEvidence.NOT_PROCESSED
        candidate.depth_evidence.reason = "NOT_PROCESSED"

        candidate.person_id = ""
        candidate.identity_state = PersonDetection.INSUFFICIENT
        candidate.identity_confirmed = False
        candidate.identity_generation = 0

        return candidate

    def _build_source_array(self, source, persons):
        """
        Allocate the source observation key and construct its full envelope.

        All selected candidates remain represented.
        A zero-candidate array is a valid complete source observation.

        Completeness concerns the SOURCE stage only:
        downstream features are not required to exist yet.
        """
        if self.source_sequence >= (1 << 64) - 1:
            raise OverflowError("Source sequence exhausted")

        self.source_sequence += 1

        # Create one message instance.
        output = PersonDetectionArray()

        output.header = copy.deepcopy(source.header)
        output.image_header = copy.deepcopy(source.header)

        # Populate the instance fields.
        output.source_session_id = self.source_session_id
        output.source_sequence = self.source_sequence

        # Read a constant defined by the message type.
        output.processing_stage = PersonDetectionArray.SOURCE

        output.processing_stage = PersonDetectionArray.SOURCE
        output.tracking_session_id = ""
        output.identity_evaluated = False

        output.source_detection_count = len(persons)
        output.source_observation_complete = True

        output.candidates = [
            self._build_candidate(
                source_header=source.header,
                source_index=source_index,
                detection=detection,
                box=box,
                confidence=confidence,
            )
            for source_index, detection, box, confidence in persons
        ]

        # Return output for publication the populated message instance.
        return output

    # =========================================================================
    # OUTPUT / SOURCE OBSERVATIONS
    # =========================================================================

    def _publish_source_array(self, output):
        """
        Publish the envelope without altering its source timestamp.

        Publisher return does not establish downstream delivery.
        """
        self.publisher.publish(output)

        self.published_arrays += 1
        self.published_candidates += len(output.candidates)

        if not output.candidates:
            self.published_empty_arrays += 1

    # =========================================================================
    # OUTPUT / TIMING AND DIAGNOSTICS
    # =========================================================================

    def _publish_timing(self, trace):
        """
        Publish a callback trace for experiment analysis.

        Source sequence may have gaps after failed or expired publications.
        Rejected input before envelope construction has no source sequence.

        Local duration excludes serialization/publication of this trace.
        """
        self.timing_publisher.publish(
            String(data=json.dumps(trace, allow_nan=False))
        )

    def log_diagnostics(self):
        """Report source throughput and rejected observations."""
        self.get_logger().info(
            "[PERSON SOURCE] "
            f"RX={self.received_arrays} | "
            f"PUBLISHED={self.published_arrays} | "
            f"CANDIDATES={self.published_candidates} | "
            f"EMPTY={self.published_empty_arrays} | "
            f"REJECTED={self.rejected_arrays} | "
            f"ERRORS={self.failed_callbacks} | "
            f"SOURCE_SEQUENCE={self.source_sequence}"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the source adapter and release ROS resources."""
    rclpy.init(args=args)
    node = None

    try:
        node = PersonDetectionSourceNode()
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