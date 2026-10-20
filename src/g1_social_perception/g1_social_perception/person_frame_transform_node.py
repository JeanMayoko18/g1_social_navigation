"""
Transform fused 3D person detections between ROS 2 coordinate frames.

Responsibility:
    Subscribe to fused 3D detections expressed in the camera frame and
    transform each valid 3D bounding-box pose into a configurable target frame.

Flow:
    INPUT
        /yolo/detections_3d
            ↓
    VALIDATION
            ↓
    TF LOOKUP AT DETECTION TIMESTAMP
            ↓
    3D POSE TRANSFORMATION
            ↓
    RESULT OBJECT
            ↓
    OUTPUT
        /yolo/detections_3d_base

The original 2D bounding box, class information, confidence score and other
detection metadata are preserved.
"""

import copy
import math
import time

import rclpy
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)
from rclpy.time import Time
from rclpy.duration import Duration
from yolo_msgs.msg import DetectionArray

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


class PersonFrameTransformNode(Node):
    """
    Transform fused person detections into a robot-relative coordinate frame.
    """

    def __init__(self):
        super().__init__("person_frame_transform")

        # =====================================================================
        # PARAMETERS
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/yolo/detections_3d",
        )

        self.declare_parameter(
            "output_topic",
            "/yolo/detections_3d_base",
        )

        self.declare_parameter(
            "target_frame",
            "base_footprint",
        )

        self.declare_parameter(
            "person_class_id",
            0,
        )

        self.declare_parameter(
            "transform_timeout_sec",
            0.10,
        )

        self.declare_parameter(
            "tf_timestamp_policy",
            "detection",
        )

        self.declare_parameter(
            "diagnostics_period_sec",
            5.0,
        )

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.target_frame = (
            self.get_parameter("target_frame")
            .get_parameter_value()
            .string_value
        )

        self.person_class_id = (
            self.get_parameter("person_class_id")
            .get_parameter_value()
            .integer_value
        )

        self.transform_timeout_sec = (
            self.get_parameter("transform_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.tf_timestamp_policy = (
            self.get_parameter("tf_timestamp_policy")
            .get_parameter_value()
            .string_value
        )

        self.diagnostics_period_sec = (
            self.get_parameter("diagnostics_period_sec")
            .get_parameter_value()
            .double_value
        )

        valid_tf_timestamp_policies = {
            "detection",
            "latest",
        }

        if (
            self.tf_timestamp_policy
            not in valid_tf_timestamp_policies
        ):
            raise ValueError(
                "Invalid tf_timestamp_policy="
                f"{self.tf_timestamp_policy!r}. "
                "Expected 'detection' or 'latest'."
            )

        # =====================================================================
        # TF
        # =====================================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # =====================================================================
        # QOS
        # =====================================================================

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self,
            self.input_topic,
            input_qos,
        )

        output_qos = reliable_qos(
            depth=10,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.subscription = self.create_subscription(
            DetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            DetectionArray,
            self.output_topic,
            output_qos,
        )

        # =====================================================================
        # DIAGNOSTICS STATE
        # =====================================================================

        self.received_arrays = 0
        self.published_arrays = 0

        self.received_persons = 0
        self.transformed_persons = 0

        self.invalid_3d = 0
        self.tf_failures = 0

        self.last_diagnostics_time = time.monotonic()

        # Latest accepted upstream observation.
        # This is the original sensor timestamp, not timer publication time.
        self.last_input_stamp = None
        self.last_input_stamp_sec = None
        self.last_input_received_monotonic = None

        # Observation diagnostics.
        self.accepted_input_arrays = 0
        self.empty_input_arrays = 0
        self.rejected_input_timestamps = 0

        # =====================================================================
        # INITIALIZATION LOG
        # =====================================================================

        self.get_logger().info("")
        self.get_logger().info("=" * 82)
        self.get_logger().info(
            "  PERSON FRAME TRANSFORM - INITIALIZATION"
        )
        self.get_logger().info("=" * 82)

        self.get_logger().info(
            f"  Input topic          : {self.input_topic}"
        )
        self.get_logger().info(
            f"  Output topic         : {self.output_topic}"
        )
        self.get_logger().info(
            f"  Target frame         : {self.target_frame}"
        )
        self.get_logger().info(
            f"  Person class ID      : {self.person_class_id}"
        )

        self.get_logger().info(
            f"  TF timestamp policy  : {self.tf_timestamp_policy}"
        )
        self.get_logger().info(
            f"  TF timeout           : "
            f"{self.transform_timeout_sec:.3f} s"
        )

        self.get_logger().info("-" * 82)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def detections_callback(
        self,
        msg: DetectionArray,
    ) -> None:
        """
        Transform person detections without inventing empty observations.

        Guarantees:
            - Reject missing frames and invalid timestamps.
            - Reject duplicate or out-of-order sensor timestamps.
            - Preserve genuine empty detection arrays.
            - Reject the entire array if any person has invalid
            3D geometry or fails TF transformation.
            - Preserve the original sensor timestamp.
        """

        self.received_arrays += 1

        source_frame = str(msg.header.frame_id).strip()

        if not source_frame:
            self.tf_failures += 1
            self.get_logger().warning(
                "[PERSON TF] Rejected array: empty source frame.",
                throttle_duration_sec=2.0,
            )
            self._publish_diagnostics_if_due()
            return

        stamp = msg.header.stamp

        stamp_sec = (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

        if (
            not math.isfinite(stamp_sec)
            or stamp_sec <= 0.0
            or (
                self.last_input_stamp_sec is not None
                and stamp_sec <= self.last_input_stamp_sec
            )
        ):
            self.rejected_input_timestamps += 1
            self.get_logger().warning(
                "[PERSON TF] Rejected non-positive, duplicate "
                "or out-of-order observation timestamp.",
                throttle_duration_sec=2.0,
            )
            self._publish_diagnostics_if_due()
            return

        output_msg = copy.deepcopy(msg)
        output_msg.header.frame_id = self.target_frame

        # Preserve sensor time even when using latest TF.
        output_msg.header.stamp = copy.deepcopy(stamp)

        valid_output_detections = []
        rejected_persons = 0

        for detection in output_msg.detections:
            if detection.class_id != self.person_class_id:
                continue

            self.received_persons += 1

            position = detection.bbox3d.center.position

            if not self._is_valid_position(
                position.x,
                position.y,
                position.z,
            ):
                self.invalid_3d += 1
                rejected_persons += 1
                continue

            transformed_pose = self._transform_bbox_pose(
                detection=detection,
                source_frame=source_frame,
                stamp=stamp,
            )

            if transformed_pose is None:
                self.tf_failures += 1
                rejected_persons += 1
                continue

            detection.bbox3d.center = transformed_pose.pose
            detection.bbox3d.frame_id = self.target_frame
            valid_output_detections.append(detection)

        if rejected_persons > 0:
            self.get_logger().warning(
                "[PERSON TF] Incomplete observation rejected | "
                f"REJECTED_PERSONS={rejected_persons} | "
                f"VALID_PERSONS={len(valid_output_detections)}",
                throttle_duration_sec=2.0,
            )
            self._publish_diagnostics_if_due()
            return

        output_msg.detections = valid_output_detections

        self.publisher.publish(output_msg)

        # Record only successfully accepted observations.
        self.last_input_stamp = copy.deepcopy(stamp)
        self.last_input_stamp_sec = stamp_sec
        self.last_input_received_monotonic = time.monotonic()

        self.accepted_input_arrays += 1
        self.published_arrays += 1
        self.transformed_persons += len(valid_output_detections)

        if not valid_output_detections:
            self.empty_input_arrays += 1

        self._publish_diagnostics_if_due()


    # =========================================================================
    # PROCESSING
    # =========================================================================

    def _resolve_tf_lookup_time(
        self,
        detection_stamp,
    ) -> Time:
        """
        Resolve the TF lookup timestamp according to the configured policy.

        detection
            Use the exact DetectionArray sensor timestamp.

            A zero timestamp is rejected because TF2 interprets Time() as
            "latest", which would silently violate temporal consistency.

        latest
            Explicitly request the latest available transform.
        """

        if self.tf_timestamp_policy == "latest":
            return Time()

        sec = int(
            detection_stamp.sec
        )

        nanosec = int(
            detection_stamp.nanosec
        )

        if sec == 0 and nanosec == 0:

            raise ValueError(
                "Detection timestamp is zero while "
                "tf_timestamp_policy='detection'."
            )

        return Time.from_msg(
            detection_stamp
        )

    def _transform_bbox_pose(
        self,
        detection,
        source_frame: str,
        stamp,
    ):
        """
        Transform one bbox3d pose into target_frame.

        With tf_timestamp_policy='detection', the exact sensor timestamp is
        used so that the resulting robot-relative geometry represents the
        robot pose at the instant the person was observed.

        With tf_timestamp_policy='latest', latest TF is used explicitly.

        Returns:
            Transformed PoseStamped, or None when the required transform is
            unavailable or the input timestamp is invalid.
        """

        try:

            lookup_time = (
                self._resolve_tf_lookup_time(
                    stamp
                )
            )

        except ValueError as exc:

            self.get_logger().warn(
                "[PERSON TF] Invalid detection timestamp | "
                f"SOURCE={source_frame} | "
                f"TARGET={self.target_frame} | "
                f"ERROR={exc}",
                throttle_duration_sec=2.0,
            )

            return None

        pose_in = PoseStamped()

        pose_in.header.frame_id = (
            source_frame
        )

        pose_in.header.stamp = (
            lookup_time.to_msg()
        )

        pose_in.pose = copy.deepcopy(
            detection.bbox3d.center
        )

        try:

            pose_out = (
                self.tf_buffer.transform(
                    pose_in,
                    self.target_frame,
                    timeout=Duration(
                        seconds=(
                            self.transform_timeout_sec
                        )
                    ),
                )
            )

            return pose_out

        except TransformException as exc:

            self.get_logger().warn(
                "[PERSON TF] Transform failed | "
                f"{source_frame} -> {self.target_frame} | "
                f"POLICY={self.tf_timestamp_policy} | "
                f"STAMP={stamp.sec}."
                f"{stamp.nanosec:09d} | "
                f"{exc}",
                throttle_duration_sec=2.0,
            )

            return None
        
    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _is_valid_position(
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """
        Validate a fused 3D position.

        Output:
            True when all coordinates are finite and camera depth is positive.
        """

        return (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
            and z > 0.0
        )

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    def _publish_diagnostics_if_due(self) -> None:
        """Report person transformation and observation health."""

        now = time.monotonic()

        if (
            now - self.last_diagnostics_time
            < self.diagnostics_period_sec
        ):
            return

        self.last_diagnostics_time = now

        if self.last_input_received_monotonic is None:
            observation_age = float("inf")
        else:
            observation_age = max(
                0.0,
                now - self.last_input_received_monotonic,
            )

        self.get_logger().info(
            "[PERSON TF] "
            f"ARRAYS_RX={self.received_arrays} | "
            f"ARRAYS_TX={self.published_arrays} | "
            f"PERSONS_RX={self.received_persons} | "
            f"PERSONS_TF={self.transformed_persons} | "
            f"INVALID_3D={self.invalid_3d} | "
            f"TF_FAILURES={self.tf_failures} | "
            f"ACCEPTED={self.accepted_input_arrays} | "
            f"EMPTY_ACCEPTED={self.empty_input_arrays} | "
            f"REJECTED_STAMPS={self.rejected_input_timestamps} | "
            f"OBSERVATION_AGE={observation_age:.2f}s"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================

def main(args=None):
    """Run the person frame transform node."""

    rclpy.init(
        args=args
    )

    node = PersonFrameTransformNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
