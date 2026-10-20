#!/usr/bin/env python3

# Imports
import copy
from collections import deque
from typing import Deque, Optional, Tuple
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from yolo_msgs.msg import DetectionArray
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos
from g1_qos_manager.diagnostics import log_qos_resolution
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)



# =============================================================================
# PERSON DEPTH FUSION NODE
#
# Responsibility:
#   Fuse YOLO 2D person detections with RealSense aligned depth images in order
#   to estimate each detected person's 3D position in the camera optical frame.
#
# Inputs:
#   /yolo/detections
#       yolo_msgs/msg/DetectionArray
#
#   /camera/aligned_depth_to_color/image_raw
#       sensor_msgs/msg/Image
#
#   /camera/aligned_depth_to_color/camera_info
#       sensor_msgs/msg/CameraInfo
#
# Processing:
#   DetectionArray
#       -> select class "person"
#       -> find temporally closest depth image
#       -> extract central ROI inside person bounding box
#       -> reject invalid depth samples
#       -> compute median depth
#       -> deproject bbox center using CameraInfo
#       -> populate Detection.bbox3d
#
# Output:
#   /yolo/detections_3d
#       yolo_msgs/msg/DetectionArray
#
# Coordinate convention:
#   RealSense optical frame:
#       +X = right
#       +Y = down
#       +Z = forward
# =============================================================================


class PersonDepthFusionNode(Node):

    # =========================================================================
    # NODE CONSTRUCTOR
    # =========================================================================

    def __init__(self) -> None:
        super().__init__("person_depth_fusion")

        # ---------------------------------------------------------------------
        # CONSTRUCTOR STEP 1 - LOAD CONFIGURATION
        # ---------------------------------------------------------------------

        self.detections_topic = self.declare_parameter(
            "detections_topic",
            "/yolo/detections",
        ).value

        self.depth_topic = self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        ).value

        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic",
            "/camera/aligned_depth_to_color/camera_info",
        ).value

        self.output_topic = self.declare_parameter(
            "output_topic",
            "/yolo/detections_3d",
        ).value

        self.person_class_id = int(
            self.declare_parameter(
                "person_class_id",
                0,
            ).value
        )

        self.min_score = float(
            self.declare_parameter(
                "min_score",
                0.50,
            ).value
        )

        self.roi_scale = float(
            self.declare_parameter(
                "roi_scale",
                0.30,
            ).value
        )

        self.depth_scale = float(
            self.declare_parameter(
                "depth_scale",
                0.001,
            ).value
        )

        self.min_depth_m = float(
            self.declare_parameter(
                "min_depth_m",
                0.20,
            ).value
        )

        self.max_depth_m = float(
            self.declare_parameter(
                "max_depth_m",
                8.0,
            ).value
        )

        self.max_sync_error_s = float(
            self.declare_parameter(
                "max_sync_error_s",
                0.10,
            ).value
        )

        self.depth_buffer_size = int(
            self.declare_parameter(
                "depth_buffer_size",
                15,
            ).value
        )

        self.log_period_s = float(
            self.declare_parameter(
                "log_period_s",
                5.0,
            ).value
        )

        # ---------------------------------------------------------------------
        # CONSTRUCTOR STEP 2 - INITIALIZE RESULT / CACHE STATE
        # ---------------------------------------------------------------------

        self.camera_info: Optional[CameraInfo] = None
        self.missing_camera_info = 0

        self.depth_buffer: Deque[Image] = deque(
            maxlen=self.depth_buffer_size
        )

        self.received_detection_arrays = 0
        self.person_detections = 0
        self.valid_3d_detections = 0
        self.no_depth_match = 0
        self.invalid_depth = 0

        self.received_depth_frames = 0
        self.empty_depth_buffer = 0

        self.last_sync_error_s = 0.0
        self.max_observed_sync_error_s = 0.0

        self.last_depth_received_monotonic = None
        self.last_depth_diagnostic_count = 0
        self.last_depth_diagnostic_time = time.monotonic()

        self.last_log_time = self.get_clock().now()

        # ---------------------------------------------------------------------
        # CONSTRUCTOR STEP 3 - CONFIGURE ROS 2 QoS PROFILES
        # ---------------------------------------------------------------------

        depth_qos_auto = resolve_subscription_qos(
            self,
            self.depth_topic,
            fallback_depth=1,
            prefer_sensor_data=True,
        )
        depth_qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # Depth images are sensor streams.
        # Use the standard ROS 2 sensor-data QoS directly here so the subscription
        # remains BEST_EFFORT and does not depend on runtime endpoint discovery.
        #depth_qos = depth_qos_reliable

        depth_qos = qos_profile_sensor_data

        log_qos_resolution(
            self,
            self.depth_topic,
            depth_qos,
        )

        camera_info_qos = resolve_subscription_qos(
            self,
            self.camera_info_topic,
            fallback_depth=10,
        )

        log_qos_resolution(
            self,
            self.camera_info_topic,
            camera_info_qos,
        )

        detections_qos = resolve_subscription_qos(
            self,
            self.detections_topic,
            fallback_depth=10,
        )

        log_qos_resolution(
            self,
            self.detections_topic,
            detections_qos,
        )

        output_qos = reliable_qos(depth=10)

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos,
        )

        #self.depth_probe_rx = 0

        #self.depth_probe_sub = self.create_subscription(
        #    Image,
        #    self.depth_topic,
        #    self.depth_probe_callback,
        #    qos_profile_sensor_data,
        #)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            camera_info_qos,
        )

        self.detections_sub = self.create_subscription(
            DetectionArray,
            self.detections_topic,
            self.detections_callback,
            detections_qos,
        )

        self.output_pub = self.create_publisher(
            DetectionArray,
            self.output_topic,
            output_qos,
        )

        self.depth_watchdog_timer = self.create_timer(
            5.0,
            self.depth_watchdog_callback,
        )

        # ---------------------------------------------------------------------
        # CONSTRUCTOR STEP 4 - REPORT EFFECTIVE CONFIGURATION
        # ---------------------------------------------------------------------

        self.log_initialization_summary()

    # =========================================================================
    # COLLECTION - REALSENSE DEPTH
    # =========================================================================

    def depth_callback(self, msg: Image) -> None:
        """Store recent aligned depth frames for timestamp-based association."""

        first_frame = (
            self.received_depth_frames == 0
        )

        self.received_depth_frames += 1
        self.last_depth_received_monotonic = time.monotonic()

        self.depth_buffer.append(msg)

        if self.received_depth_frames % 5 == 0:
            self.get_logger().info(
                "[DEPTH RX] "
                f"COUNT={self.received_depth_frames} | "
                f"BUFFER={len(self.depth_buffer)}"
            )

        if first_frame:
            self.get_logger().info(
                "[DEPTH] "
                f"READY | "
                f"FRAME={msg.header.frame_id} | "
                f"RES={msg.width}x{msg.height} | "
                f"ENCODING={msg.encoding} | "
                f"BUFFER={len(self.depth_buffer)}"
            )

    def depth_watchdog_callback(self) -> None:

        now = time.monotonic()

        elapsed = max(
            now - self.last_depth_diagnostic_time,
            1e-6,
        )

        received_delta = (
            self.received_depth_frames
            - self.last_depth_diagnostic_count
        )

        rx_hz = received_delta / elapsed

        if self.last_depth_received_monotonic is None:
            silence = float("inf")
        else:
            silence = (
                now - self.last_depth_received_monotonic
            )

        status = (
            "STALLED"
            if silence > 2.0
            else "RECEIVING"
        )

        self.get_logger().info(
            f"[DEPTH WATCHDOG] "
            f"STATUS={status} | "
            f"RX_TOTAL={self.received_depth_frames} | "
            f"RX_HZ={rx_hz:.2f} | "
            f"SILENCE={silence:.2f}s | "
            f"BUFFER={len(self.depth_buffer)}"
        )

        self.last_depth_diagnostic_count = (
            self.received_depth_frames
        )

        self.last_depth_diagnostic_time = now
        
    def depth_callback_Test(self, msg: Image):
        self.received_depth_frames += 1
        self.depth_buffer.append(msg)

        if self.received_depth_frames % 50 == 0:
            self.get_logger().info(
                "[DEPTH CALLBACK TEST] "
                f"DEPTH_RX={self.received_depth_frames} | "
                f"BUFFER={len(self.depth_buffer)}"
            )
    # =========================================================================
    # COLLECTION - CAMERA CALIBRATION
    # =========================================================================
    def depth_probe_callback(self, msg: Image) -> None:
        self.depth_probe_rx += 1

        if self.depth_probe_rx % 50 == 0:
            self.get_logger().info(
                "[IN-NODE DEPTH PROBE] "
                f"RX={self.depth_probe_rx}"
            )
            
    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Cache the latest aligned-depth camera intrinsic calibration."""

        first_message = self.camera_info is None

        self.camera_info = msg

        if first_message:
            self.get_logger().info(
                "[CAMERA INFO] "
                f"READY | "
                f"FRAME={msg.header.frame_id} | "
                f"RES={msg.width}x{msg.height} | "
                f"FX={msg.k[0]:.2f} | "
                f"FY={msg.k[4]:.2f} | "
                f"CX={msg.k[2]:.2f} | "
                f"CY={msg.k[5]:.2f}"
            )
    # =========================================================================
    # PROCESSING - YOLO DETECTION / DEPTH FUSION
    # =========================================================================

    def detections_callback(self, msg: DetectionArray) -> None:
        """
        Fuse YOLO person detections with aligned RealSense depth.

        Publish only valid 3D person detections.
        Preserve the original observation timestamp.
        Never forward an un-fused 2D detection as a 3D person.
        """

        self.received_detection_arrays += 1

        output_msg = copy.deepcopy(msg)
        output_msg.detections = []

        # ---------------------------------------------------------
        # 1. Camera calibration
        # ---------------------------------------------------------
        if self.camera_info is None:
            self.missing_camera_info += 1
            self.output_pub.publish(output_msg)
            self.maybe_log_statistics()
            return

        # ---------------------------------------------------------
        # 2. RGB / Depth synchronization
        # ---------------------------------------------------------
        depth_msg, sync_error_s = self.find_closest_depth(msg)

        if depth_msg is None:
            self.no_depth_match += 1
            self.output_pub.publish(output_msg)
            self.maybe_log_statistics()
            return

        # ---------------------------------------------------------
        # 3. Depth conversion
        # ---------------------------------------------------------
        depth_image = self.depth_message_to_numpy(depth_msg)

        if depth_image is None:
            self.invalid_depth += 1
            self.output_pub.publish(output_msg)
            self.maybe_log_statistics()
            return

        # ---------------------------------------------------------
        # 4. Source frame
        # ---------------------------------------------------------
        source_frame = str(depth_msg.header.frame_id).strip()

        if not source_frame:
            self.invalid_depth += 1
            self.output_pub.publish(output_msg)
            self.maybe_log_statistics()
            return

        output_msg.header.frame_id = source_frame

        # ---------------------------------------------------------
        # 5. Fuse individual persons
        # ---------------------------------------------------------
        valid_detections = []

        for detection in msg.detections:

            if not self.is_valid_person_detection(detection):
                continue

            self.person_detections += 1

            depth_m = self.estimate_person_depth(
                detection,
                depth_image,
            )

            if depth_m is None:
                self.invalid_depth += 1
                continue

            fused_detection = copy.deepcopy(detection)

            self.populate_bbox3d(
                fused_detection,
                depth_m,
                source_frame,
            )

            position = fused_detection.bbox3d.center.position

            if not (
                np.isfinite(position.x)
                and np.isfinite(position.y)
                and np.isfinite(position.z)
                and position.z > 0.0
            ):
                self.invalid_depth += 1
                continue

            valid_detections.append(fused_detection)
            self.valid_3d_detections += 1

        # ---------------------------------------------------------
        # 6. Publish validated 3D observation
        # ---------------------------------------------------------
        output_msg.detections = valid_detections

        self.output_pub.publish(output_msg)
        self.maybe_log_statistics()

    # =========================================================================
    # PROCESSING / VALIDATION - TEMPORAL ASSOCIATION
    # =========================================================================

    def find_closest_depth(
        self,
        detections_msg: DetectionArray,
    ) -> Tuple[Optional[Image], Optional[float]]:
        """Return the buffered depth frame closest to the YOLO source timestamp."""

        if not self.depth_buffer:
            self.empty_depth_buffer += 1
            return None, None

        detection_time = self.stamp_to_seconds(
            detections_msg.header.stamp
        )

        best_msg = None
        best_error = float("inf")

        for depth_msg in self.depth_buffer:

            depth_time = self.stamp_to_seconds(
                depth_msg.header.stamp
            )

            error = abs(
                depth_time - detection_time
            )

            if error < best_error:
                best_error = error
                best_msg = depth_msg

        self.last_sync_error_s = best_error

        self.max_observed_sync_error_s = max(
            self.max_observed_sync_error_s,
            best_error,
        )
        if best_error > self.max_sync_error_s:
            return None, best_error

        return best_msg, best_error

    # =========================================================================
    # PROCESSING / VALIDATION - DEPTH IMAGE CONVERSION
    # =========================================================================

    def depth_message_to_numpy(
        self,
        msg: Image,
    ) -> Optional[np.ndarray]:
        """Convert a ROS 16UC1 depth image to a NumPy uint16 matrix."""

        if msg.encoding not in ("16UC1", "mono16"):
            self.get_logger().warning(
                f"Unsupported depth encoding: {msg.encoding}"
            )
            return None

        dtype = np.dtype(np.uint16)

        if msg.is_bigendian:
            dtype = dtype.newbyteorder(">")

        depth = np.frombuffer(
            msg.data,
            dtype=dtype,
        )

        expected_pixels = (
            int(msg.width) *
            int(msg.height)
        )

        if depth.size != expected_pixels:
            self.get_logger().warning(
                "Depth payload size does not match image dimensions."
            )
            return None

        return depth.reshape(
            int(msg.height),
            int(msg.width),
        )

    # =========================================================================
    # PROCESSING / VALIDATION - PERSON FILTER
    # =========================================================================

    def is_valid_person_detection(
        self,
        detection,
    ) -> bool:
        """Accept only sufficiently confident YOLO person detections."""

        return (
            detection.class_id == self.person_class_id
            and detection.score >= self.min_score
        )

    # =========================================================================
    # PROCESSING - DEPTH ESTIMATION INSIDE PERSON ROI
    # =========================================================================

    def estimate_person_depth(
        self,
        detection,
        depth_image: np.ndarray,
    ) -> Optional[float]:
        """Estimate robust person depth from the central region of the bbox."""

        image_height, image_width = depth_image.shape

        center_x = float(
            detection.bbox.center.position.x
        )

        center_y = float(
            detection.bbox.center.position.y
        )

        bbox_width = float(
            detection.bbox.size.x
        )

        bbox_height = float(
            detection.bbox.size.y
        )

        roi_width = max(
            3.0,
            bbox_width * self.roi_scale,
        )

        roi_height = max(
            3.0,
            bbox_height * self.roi_scale,
        )

        x_min = int(
            max(
                0,
                center_x - roi_width / 2.0,
            )
        )

        x_max = int(
            min(
                image_width,
                center_x + roi_width / 2.0,
            )
        )

        y_min = int(
            max(
                0,
                center_y - roi_height / 2.0,
            )
        )

        y_max = int(
            min(
                image_height,
                center_y + roi_height / 2.0,
            )
        )

        if (
            x_max <= x_min
            or y_max <= y_min
        ):
            return None

        roi = depth_image[
            y_min:y_max,
            x_min:x_max
        ]

        depth_m = (
            roi.astype(np.float32) *
            self.depth_scale
        )

        valid_depth = depth_m[
            (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        ]

        if valid_depth.size == 0:
            return None

        return float(
            np.median(valid_depth)
        )

    # =========================================================================
    # RESULT OBJECT - POPULATE YOLO 3D BOUNDING BOX
    # =========================================================================

    def populate_bbox3d(
        self,
        detection,
        depth_m: float,
        frame_id: str,
    ) -> None:
        """Populate Detection.bbox3d using camera intrinsics and median depth."""

        if self.camera_info is None:
            return

        fx = float(
            self.camera_info.k[0]
        )

        fy = float(
            self.camera_info.k[4]
        )

        cx = float(
            self.camera_info.k[2]
        )

        cy = float(
            self.camera_info.k[5]
        )

        if fx <= 0.0 or fy <= 0.0:
            return

        u = float(
            detection.bbox.center.position.x
        )

        v = float(
            detection.bbox.center.position.y
        )

        x_m = (
            (u - cx) *
            depth_m /
            fx
        )

        y_m = (
            (v - cy) *
            depth_m /
            fy
        )

        detection.bbox3d.center.position.x = x_m
        detection.bbox3d.center.position.y = y_m
        detection.bbox3d.center.position.z = depth_m

        detection.bbox3d.center.orientation.x = 0.0
        detection.bbox3d.center.orientation.y = 0.0
        detection.bbox3d.center.orientation.z = 0.0
        detection.bbox3d.center.orientation.w = 1.0

        detection.bbox3d.size.x = (
            float(detection.bbox.size.x) *
            depth_m /
            fx
        )

        detection.bbox3d.size.y = (
            float(detection.bbox.size.y) *
            depth_m /
            fy
        )

        detection.bbox3d.size.z = 0.0

        detection.bbox3d.frame_id = (
            frame_id
            if frame_id
            else "camera_color_optical_frame"
        )

    # =========================================================================
    # OUTPUT / REPORTING - TERMINAL DIAGNOSTICS
    # =========================================================================

    def log_initialization_summary(self) -> None:
        """Report effective node configuration."""

        separator = "=" * 82
        footer = "-" * 82

        self.get_logger().info(
            f"\n{separator}"
        )

        self.get_logger().info(
            "  PERSON DEPTH FUSION - INITIALIZATION"
        )

        self.get_logger().info(
            separator
        )

        self.get_logger().info(
            f"  Detections input     : {self.detections_topic}"
        )

        self.get_logger().info(
            f"  Aligned depth input  : {self.depth_topic}"
        )

        self.get_logger().info(
            f"  CameraInfo input     : {self.camera_info_topic}"
        )

        self.get_logger().info(
            f"  3D output            : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Person class ID      : {self.person_class_id}"
        )

        self.get_logger().info(
            f"  Minimum score        : {self.min_score:.2f}"
        )

        self.get_logger().info(
            f"  ROI scale            : {self.roi_scale:.2f}"
        )

        self.get_logger().info(
            f"  Max sync error       : {self.max_sync_error_s:.3f} s"
        )

        self.get_logger().info(
            "  Fusion state         : READY"
        )

        self.get_logger().info(
            f"{footer}\n"
        )

    def maybe_log_statistics(self) -> None:
        """Periodically report fusion runtime statistics."""

        now = self.get_clock().now()

        elapsed = (
            now - self.last_log_time
        ).nanoseconds / 1e9

        if elapsed < self.log_period_s:
            return

        self.last_log_time = now

        self.get_logger().info(
            "[PERSON 3D] "
            f"ARRAYS={self.received_detection_arrays} | "
            f"DEPTH_RX={self.received_depth_frames} | "
            f"DEPTH_BUFFER={len(self.depth_buffer)} | "
            f"PERSONS={self.person_detections} | "
            f"VALID_3D={self.valid_3d_detections} | "
            f"NO_CAMERA_INFO={self.missing_camera_info} | "
            f"EMPTY_DEPTH={self.empty_depth_buffer} | "
            f"NO_DEPTH_SYNC={self.no_depth_match} | "
            f"SYNC_ERR={self.last_sync_error_s:.3f}s | "
            f"INVALID_DEPTH={self.invalid_depth}"
        )

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def stamp_to_seconds(stamp) -> float:
        """Convert builtin_interfaces/Time to floating-point seconds."""

        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )


# =============================================================================
# NODE ORCHESTRATION / PROCESS ENTRY POINT
# =============================================================================


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PersonDepthFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()