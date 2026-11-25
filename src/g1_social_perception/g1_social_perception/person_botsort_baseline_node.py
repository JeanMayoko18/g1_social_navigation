
#!/usr/bin/env python3
"""
Experimental Ultralytics BoT-SORT baseline for G1.

Input:
    ROS 2 sensor_msgs/Image (RGB camera stream)

Output:
    /social_perception/benchmark/botsort
    std_msgs/String containing JSON observations.

Isolation:
    Does not publish production PersonDetectionArray.
    Does not assign G1 production identities.
    Does not influence social navigation.

This initial baseline reruns YOLO and must therefore be
evaluated separately from an association-only benchmark.
"""

import json
import math
import time
import uuid

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


class PersonBoTSORTBaseline(Node):

    def __init__(self):
        super().__init__("person_botsort_baseline")

        self.declare_parameter(
            "image_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "output_topic",
            "/social_perception/benchmark/botsort",
        )
        self.declare_parameter(
            "model_path",
            "yolo11n.pt",
        )
        self.declare_parameter(
            "tracker_config",
            "botsort.yaml",
        )
        self.declare_parameter(
            "confidence",
            0.25,
        )
        self.declare_parameter(
            "image_size",
            640,
        )
        self.declare_parameter(
            "maximum_processing_hz",
            0.0,
        )

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )
        self.model_path = str(
            self.get_parameter("model_path").value
        )
        self.tracker_config = str(
            self.get_parameter("tracker_config").value
        )
        self.confidence = float(
            self.get_parameter("confidence").value
        )
        self.image_size = int(
            self.get_parameter("image_size").value
        )
        self.maximum_hz = float(
            self.get_parameter("maximum_processing_hz").value
        )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("INVALID_CONFIDENCE")
        if self.image_size < 32:
            raise ValueError("INVALID_IMAGE_SIZE")
        if not (
            math.isfinite(self.maximum_hz)
            and self.maximum_hz >= 0.0
        ):
            raise ValueError("INVALID_MAXIMUM_PROCESSING_HZ")

        # Import here so failures identify the optional
        # experimental dependency rather than the main pipeline.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ULTRALYTICS_NOT_INSTALLED"
            ) from exc

        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_source_ns = 0
        self.last_processed_at = None
        self.received = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0

        self.publisher = self.create_publisher(
            String,
            self.output_topic,
            reliable_qos(depth=10),
        )

        image_qos = resolve_subscription_qos(
            self,
            self.image_topic,
            fallback_depth=5,
            prefer_sensor_data=True,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos,
        )

        self.get_logger().info(
            "[BOTSORT BASELINE] "
            f"input={self.image_topic} | "
            f"output={self.output_topic} | "
            f"model={self.model_path} | "
            f"tracker={self.tracker_config} | "
            f"session={self.session_id}"
        )

    @staticmethod
    def stamp_to_ns(stamp):
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)

        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError("INVALID_SOURCE_TIMESTAMP")

        value = seconds * 1_000_000_000 + nanoseconds
        if value <= 0:
            raise ValueError("ZERO_SOURCE_TIMESTAMP")

        return value

    def image_callback(self, msg):
        self.received += 1

        try:
            source_ns = self.stamp_to_ns(msg.header.stamp)

            if source_ns <= self.last_source_ns:
                self.skipped += 1
                return

            # Frequency limiting is based on monotonic time.
            # Skipped frames are NOT fed to BoT-SORT.
            now = time.monotonic()

            if (
                self.maximum_hz > 0.0
                and self.last_processed_at is not None
                and now - self.last_processed_at
                    < 1.0 / self.maximum_hz
            ):
                self.skipped += 1
                return

            # Convert to BGR, as expected by OpenCV/Ultralytics.
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            started_ns = time.perf_counter_ns()

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence,
                imgsz=self.image_size,
                classes=[0],
                verbose=False,
            )

            processing_ms = (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0

            if not results:
                raise RuntimeError("EMPTY_YOLO_RESULT")

            result = results[0]
            observations = []

            if result.boxes is not None:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()

                if boxes.id is None:
                    ids = [None] * len(xyxy)
                else:
                    ids = (
                        boxes.id.int().cpu().tolist()
                    )

                for box, score, track_id in zip(
                    xyxy,
                    scores,
                    ids,
                ):
                    # An unassigned detection is not
                    # falsely declared a confirmed track.
                    observations.append({
                        "track_id": (
                            None if track_id is None
                            else int(track_id)
                        ),
                        "confidence": float(score),
                        "bbox_xyxy": [
                            float(value) for value in box
                        ],
                    })

            self.sequence += 1
            self.last_source_ns = source_ns
            self.last_processed_at = time.monotonic()
            self.processed += 1

            payload = {
                "schema_version": 1,
                "method": "botsort",
                "tracking_session_id": self.session_id,
                "source_sequence": self.sequence,
                "source_stamp_ns": source_ns,
                "source_frame_id": msg.header.frame_id,
                "processing_ms": processing_ms,
                "detections": observations,
                "detection_count": len(observations),
                "reid_enabled": False,
                "detection_included_in_timing": True,
            }

            self.publisher.publish(
                String(
                    data=json.dumps(
                        payload,
                        allow_nan=False,
                    )
                )
            )

        except Exception as exc:
            self.failed += 1
            self.get_logger().error(
                "[BOTSORT BASELINE] "
                + type(exc).__name__
                + ": "
                + str(exc),
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonBoTSORTBaseline()
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


"""
conda activate yolos

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip check


unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip install \
  filelock \
  fsspec \
  typing-extensions \
  opencv-python \
  pyyaml \
  requests
"""