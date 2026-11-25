
#!/usr/bin/env python3
"""ROS 2 source for a single external YOLO tracking worker."""

import base64
import json
import math
import os
import queue
import selectors
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection, DetectionArray

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


PREFIX = b"G1RESULT "
MAX_RESPONSE_BYTES = 2_000_000


def image_stamp_ns(stamp):
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if (
        sec < 0
        or not 0 <= nanosec < 1_000_000_000
    ):
        raise ValueError("INVALID_IMAGE_TIMESTAMP")

    value = sec * 1_000_000_000 + nanosec

    if value <= 0:
        raise ValueError("ZERO_IMAGE_TIMESTAMP")

    return value


def encode_image(msg, quality):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(
            f"UNSUPPORTED_ENCODING:{msg.encoding}"
        )

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if (
        height <= 0
        or width <= 0
        or step < width * 3
        or len(msg.data) < height * step
    ):
        raise ValueError("INVALID_IMAGE_LAYOUT")

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    pixels = raw[:height * step].reshape(
        height, step
    )[:, :width * 3].reshape(
        height, width, 3
    )

    if msg.encoding == "rgb8":
        bgr = cv2.cvtColor(
            pixels, cv2.COLOR_RGB2BGR
        )
    else:
        bgr = np.ascontiguousarray(pixels)

    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not ok:
        raise RuntimeError("JPEG_ENCODING_FAILED")

    if len(encoded) > 8_000_000:
        raise ValueError("JPEG_TOO_LARGE")

    return base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")


class PersonYoloTrackSourceNode(Node):

    def __init__(self):
        super().__init__("person_yolo_track_source")

        defaults = {
            "image_topic": "/perception/transport/color",
            "output_topic": "/yolo/detections",
            "conda_python": (
                "~/miniconda3/envs/yolos/bin/python"
            ),
            "model_path": "~/g1_ws/models/yolo11n.pt",
            "tracker_config": (
                "~/g1_ws/src/g1_social_perception/"
                "config/benchmark/botsort_reid.yaml"
            ),
            "publish_debug_image": False,
            "debug_image_topic": "/yolo/dbg_image",
            "confidence": 0.20,
            "image_size": 640,
            "jpeg_quality": 90,
            "maximum_processing_hz": 10.0,
            "worker_timeout_sec": 30.0,
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )
        self.confidence = float(
            self.get_parameter("confidence").value
        )
        self.image_size = int(
            self.get_parameter("image_size").value
        )
        self.jpeg_quality = int(
            self.get_parameter("jpeg_quality").value
        )
        self.maximum_hz = float(
            self.get_parameter(
                "maximum_processing_hz"
            ).value
        )
        self.timeout = float(
            self.get_parameter(
                "worker_timeout_sec"
            ).value
        )
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )

        self.debug_image_topic = str(
            self.get_parameter("debug_image_topic").value
        )

        self.python_path = Path(
            str(self.get_parameter("conda_python").value)
        ).expanduser()

        self.model_path = Path(
            str(self.get_parameter("model_path").value)
        ).expanduser()

        self.tracker_path = Path(
            str(self.get_parameter("tracker_config").value)
        ).expanduser()

        self.worker_path = (
            Path(__file__).resolve().parent
            / "benchmark/person_yolo_track_worker.py"
        )

        for path in (
            self.python_path,
            self.model_path,
            self.tracker_path,
            self.worker_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(str(path))

        if not (
            math.isfinite(self.confidence)
            and 0 <= self.confidence <= 1
            and self.image_size >= 32
            and 1 <= self.jpeg_quality <= 100
            and math.isfinite(self.maximum_hz)
            and self.maximum_hz >= 0
            and math.isfinite(self.timeout)
            and self.timeout > 0
        ):
            raise ValueError("INVALID_PARAMETERS")

        if self.image_topic == self.output_topic:
            raise ValueError("TOPIC_COLLISION")

        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.process = None

        self.last_stamp_ns = 0
        self.last_accepted_at = None
        self.sequence = 0

        self.source_session = None

        self.stats = {
            "received": 0,
            "accepted": 0,
            "published": 0,
            "detections": 0,
            "dropped": 0,
            "rejected": 0,
            "errors": 0,
            "worker_starts": 0,
        }

        self.stats_lock = threading.Lock()

        self.publisher = self.create_publisher(
            DetectionArray,
            self.output_topic,
            reliable_qos(depth=10),
        )
        self.debug_publisher = None

        if self.publish_debug_image:
            self.debug_publisher = self.create_publisher(
                Image,
                self.debug_image_topic,
                reliable_qos(depth=1),
            )

        qos = resolve_subscription_qos(
            self,
            self.image_topic,
            fallback_depth=5,
            prefer_sensor_data=True,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos,
        )

        self.thread = threading.Thread(
            target=self.processing_loop,
            daemon=True,
        )
        self.thread.start()

        self.timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self.get_logger().info(
            "[YOLO TRACK SOURCE] "
            f"{self.image_topic} -> {self.output_topic}"
        )

    def increment(self, name, value=1):
        with self.stats_lock:
            self.stats[name] += value

    def image_callback(self, msg):
        self.increment("received")

        try:
            stamp_ns = image_stamp_ns(
                msg.header.stamp
            )

            if not str(msg.header.frame_id).strip():
                raise ValueError("EMPTY_IMAGE_FRAME")

            if stamp_ns <= self.last_stamp_ns:
                raise ValueError("NON_MONOTONIC_IMAGE")

            now = time.monotonic()

            if (
                self.maximum_hz > 0
                and self.last_accepted_at is not None
                and now - self.last_accepted_at
                < 1.0 / self.maximum_hz
            ):
                self.increment("dropped")
                return

            encoded = encode_image(
                msg, self.jpeg_quality
            )

            self.sequence += 1

            frame = {
                "sequence": self.sequence,
                "stamp_ns": stamp_ns,
                "header": msg.header,
                "jpeg_base64": encoded,
                "debug_image": msg if self.publish_debug_image else None,
            }

            if self.frames.full():
                try:
                    self.frames.get_nowait()
                    self.increment("dropped")
                except queue.Empty:
                    pass

            self.frames.put_nowait(frame)

            self.last_stamp_ns = stamp_ns
            self.last_accepted_at = now

            self.increment("accepted")

        except queue.Full:
            self.increment("dropped")
        except Exception as exc:
            self.increment("rejected")
            self.get_logger().warning(
                f"[YOLO TRACK INPUT] {exc}",
                throttle_duration_sec=2.0,
            )
    
    def publish_annotated_image(self, frame, detections):

        if self.debug_publisher is None:
            return

        source = frame.get("debug_image")

        if source is None:
            return

        try:
            height = int(source.height)
            width = int(source.width)
            step = int(source.step)

            pixels = np.frombuffer(
                source.data, dtype=np.uint8
            )

            image = pixels[:height * step].reshape(
                height, step
            )[:, :width * 3].reshape(
                height, width, 3
            ).copy()

            if source.encoding == "rgb8":
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            for detection in detections.detections:

                cx = float(detection.bbox.center.position.x)
                cy = float(detection.bbox.center.position.y)
                bw = float(detection.bbox.size.x)
                bh = float(detection.bbox.size.y)

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"person {float(detection.score):.2f}"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            debug_msg = Image()
            debug_msg.header = source.header
            debug_msg.height = height
            debug_msg.width = width
            debug_msg.encoding = "bgr8"
            debug_msg.is_bigendian = 0
            debug_msg.step = width * 3
            debug_msg.data = image.tobytes()

            self.debug_publisher.publish(debug_msg)

        except Exception as exc:
            self.get_logger().warning(
                f"[YOLO DEBUG IMAGE] {exc}",
                throttle_duration_sec=2.0,
            )
            
    def start_worker(self):
        environment = os.environ.copy()

        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"

        # Every worker restart creates a new source session.
        # Downstream nodes must not reuse prior tracker identities.
        new_source_session = uuid.uuid4().hex

        process = subprocess.Popen(
            [
                str(self.python_path),
                "-u",
                str(self.worker_path),
                str(self.model_path),
                str(self.tracker_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=0,
        )

        try:
            os.set_blocking(
                process.stdin.fileno(),
                False,
            )

            os.set_blocking(
                process.stdout.fileno(),
                False,
            )

        except Exception:
            process.terminate()
            process.wait(timeout=2.0)
            raise

        self.process = process
        self.source_session = new_source_session

        # Set when the first successful worker response arrives.
        self.active_worker_session_id = None

        self.increment("worker_starts")

        self.get_logger().info(
            "[YOLO TRACK] Worker started | "
            f"source_session={self.source_session} | "
            f"tracker_config={self.tracker_path}"
        )
        
    def stop_worker(self):
        process = self.process
        self.process = None

        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()

    def exchange(self, frame):
        if (
            self.process is None
            or self.process.poll() is not None
        ):
            self.stop_worker()
            self.start_worker()

        request_id = uuid.uuid4().hex

        request = {
            "schema_version": 1,
            "request_id": request_id,
            "source": {
                "source_session_id": self.source_session,
                "source_sequence": frame["sequence"],
                "source_stamp_ns": frame["stamp_ns"],
                "source_frame_id":
                    frame["header"].frame_id,
                "confidence": self.confidence,
                "image_size": self.image_size,
            },
            "jpeg_base64": frame["jpeg_base64"],
        }

        payload = (
            json.dumps(request) + "\n"
        ).encode("utf-8")

        deadline = time.monotonic() + self.timeout

        with selectors.DefaultSelector() as selector:
            selector.register(
                self.process.stdin,
                selectors.EVENT_WRITE,
            )

            written = 0

            while written < len(payload):
                if self.stop_event.is_set():
                    raise RuntimeError("SHUTDOWN")

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError("WORKER_WRITE_TIMEOUT")

                if not selector.select(
                    min(0.2, remaining)
                ):
                    continue

                count = os.write(
                    self.process.stdin.fileno(),
                    payload[written:],
                )

                if count <= 0:
                    raise RuntimeError("WORKER_WRITE_FAILED")

                written += count

        buffer = b""

        with selectors.DefaultSelector() as selector:
            selector.register(
                self.process.stdout,
                selectors.EVENT_READ,
            )

            while not self.stop_event.is_set():
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError(
                        "WORKER_RESPONSE_TIMEOUT"
                    )

                ready = selector.select(
                    min(0.2, remaining)
                )

                if not ready:
                    if self.process.poll() is not None:
                        raise RuntimeError("WORKER_EXITED")
                    continue

                chunk = os.read(
                    self.process.stdout.fileno(),
                    65536,
                )

                if not chunk:
                    raise RuntimeError("WORKER_STDOUT_CLOSED")

                buffer += chunk

                if len(buffer) > MAX_RESPONSE_BYTES:
                    raise RuntimeError(
                        "WORKER_OUTPUT_TOO_LARGE"
                    )

                while b"\n" in buffer:
                    line, buffer = buffer.split(
                        b"\n", 1
                    )

                    if not line.startswith(PREFIX):
                        continue

                    response = json.loads(
                        line[len(PREFIX):]
                    )

                    if response.get("request_id") != request_id:
                        raise RuntimeError(
                            "WORKER_REQUEST_MISMATCH"
                        )

                    return response

        raise RuntimeError("SHUTDOWN")

    def build_message(self, response, frame):

        if response.get("status") != "OK":
            raise RuntimeError(
                "WORKER_ERROR:"
                + str(response.get("error"))
            )

        if (
            response.get("schema_version") != 1
            or response.get("source_session_id")
                != self.source_session
            or response.get("source_sequence")
                != frame["sequence"]
            or response.get("source_stamp_ns")
                != frame["stamp_ns"]
            or response.get("source_frame_id")
                != frame["header"].frame_id
        ):
            raise RuntimeError(
                "WORKER_SOURCE_MISMATCH"
            )

        worker_session = str(
            response.get("worker_session_id", "")
        ).strip()

        if not worker_session:
            raise ValueError(
                "MISSING_WORKER_SESSION"
            )

        observations = response.get("detections")

        if (
            not isinstance(observations, list)
            or len(observations)
                != response.get("detection_count")
        ):
            raise ValueError(
                "INVALID_WORKER_DETECTIONS"
            )

        output = DetectionArray()
        output.header = frame["header"]

        seen_ids = set()

        for item in observations:

            x1, y1, x2, y2 = map(
                float,
                item["bbox_xyxy"],
            )

            score = float(item["confidence"])
            class_id = int(item["class_id"])

            if (
                not all(
                    math.isfinite(v)
                    for v in (x1, y1, x2, y2, score)
                )
                or x2 <= x1
                or y2 <= y1
                or not 0.0 <= score <= 1.0
                or class_id != 0
            ):
                raise ValueError(
                    "INVALID_DETECTION"
                )

            det = Detection()
            det.class_id = 0
            det.class_name = "person"
            det.score = score

            track_id = item.get("track_id")

            if track_id is None:
                det.id = ""

            else:
                track_id = int(track_id)

                if (
                    track_id < 0
                    or track_id in seen_ids
                ):
                    raise ValueError(
                        "DUPLICATE_TRACK_ID"
                    )

                seen_ids.add(track_id)

                # The canonical tracker key includes both
                # worker identity and numerical track ID.
                det.id = (
                    worker_session
                    + ":"
                    + str(track_id)
                )

            det.bbox.center.position.x = (
                x1 + x2
            ) / 2.0

            det.bbox.center.position.y = (
                y1 + y2
            ) / 2.0

            det.bbox.center.theta = 0.0

            det.bbox.size.x = x2 - x1
            det.bbox.size.y = y2 - y1

            output.detections.append(det)

        # Only accept the worker session after validating
        # the entire response. Never accept a session change
        # silently within one running worker.
        active_session = getattr(
            self,
            "active_worker_session_id",
            None,
        )

        if active_session is None:
            self.active_worker_session_id = worker_session

            self.get_logger().info(
                "[YOLO TRACK] "
                "Worker session established | "
                f"worker_session={worker_session}"
            )

        elif worker_session != active_session:
            raise RuntimeError(
                "WORKER_SESSION_CHANGED_WITHOUT_RESTART"
            )

        return output

    def processing_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frames.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue

            try:
                response = self.exchange(frame)
                output = self.build_message(
                    response, frame
                )

                self.publisher.publish(output)

                if self.publish_debug_image:
                    self.publish_annotated_image(frame, output)

                self.increment("published")
                self.increment(
                    "detections",
                    len(output.detections),
                )

            except Exception as exc:
                self.increment("errors")

                if not self.stop_event.is_set():
                    self.get_logger().error(
                        "[YOLO TRACK] "
                        f"{type(exc).__name__}: {exc}",
                        throttle_duration_sec=2.0,
                    )

                self.stop_worker()

        self.stop_worker()

    def log_diagnostics(self):
        with self.stats_lock:
            snapshot = dict(self.stats)

        self.get_logger().info(
            "[YOLO TRACK] "
            + " | ".join(
                f"{key.upper()}={value}"
                for key, value in snapshot.items()
            )
        )

    def destroy_node(self):
        self.stop_event.set()
        self.stop_worker()

        if hasattr(self, "thread"):
            self.thread.join(timeout=3.0)

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonYoloTrackSourceNode()
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
