
#!/usr/bin/env python3
"""
ROS 2 Humble <-> Conda BoT-SORT bridge.

- Never performs GPU inference in a ROS callback.
- Bounded queue: one pending image maximum.
- Explicit worker response timeout.
- Worker restart creates a new tracking session.
- No imports from Ultralytics or cv_bridge.
- No publications to production identity topics.

Supported sensor_msgs/Image encodings:
    rgb8, bgr8
"""

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
from std_msgs.msg import String

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


PREFIX = b"G1RESULT "
METHODS = {"botsort", "botsort_reid"}


def stamp_to_ns(stamp):
    sec = int(stamp.sec)
    nsec = int(stamp.nanosec)

    if sec < 0 or nsec < 0 or nsec >= 1_000_000_000:
        raise ValueError("INVALID_IMAGE_TIMESTAMP")

    result = sec * 1_000_000_000 + nsec

    if result <= 0:
        raise ValueError("ZERO_IMAGE_TIMESTAMP")

    return result


def encode_ros_image(msg, jpeg_quality):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(
            f"UNSUPPORTED_IMAGE_ENCODING:{msg.encoding}"
        )

    if bool(msg.is_bigendian):
        # No multi-byte pixel components are accepted here.
        # rgb8/bgr8 are byte-oriented and unaffected.
        pass

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0 or step < width * 3:
        raise ValueError("INVALID_IMAGE_DIMENSIONS")

    raw = np.frombuffer(
        msg.data,
        dtype=np.uint8,
    )

    if raw.size < height * step:
        raise ValueError("TRUNCATED_IMAGE")

    array = raw[:height * step].reshape(
        height,
        step,
    )

    pixels = array[:, :width * 3].reshape(
        height,
        width,
        3,
    )

    if msg.encoding == "rgb8":
        bgr = cv2.cvtColor(
            pixels,
            cv2.COLOR_RGB2BGR,
        )
    else:
        bgr = np.ascontiguousarray(pixels)

    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )

    if not ok:
        raise RuntimeError("JPEG_ENCODING_FAILED")

    return base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")


class PersonBoTSORTBridgeNode(Node):

    def __init__(self):
        super().__init__("person_botsort_bridge")

        root = Path(__file__).resolve().parent
        share_root = root.parent

        defaults = {
            "image_topic": "/perception/transport/color",
            "conda_python": (
                "~/miniconda3/envs/yolos/bin/python"
            ),
            "model_path": (
                "~/g1_ws/models/yolo11n.pt"
            ),
            "baseline_config": str(
                share_root
                / "config/benchmark/botsort_baseline.yaml"
            ),
            "reid_config": str(
                share_root
                / "config/benchmark/botsort_reid.yaml"
            ),
            "confidence": 0.25,
            "image_size": 640,
            "jpeg_quality": 90,
            "maximum_processing_hz": 0.0,
            "worker_timeout_sec": 30.0,
            "diagnostics_period_sec": 10.0,
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.image_topic = str(
            self.get_parameter("image_topic").value
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
        self.worker_timeout = float(
            self.get_parameter(
                "worker_timeout_sec"
            ).value
        )

        self.diagnostics_period = float(
            self.get_parameter(
                "diagnostics_period_sec"
            ).value
        )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("INVALID_CONFIDENCE")

        if self.image_size < 32:
            raise ValueError("INVALID_IMAGE_SIZE")

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("INVALID_JPEG_QUALITY")

        if not (
            math.isfinite(self.maximum_hz)
            and self.maximum_hz >= 0
        ):
            raise ValueError("INVALID_MAXIMUM_HZ")

        if not (
            math.isfinite(self.worker_timeout)
            and self.worker_timeout > 0
        ):
            raise ValueError("INVALID_WORKER_TIMEOUT")

        if self.diagnostics_period <= 0:
            raise ValueError("INVALID_DIAGNOSTICS_PERIOD")

        self.python_path = Path(
            str(self.get_parameter("conda_python").value)
        ).expanduser()

        self.worker_path = (
            root / "benchmark/botsort_worker.py"
        )

        self.model_path = Path(
            str(self.get_parameter("model_path").value)
        ).expanduser()

        self.baseline_path = Path(
            str(self.get_parameter("baseline_config").value)
        ).expanduser()

        self.reid_path = Path(
            str(self.get_parameter("reid_config").value)
        ).expanduser()

        for path in (
            self.python_path,
            self.worker_path,
            self.model_path,
            self.baseline_path,
            self.reid_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"BENCHMARK_FILE_MISSING:{path}"
                )

        # Never overwrite Node.publishers or Node.subscriptions.
        self.benchmark_publishers = {
            name: self.create_publisher(
                String,
                "/social_perception/benchmark/" + name,
                reliable_qos(depth=10),
            )
            for name in METHODS
        }

        self.timing_publisher = self.create_publisher(
            String,
            "/social_perception/benchmark/bridge_timing",
            reliable_qos(depth=100),
        )

        self.pending_frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()

        self.worker_process = None
        self.worker_stdout_buffer = b""

        self.source_session_id = None
        self.sequence = 0
        self.last_accepted_stamp_ns = 0
        self.last_accepted_monotonic = None

        self.stats = {
            "received": 0,
            "accepted": 0,
            "processed": 0,
            "dropped_queue": 0,
            "dropped_rate": 0,
            "rejected": 0,
            "errors": 0,
            "restarts": 0,
        }

        qos = resolve_subscription_qos(
            self,
            self.image_topic,
            fallback_depth=5,
            prefer_sensor_data=True,
        )

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos,
        )

        self.processing_thread = threading.Thread(
            target=self.processing_loop,
            name="botsort_worker_bridge",
            daemon=True,
        )
        self.processing_thread.start()

        self.diagnostics_timer = self.create_timer(
            self.diagnostics_period,
            self.log_diagnostics,
        )

        self.get_logger().info(
            "[BOTSORT BRIDGE] "
            f"input={self.image_topic} "
            f"model={self.model_path} "
            f"timeout={self.worker_timeout}s"
        )

    def publish_timing(self, payload):
        payload["schema_version"] = 1

        self.timing_publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    allow_nan=False,
                )
            )
        )

    def image_callback(self, msg):
        with self.state_lock:
            self.stats["received"] += 1

        try:
            source_ns = stamp_to_ns(msg.header.stamp)

            now = time.monotonic()

            if source_ns <= self.last_accepted_stamp_ns:
                return

            if (
                self.maximum_hz > 0
                and self.last_accepted_monotonic is not None
                and now - self.last_accepted_monotonic
                    < 1.0 / self.maximum_hz
            ):
                with self.state_lock:
                    self.stats["dropped_rate"] += 1
                return

            encoded = encode_ros_image(
                msg,
                self.jpeg_quality,
            )

            self.sequence += 1

            frame = {
                "source_sequence": self.sequence,
                "source_stamp_ns": source_ns,
                "source_frame_id": str(
                    msg.header.frame_id
                ),
                "jpeg_base64": encoded,
                "queued_at_monotonic": now,
            }

            # Preserve the newest observation when a backlog
            # exists. Each replaced frame is explicitly counted.
            if self.pending_frames.full():
                try:
                    self.pending_frames.get_nowait()
                    with self.state_lock:
                        self.stats["dropped_queue"] += 1
                except queue.Empty:
                    pass

            try:
                self.pending_frames.put_nowait(frame)
            except queue.Full:
                with self.state_lock:
                    self.stats["dropped_queue"] += 1
                return

            self.last_accepted_stamp_ns = source_ns
            self.last_accepted_monotonic = now

            with self.state_lock:
                self.stats["accepted"] += 1

        except Exception as exc:
            with self.state_lock:
                self.stats["rejected"] += 1

            self.get_logger().warning(
                "[BOTSORT INPUT] "
                f"{type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )

    def start_worker(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"

        self.source_session_id = uuid.uuid4().hex
        self.worker_stdout_buffer = b""

        self.worker_process = subprocess.Popen(
            [
                str(self.python_path),
                "-u",
                str(self.worker_path),
                str(self.model_path),
                str(self.baseline_path),
                str(self.reid_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            #stderr=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=environment,
        )

        os.set_blocking(
            self.worker_process.stdout.fileno(),
            False,
        )

        with self.state_lock:
            self.stats["restarts"] += 1

        self.get_logger().info(
            "[BOTSORT BRIDGE] Worker started; "
            f"session={self.source_session_id}"
        )

    def stop_worker(self):
        process = self.worker_process
        self.worker_process = None
        self.worker_stdout_buffer = b""

        if process is None:
            return

        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

        for stream in (
            process.stdin,
            process.stdout,
        ):
            if stream is not None:
                stream.close()

    def read_response(self, request_id):
        process = self.worker_process

        if process is None:
            raise RuntimeError("WORKER_NOT_STARTED")

        deadline = (
            time.monotonic() + self.worker_timeout
        )

        selector = selectors.DefaultSelector()

        try:
            selector.register(
                process.stdout,
                selectors.EVENT_READ,
            )

            while not self.stop_event.is_set():
                if process.poll() is not None:
                    raise RuntimeError(
                        "WORKER_EXITED"
                    )

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError(
                        "WORKER_RESPONSE_TIMEOUT"
                    )

                ready = selector.select(
                    timeout=min(remaining, 0.2)
                )

                if not ready:
                    continue

                chunk = os.read(
                    process.stdout.fileno(),
                    65536,
                )

                if not chunk:
                    raise RuntimeError(
                        "WORKER_STDOUT_CLOSED"
                    )

                self.worker_stdout_buffer += chunk

                if len(self.worker_stdout_buffer) > 4_000_000:
                    raise RuntimeError(
                        "WORKER_RESPONSE_TOO_LARGE"
                    )

                while b"\n" in self.worker_stdout_buffer:
                    line, self.worker_stdout_buffer = (
                        self.worker_stdout_buffer.split(
                            b"\n", 1
                        )
                    )

                    if not line.startswith(PREFIX):
                        continue

                    payload = json.loads(
                        line[len(PREFIX):].decode("utf-8")
                    )

                    if (
                        payload.get("request_id")
                        != request_id
                    ):
                        raise RuntimeError(
                            "WORKER_REQUEST_MISMATCH"
                        )

                    return payload

            raise RuntimeError("WORKER_STOP_REQUESTED")

        finally:
            selector.close()

    def send_frame(self, frame):
        if (
            self.worker_process is None
            or self.worker_process.poll() is not None
        ):
            self.stop_worker()
            self.start_worker()

        request_id = uuid.uuid4().hex

        request = {
            "schema_version": 1,
            "request_id": request_id,
            "source": {
                "source_session_id": self.source_session_id,
                "source_sequence": frame["source_sequence"],
                "source_stamp_ns": frame["source_stamp_ns"],
                "source_frame_id": frame["source_frame_id"],
                "confidence": self.confidence,
                "image_size": self.image_size,
            },
            "jpeg_base64": frame["jpeg_base64"],
        }

        payload = (
            json.dumps(
                request,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")

        # stdin should not remain blocked indefinitely if the
        # worker stops reading. Write with a deadline.
        descriptor = self.worker_process.stdin.fileno()
        os.set_blocking(descriptor, False)

        deadline = time.monotonic() + self.worker_timeout
        offset = 0

        selector = selectors.DefaultSelector()

        try:
            selector.register(
                descriptor,
                selectors.EVENT_WRITE,
            )

            while offset < len(payload):
                if self.stop_event.is_set():
                    raise RuntimeError(
                        "WORKER_STOP_REQUESTED"
                    )

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise TimeoutError(
                        "WORKER_REQUEST_TIMEOUT"
                    )

                if not selector.select(
                    min(remaining, 0.2)
                ):
                    continue

                try:
                    written = os.write(
                        descriptor,
                        payload[offset:],
                    )
                except BlockingIOError:
                    continue

                if written <= 0:
                    raise RuntimeError(
                        "WORKER_WRITE_FAILED"
                    )

                offset += written

        finally:
            selector.close()

        response = self.read_response(request_id)

        if response.get("status") != "OK":
            raise RuntimeError(
                "WORKER_PROCESSING_ERROR:"
                + str(response.get("error_type"))
                + ":"
                + str(response.get("error"))
            )

        results = response.get("results")

        if (
            not isinstance(results, list)
            or len(results) != 2
            or {r.get("method") for r in results} != METHODS
        ):
            raise RuntimeError(
                "INVALID_WORKER_RESULTS"
            )

        for result in results:
            if (
                result.get("source_stamp_ns")
                != frame["source_stamp_ns"]
                or result.get("source_sequence")
                != frame["source_sequence"]
            ):
                raise RuntimeError(
                    "WORKER_SOURCE_MISMATCH"
                )

        for result in results:
            self.benchmark_publishers[
                result["method"]
            ].publish(
                String(
                    data=json.dumps(
                        result,
                        allow_nan=False,
                    )
                )
            )

    def processing_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.pending_frames.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue

            started = time.perf_counter_ns()

            try:
                self.send_frame(frame)

                with self.state_lock:
                    self.stats["processed"] += 1

                status = "OK"
                error = None

            except Exception as exc:
                with self.state_lock:
                    self.stats["errors"] += 1

                status = "ERROR"
                error = (
                    type(exc).__name__ + ":" + str(exc)
                )

                if not self.stop_event.is_set():
                    self.get_logger().error(
                        "[BOTSORT WORKER] " + error,
                        throttle_duration_sec=2.0,
                    )

                self.stop_worker()

            finally:
                duration_ms = (
                    time.perf_counter_ns() - started
                ) / 1_000_000.0

                self.publish_timing({
                    "record_type": "bridge_frame",
                    "source_sequence":
                        frame["source_sequence"],
                    "source_stamp_ns":
                        frame["source_stamp_ns"],
                    "source_frame_id":
                        frame["source_frame_id"],
                    "source_session_id":
                        self.source_session_id,
                    "status": status,
                    "error": error,
                    "bridge_processing_ms":
                        duration_ms,
                })

        self.stop_worker()

    def log_diagnostics(self):
        with self.state_lock:
            snapshot = dict(self.stats)

        self.get_logger().info(
            "[BOTSORT BRIDGE] "
            + " | ".join(
                f"{key.upper()}={value}"
                for key, value in snapshot.items()
            )
        )

    def destroy_node(self):
        self.stop_event.set()

        if hasattr(self, "processing_thread"):
            self.processing_thread.join(
                timeout=self.worker_timeout + 3.0
            )

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PersonBoTSORTBridgeNode()
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
