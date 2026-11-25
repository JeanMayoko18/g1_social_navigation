
#!/usr/bin/env python3
"""
GPU worker for independent BoT-SORT baselines.

Run using the Conda yolos Python interpreter.
No ROS imports.

Protocol:
    stdin:  one JSON object per line
    stdout: one G1RESULT-prefixed JSON object per line

Both trackers process the same source image, in the
same source order, but retain independent states.

This version includes YOLO inference in each timing.
"""

import base64
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PREFIX = "G1RESULT "


def emit(payload):
    sys.stdout.write(
        PREFIX + json.dumps(payload, allow_nan=False) + "\n"
    )
    sys.stdout.flush()


def decode_image(encoded):
    raw = base64.b64decode(
        encoded,
        validate=True,
    )

    array = np.frombuffer(raw, dtype=np.uint8)

    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None or frame.ndim != 3:
        raise ValueError("INVALID_JPEG_FRAME")

    return frame


def extract_tracks(result):
    observations = []

    if result.boxes is None:
        return observations

    boxes = result.boxes

    coordinates = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    if boxes.id is None:
        identifiers = [None] * len(coordinates)
    else:
        identifiers = boxes.id.int().cpu().tolist()

    for bbox, score, track_id in zip(
        coordinates,
        confidences,
        identifiers,
    ):
        observations.append({
            "track_id": (
                None if track_id is None
                else int(track_id)
            ),
            "confidence": float(score),
            "bbox_xyxy": [float(x) for x in bbox],
        })

    return observations


class TrackerInstance:

    def __init__(
        self,
        *,
        method,
        model_path,
        tracker_config,
        reid_enabled,
    ):
        self.method = method
        self.model = YOLO(model_path)
        self.tracker_config = str(tracker_config)
        self.reid_enabled = reid_enabled
        self.session_id = uuid.uuid4().hex

    def process(self, frame, source):
        start = time.perf_counter_ns()

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=source["confidence"],
            imgsz=source["image_size"],
            classes=[0],
            verbose=False,
        )

        duration_ms = (
            time.perf_counter_ns() - start
        ) / 1_000_000.0

        if not results:
            raise RuntimeError("EMPTY_TRACKING_RESULTS")

        detections = extract_tracks(results[0])

        return {
            "schema_version": 1,
            "method": self.method,
            "tracking_session_id": self.session_id,
            "source_session_id": source["source_session_id"],
            "source_sequence": source["source_sequence"],
            "source_stamp_ns": source["source_stamp_ns"],
            "source_frame_id": source["source_frame_id"],
            "processing_ms": duration_ms,
            "detection_included_in_timing": True,
            "reid_enabled": self.reid_enabled,
            "detection_count": len(detections),
            "detections": detections,
            "status": "OK",
        }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: botsort_worker.py "
            "MODEL_PATH BOTSORT_YAML BOTSORT_REID_YAML"
        )

    model_path = sys.argv[1]
    baseline_yaml = Path(sys.argv[2]).expanduser()
    reid_yaml = Path(sys.argv[3]).expanduser()

    for path in (baseline_yaml, reid_yaml):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    # Separate YOLO objects preserve independent tracker state.
    # Their model weights and image input are identical.
    trackers = (
        TrackerInstance(
            method="botsort",
            model_path=model_path,
            tracker_config=baseline_yaml,
            reid_enabled=False,
        ),
        TrackerInstance(
            method="botsort_reid",
            model_path=model_path,
            tracker_config=reid_yaml,
            reid_enabled=True,
        ),
    )

    last_source_ns = 0
    source_session = None

    for line in sys.stdin:
        if not line.strip():
            continue

        request_id = None

        try:
            request = json.loads(line)
            request_id = request["request_id"]

            if request.get("schema_version") != 1:
                raise ValueError("UNSUPPORTED_PROTOCOL_VERSION")

            source = request["source"]

            stamp_ns = int(source["source_stamp_ns"])
            session = str(source["source_session_id"])

            if not session or stamp_ns <= 0:
                raise ValueError("INVALID_SOURCE")

            # A different stream must not inherit old tracker
            # states: the bridge must restart the worker.
            if source_session is None:
                source_session = session
            elif source_session != session:
                raise ValueError("SOURCE_SESSION_CHANGED")

            if stamp_ns <= last_source_ns:
                raise ValueError("NON_MONOTONIC_FRAME")

            frame = decode_image(request["jpeg_base64"])

            results = []
            for tracker in trackers:
                results.append(
                    tracker.process(frame, source)
                )

            last_source_ns = stamp_ns

            emit({
                "request_id": request_id,
                "status": "OK",
                "results": results,
            })

        except Exception as exc:
            emit({
                "request_id": request_id,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })


if __name__ == "__main__":
    main()
