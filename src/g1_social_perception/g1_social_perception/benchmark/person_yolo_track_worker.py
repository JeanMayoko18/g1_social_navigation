
#!/usr/bin/env python3
"""Single YOLO + BoT-SORT/ReID worker. No ROS imports."""

import base64
import json
import math
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PREFIX = "G1RESULT "
MAX_IMAGE_BYTES = 8_000_000


def emit(payload):
    print(
        PREFIX + json.dumps(payload, allow_nan=False),
        flush=True,
    )


def decode_image(encoded):
    raw = base64.b64decode(encoded, validate=True)

    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("INVALID_IMAGE_SIZE")

    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None or frame.ndim != 3:
        raise ValueError("INVALID_IMAGE")

    return frame


def extract_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes

    coords = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()

    if boxes.id is None:
        identifiers = [None] * len(coords)
    else:
        identifiers = boxes.id.int().cpu().tolist()

    if not (
        len(coords) == len(scores)
        == len(classes) == len(identifiers)
    ):
        raise RuntimeError("INCONSISTENT_YOLO_OUTPUT")

    seen_ids = set()

    for xyxy, score, class_id, track_id in zip(
        coords, scores, classes, identifiers
    ):
        if class_id != 0:
            continue

        x1, y1, x2, y2 = map(float, xyxy)
        confidence = float(score)

        if not (
            all(math.isfinite(v) for v in (x1, y1, x2, y2))
            and x2 > x1
            and y2 > y1
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        ):
            raise ValueError("INVALID_YOLO_DETECTION")

        if track_id is not None:
            track_id = int(track_id)

            if track_id < 0 or track_id in seen_ids:
                raise ValueError("INVALID_OR_DUPLICATE_TRACK_ID")

            seen_ids.add(track_id)

        detections.append({
            "class_id": 0,
            "class_name": "person",
            "confidence": confidence,
            "track_id": track_id,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return detections


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: person_yolo_track_worker.py "
            "MODEL_PATH TRACKER_YAML"
        )

    model_path = Path(sys.argv[1]).expanduser()
    tracker_path = Path(sys.argv[2]).expanduser()

    for path in (model_path, tracker_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    model = YOLO(str(model_path))
    worker_session = uuid.uuid4().hex

    source_session = None
    last_source_ns = 0
    last_sequence = 0

    for line in sys.stdin:
        if not line.strip():
            continue

        request_id = None

        try:
            request = json.loads(line)
            request_id = request["request_id"]

            if request.get("schema_version") != 1:
                raise ValueError("UNSUPPORTED_PROTOCOL")

            source = request["source"]

            session = str(source["source_session_id"])
            stamp_ns = int(source["source_stamp_ns"])
            sequence = int(source["source_sequence"])

            if not session or stamp_ns <= 0 or sequence < 1:
                raise ValueError("INVALID_SOURCE")

            if source_session is None:
                source_session = session
            elif session != source_session:
                raise ValueError("SOURCE_SESSION_CHANGED")

            if (
                stamp_ns <= last_source_ns
                or sequence <= last_sequence
            ):
                raise ValueError("NON_MONOTONIC_FRAME")

            confidence = float(source["confidence"])
            image_size = int(source["image_size"])

            if not (
                math.isfinite(confidence)
                and 0.0 <= confidence <= 1.0
                and image_size >= 32
            ):
                raise ValueError("INVALID_INFERENCE_PARAMETERS")

            frame = decode_image(request["jpeg_base64"])

            start = time.perf_counter()

            results = model.track(
                source=frame,
                persist=True,
                tracker=str(tracker_path),
                classes=[0],
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )

            processing_ms = (
                time.perf_counter() - start
            ) * 1000.0

            if not results:
                raise RuntimeError("MISSING_YOLO_RESULT")

            detections = extract_detections(results[0])

            last_source_ns = stamp_ns
            last_sequence = sequence

            emit({
                "schema_version": 1,
                "request_id": request_id,
                "status": "OK",
                "source_session_id": source_session,
                "source_sequence": sequence,
                "source_stamp_ns": stamp_ns,
                "source_frame_id": str(
                    source["source_frame_id"]
                ),
                "worker_session_id": worker_session,
                "processing_ms": processing_ms,
                "detection_count": len(detections),
                "detections": detections,
            })

        except Exception as exc:
            emit({
                "schema_version": 1,
                "request_id": request_id,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })


if __name__ == "__main__":
    main()
