
#!/usr/bin/env python3
"""
Offline G1 tracking identity benchmark evaluator.

Compares:
    M1, M2, M3, M4, botsort, botsort_reid

Input:
    identity_observations.jsonl from the G1 recorder
    ground_truth.jsonl annotated against original frames

Output:
    evaluation.json
    evaluation.csv

Metrics:
    IDF1, IDP, IDR, MOTA, IDSW, FP, FN

HOTA must be computed separately using TrackEval.

No ROS dependency. Run in the Conda evaluation environment.

Strict policy:
    - Every method must cover every annotated frame.
    - No timestamp approximation.
    - No automatic ground-truth identity inference.
    - No mixing of identity domains across sessions.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import motmetrics as mm


METHODS = (
    "M1", "M2", "M3", "M4",
    "botsort", "botsort_reid",
)

METRICS = (
    "idf1",
    "idp",
    "idr",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_objects",
    "num_frames",
)


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: INVALID_JSON"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: EXPECTED_OBJECT"
                )

            records.append(record)

    return records


def check_bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("INVALID_BBOX")

    box = [float(x) for x in value]

    if not all(math.isfinite(x) for x in box):
        raise ValueError("NON_FINITE_BBOX")

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        raise ValueError("INVALID_BBOX_DIMENSIONS")

    return box


def to_xywh(box):
    x1, y1, x2, y2 = check_bbox(box)
    return [x1, y1, x2 - x1, y2 - y1]


def frame_key(record):
    stamp = int(record["source_stamp_ns"])
    frame = str(record["source_frame_id"])

    if stamp <= 0 or not frame:
        raise ValueError("INVALID_FRAME_KEY")

    return stamp, frame


def load_ground_truth(path):
    frames = {}

    for record in load_jsonl(path):
        key = frame_key(record)

        if key in frames:
            raise ValueError(
                f"DUPLICATE_GROUND_TRUTH_FRAME:{key}"
            )

        persons = record["persons"]

        if not isinstance(persons, list):
            raise ValueError("INVALID_GROUND_TRUTH_PERSONS")

        seen_ids = set()

        for person in persons:
            person_id = str(person["id"])

            if not person_id or person_id in seen_ids:
                raise ValueError(
                    "INVALID_OR_DUPLICATE_GROUND_TRUTH_ID"
                )

            seen_ids.add(person_id)
            check_bbox(person["bbox_xyxy"])

        frames[key] = persons

    if not frames:
        raise ValueError("EMPTY_GROUND_TRUTH")

    return frames


def load_predictions(path):
    grouped = {
        method: {} for method in METHODS
    }

    for record in load_jsonl(path):
        method = record.get("method")

        if method not in grouped:
            continue

        if record.get("record_type") != "observation":
            continue

        key = frame_key(record)

        if key in grouped[method]:
            raise ValueError(
                f"DUPLICATE_PREDICTION:{method}:{key}"
            )

        if not record.get("tracking_session_id"):
            raise ValueError("MISSING_TRACKING_SESSION")

        for detection in record["detections"]:
            check_bbox(detection["bbox_xyxy"])

        grouped[method][key] = record

    return grouped


def validate_coverage(ground_truth, predictions):
    """
    Reject incomplete or unaligned experiments.

    Missing output is not silently treated as an empty
    detection frame.
    """
    expected = set(ground_truth)

    for method in METHODS:
        actual = set(predictions[method])

        missing = expected - actual
        unexpected = actual - expected

        if missing or unexpected:
            raise ValueError(
                f"INCOMPLETE_OR_UNALIGNED_METHOD:{method} "
                f"missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )

    # Every method must have observed the same frame set.
    return sorted(expected)


def evaluate_method(
    method,
    frames,
    ground_truth,
    predictions,
    *,
    iou_threshold,
):
    accumulator = mm.MOTAccumulator(auto_id=False)

    for frame_index, key in enumerate(frames, 1):
        gt = ground_truth[key]
        observation = predictions[method][key]

        gt_ids = [
            str(person["id"])
            for person in gt
        ]

        gt_boxes = np.array(
            [
                to_xywh(person["bbox_xyxy"])
                for person in gt
            ],
            dtype=float,
        ).reshape((-1, 4))

        dt_ids = []
        dt_boxes = []
        seen_prediction_ids = set()

        session = observation["tracking_session_id"]

        for detection_index, detection in enumerate(
            observation["detections"]
        ):
            track_id = detection.get("track_id")

            if track_id is None or str(track_id) == "":
                # An unassigned detection still counts as a
                # prediction and cannot gain stable identity
                # by borrowing another track ID.
                identity = (
                    f"{method}:{session}:unassigned:"
                    f"{frame_index}:{detection_index}"
                )
            else:
                identity = (
                    f"{method}:{session}:"
                    f"{str(track_id)}"
                )

            if identity in seen_prediction_ids:
                raise ValueError(
                    f"DUPLICATE_TRACK_ID_IN_FRAME:{method}"
                )

            seen_prediction_ids.add(identity)
            dt_ids.append(identity)

            dt_boxes.append(
                to_xywh(detection["bbox_xyxy"])
            )

        dt_boxes = np.array(
            dt_boxes,
            dtype=float,
        ).reshape((-1, 4))

        # motmetrics expects a distance matrix.
        # 1 - IoU is used, and pairs below the required
        # overlap are made inadmissible.
        distances = mm.distances.iou_matrix(
            gt_boxes,
            dt_boxes,
            max_iou=1.0 - iou_threshold,
        )

        accumulator.update(
            gt_ids,
            dt_ids,
            distances,
            frameid=frame_index,
        )

    handler = mm.metrics.create()

    summary = handler.compute(
        accumulator,
        metrics=list(METRICS),
        name=method,
    )

    result = {
        "method": method,
        "iou_threshold": iou_threshold,
    }

    for metric in METRICS:
        value = summary.loc[method, metric]

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float):
            result[metric] = (
                value if math.isfinite(value) else None
            )
        else:
            result[metric] = int(value)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate G1 tracking baselines."
    )

    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("INVALID_IOU_THRESHOLD")

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    predictions = load_predictions(
        args.observations
    )

    frames = validate_coverage(
        ground_truth,
        predictions,
    )

    results = []

    for method in METHODS:
        results.append(
            evaluate_method(
                method,
                frames,
                ground_truth,
                predictions,
                iou_threshold=args.iou_threshold,
            )
        )

    output = args.output_directory.expanduser()
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"

    with json_path.open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "frame_count": len(frames),
                "metrics_library": "motmetrics",
                "hota_computed": False,
                "results": results,
            },
            handle,
            indent=2,
            allow_nan=False,
        )

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"EVALUATED_FRAMES={len(frames)}")
    print(f"EVALUATED_METHODS={len(results)}")
    print(f"JSON={json_path}")
    print(f"CSV={csv_path}")
    print("HOTA=NOT_COMPUTED")


if __name__ == "__main__":
    main()
