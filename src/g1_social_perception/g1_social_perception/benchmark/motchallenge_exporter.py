
#!/usr/bin/env python3
"""
Export G1 identity benchmark records to MOTChallenge format.

Supports:
    M1, M2, M3, M4, botsort, botsort_reid.

Requires fully annotated frames and complete coverage
from every evaluated tracker.

No ground-truth identities are inferred.
No detector/tracker output is silently treated as truth.
No automatic interpolation or timestamp tolerance.

MOT identifiers are local integer mappings of:
    ground truth: person ID
    predictions: tracking session + track ID

Unassigned detections receive unique, nonpersistent IDs.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


METHODS = (
    "M1", "M2", "M3", "M4",
    "botsort", "botsort_reid",
)


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"INVALID_JSON:{path}:{number}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError("EXPECTED_JSON_OBJECT")

            rows.append(row)

    return rows


def key_of(row):
    stamp = int(row["source_stamp_ns"])
    frame = str(row["source_frame_id"])

    if stamp <= 0 or not frame:
        raise ValueError("INVALID_FRAME_KEY")

    return stamp, frame


def validate_bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("INVALID_BBOX")

    x1, y1, x2, y2 = map(float, value)

    if not all(
        math.isfinite(v)
        for v in (x1, y1, x2, y2)
    ):
        raise ValueError("NON_FINITE_BBOX")

    if x2 <= x1 or y2 <= y1:
        raise ValueError("INVALID_BBOX_DIMENSIONS")

    return x1, y1, x2 - x1, y2 - y1


def mot_line(
    frame_number,
    identity,
    bbox,
    confidence,
    *,
    ground_truth=False,
):
    x, y, w, h = validate_bbox(bbox)

    if ground_truth:
        # MOT ground truth:
        # conf=1, class=1 (pedestrian), visibility=1.
        fields = (
            frame_number, identity,
            x, y, w, h,
            1, 1, 1, -1,
        )
    else:
        score = float(confidence)

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("INVALID_CONFIDENCE")

        fields = (
            frame_number, identity,
            x, y, w, h,
            score, -1, -1, -1,
        )

    return ",".join(
        str(value) for value in fields
    ) + "\n"


def load_ground_truth(path):
    frames = {}

    for row in read_jsonl(path):
        key = key_of(row)

        if key in frames:
            raise ValueError("DUPLICATE_GROUND_TRUTH_FRAME")

        people = row["persons"]

        if not isinstance(people, list):
            raise ValueError("INVALID_PERSON_LIST")

        seen = set()

        for person in people:
            identity = str(person["id"])

            if not identity or identity in seen:
                raise ValueError("DUPLICATE_GROUND_TRUTH_ID")

            seen.add(identity)
            validate_bbox(person["bbox_xyxy"])

        frames[key] = people

    if not frames:
        raise ValueError("EMPTY_GROUND_TRUTH")

    return frames


def load_predictions(path):
    predictions = {
        method: {} for method in METHODS
    }

    for row in read_jsonl(path):
        if row.get("record_type") != "observation":
            continue

        method = row.get("method")

        if method not in predictions:
            continue

        key = key_of(row)

        if key in predictions[method]:
            raise ValueError(
                f"DUPLICATE_METHOD_FRAME:{method}:{key}"
            )

        if not row.get("tracking_session_id"):
            raise ValueError("MISSING_TRACKING_SESSION")

        if not isinstance(row.get("detections"), list):
            raise ValueError("INVALID_DETECTION_LIST")

        for detection in row["detections"]:
            validate_bbox(detection["bbox_xyxy"])

        predictions[method][key] = row

    return predictions


def check_coverage(ground_truth, predictions):
    expected = set(ground_truth)

    for method in METHODS:
        actual = set(predictions[method])

        if actual != expected:
            raise ValueError(
                f"FRAME_COVERAGE_MISMATCH:{method}:"
                f"missing={len(expected - actual)}:"
                f"unexpected={len(actual - expected)}"
            )

    return sorted(expected)


def write_text(path, text):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export G1 tracking results to MOTChallenge."
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
        "--sequence-name",
        default="g1_sequence_01",
    )

    parser.add_argument(
        "--image-width",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--frame-rate",
        type=int,
        required=True,
        help="Actual or nominal source frame rate.",
    )

    args = parser.parse_args()

    if (
        not args.sequence_name
        or not all(
            ch.isalnum() or ch in "_-"
            for ch in args.sequence_name
        )
    ):
        parser.error("INVALID_SEQUENCE_NAME")

    if (
        args.image_width <= 0
        or args.image_height <= 0
        or args.frame_rate <= 0
    ):
        parser.error("INVALID_IMAGE_METADATA")

    gt = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.observations)

    frames = check_coverage(gt, predictions)

    output = args.output_directory.expanduser()
    gt_root = output / "gt"
    trackers_root = output / "trackers"

    benchmark_split = "G1MOT-train"
    seq = args.sequence_name

    gt_directory = (
        gt_root / benchmark_split / seq
    )

    gt_lines = []
    gt_id_map = {}

    for frame_number, key in enumerate(frames, 1):
        for person in gt[key]:
            identity = str(person["id"])

            if identity not in gt_id_map:
                gt_id_map[identity] = len(gt_id_map) + 1

            gt_lines.append(
                mot_line(
                    frame_number,
                    gt_id_map[identity],
                    person["bbox_xyxy"],
                    1,
                    ground_truth=True,
                )
            )

    write_text(
        gt_directory / "gt" / "gt.txt",
        "".join(gt_lines),
    )

    seqinfo = (
        "[Sequence]\n"
        f"name={seq}\n"
        "imDir=img1\n"
        f"frameRate={args.frame_rate}\n"
        f"seqLength={len(frames)}\n"
        f"imWidth={args.image_width}\n"
        f"imHeight={args.image_height}\n"
        "imExt=.jpg\n"
    )

    write_text(
        gt_directory / "seqinfo.ini",
        seqinfo,
    )

    write_text(
        gt_root / "seqmaps" / f"{benchmark_split}.txt",
        f"name\n{seq}\n",
    )

    summary = {}

    for method in METHODS:
        identity_map = {}
        next_identity = 1
        lines = []
        unassigned_count = 0

        for frame_number, key in enumerate(frames, 1):
            row = predictions[method][key]

            session = str(
                row["tracking_session_id"]
            )

            seen = set()

            for detection_index, detection in enumerate(
                row["detections"]
            ):
                track_id = detection.get("track_id")

                if track_id is None or str(track_id) == "":
                    # Preserve an unassigned prediction as a
                    # single-frame identity, without pretending
                    # it belongs to a persistent track.
                    identity_key = (
                        session,
                        "__unassigned__",
                        frame_number,
                        detection_index,
                    )
                    unassigned_count += 1
                else:
                    identity_key = (
                        session,
                        str(track_id),
                    )

                if identity_key in seen:
                    raise ValueError(
                        f"DUPLICATE_TRACK_ID:{method}:"
                        f"frame={frame_number}"
                    )

                seen.add(identity_key)

                if identity_key not in identity_map:
                    identity_map[identity_key] = next_identity
                    next_identity += 1

                lines.append(
                    mot_line(
                        frame_number,
                        identity_map[identity_key],
                        detection["bbox_xyxy"],
                        detection["confidence"],
                    )
                )

        destination = (
            trackers_root
            / benchmark_split
            / method
            / "data"
            / f"{seq}.txt"
        )

        write_text(
            destination,
            "".join(lines),
        )

        summary[method] = {
            "detections": len(lines),
            "unique_export_ids": len(identity_map),
            "unassigned_detections": unassigned_count,
        }

    manifest = {
        "schema_version": 1,
        "benchmark": "G1MOT",
        "split": "train",
        "sequence": seq,
        "frames": len(frames),
        "ground_truth_objects": len(gt_lines),
        "methods": summary,
        "frame_mapping": [
            {
                "mot_frame": number,
                "source_stamp_ns": key[0],
                "source_frame_id": key[1],
            }
            for number, key in enumerate(frames, 1)
        ],
    }

    write_text(
        output / "manifest.json",
        json.dumps(
            manifest,
            indent=2,
            allow_nan=False,
        ) + "\n",
    )

    print("MOT_EXPORT_COMPLETE")
    print(f"FRAMES={len(frames)}")
    print(f"GROUND_TRUTH_OBJECTS={len(gt_lines)}")
    print(f"METHODS={len(METHODS)}")
    print(f"GT_ROOT={gt_root}")
    print(f"TRACKERS_ROOT={trackers_root}")


if __name__ == "__main__":
    main()
