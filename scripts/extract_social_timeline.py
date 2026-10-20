
#!/usr/bin/env python3

"""
G1 SOCIAL NAVIGATION - ROSBAG DIAGNOSTIC EXTRACTOR

Extract a chronological JSONL timeline and a diagnostic report.

Preserves:
    - rosbag reception timestamps
    - message header timestamps
    - individual TF transforms
    - LiDAR and LaserScan metadata
    - YOLO detections and social messages
    - navigation decisions, commands and dialogue
    - ROS logs and diagnostic messages

Avoids exporting:
    - PointCloud2 binary point arrays
    - Livox CustomMsg point arrays
    - camera image binary arrays

Outputs:
    social_timeline_Rxx.jsonl
    social_timeline_Rxx_diagnostics.json

The original rosbag remains the authoritative source.
"""

import argparse
import array
import json
import math
import re

from collections import Counter
from pathlib import Path

import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


NS_PER_SECOND = 1_000_000_000
NS_PER_MS = 1_000_000


# These are diagnostic gap thresholds, not requirements
# imposed on publishers.
GAP_THRESHOLDS_SEC = {
    "/scan": 0.5,
    "/scan_raw": 0.5,
    "/scan_filtered": 0.5,
    "/tf": 0.5,
    "/odom": 0.5,
    "/amcl_pose": 2.0,
    "/yolo/detections": 1.0,
    "/yolo/detections_tracked": 1.0,
    "/yolo/detections_3d": 1.0,
    "/yolo/detections_3d_base": 1.0,
}

TF_EDGE_GAP_THRESHOLD_SEC = 0.5

IMPORTANT_TOPICS = (
    "/tf",
    "/tf_static",
    "/scan",
    "/odom",
    "/amcl_pose",
    "/yolo/detections",
    "/yolo/detections_tracked",
    "/yolo/detections_3d",
    "/yolo/detections_3d_base",
    "/social_perception/tracked_persons",
    "/social_navigation/decision",
    "/explainability/speech_status",
    "/hri/human_intent",
)


def stamp_to_ns(stamp):
    return (
        int(stamp.sec) * NS_PER_SECOND
        + int(stamp.nanosec)
    )


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def simplify(value):
    """Recursively convert ROS data to valid JSON values."""

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return finite_number(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"payload_bytes": len(value)}

    if isinstance(value, array.array):
        return [simplify(v) for v in value]

    if isinstance(value, (list, tuple)):
        return [simplify(v) for v in value]

    if isinstance(value, dict):
        return {
            str(key): simplify(item)
            for key, item in value.items()
        }

    if hasattr(value, "get_fields_and_field_types"):
        return {
            name: simplify(getattr(value, name))
            for name in value.get_fields_and_field_types()
        }

    # Some ROS message fields may use NumPy arrays.
    if hasattr(value, "tolist"):
        return simplify(value.tolist())

    return str(value)


def extract_header(msg, bag_timestamp_ns):
    if not hasattr(msg, "header"):
        return {}

    header = msg.header

    header_ns = stamp_to_ns(header.stamp)

    return {
        "header_frame_id": str(header.frame_id),
        "header_stamp_ns": header_ns,
        "bag_minus_header_ms": (
            round(
                (bag_timestamp_ns - header_ns) / NS_PER_MS,
                3,
            )
            if header_ns > 0
            else None
        ),
    }


def summarize_pointcloud(msg):
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "point_count": int(msg.width * msg.height),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "data_bytes": len(msg.data),
        "is_dense": bool(msg.is_dense),
        "is_bigendian": bool(msg.is_bigendian),
        "fields": [
            {
                "name": field.name,
                "offset": int(field.offset),
                "datatype": int(field.datatype),
                "count": int(field.count),
            }
            for field in msg.fields
        ],
    }


def summarize_livox_custom(msg, bag_timestamp_ns):
    """
    Preserve Livox packet metadata without exporting points.
    """

    result = {
        "lidar_id": int(msg.lidar_id),
        "point_num": int(msg.point_num),
        "actual_points_count": len(msg.points),
        "timebase_ns": int(msg.timebase),
    }

    if msg.timebase > 0:
        result["bag_minus_timebase_ms"] = round(
            (
                bag_timestamp_ns - int(msg.timebase)
            ) / NS_PER_MS,
            3,
        )

    return result


def summarize_scan(msg):
    valid_count = 0
    minimum = None
    maximum = None

    for raw_value in msg.ranges:
        value = float(raw_value)

        if not math.isfinite(value):
            continue

        if not (msg.range_min <= value <= msg.range_max):
            continue

        valid_count += 1

        if minimum is None or value < minimum:
            minimum = value

        if maximum is None or value > maximum:
            maximum = value

    total = len(msg.ranges)

    return {
        "ranges_count": total,
        "valid_ranges_count": valid_count,
        "invalid_ranges_count": total - valid_count,
        "valid_fraction": (
            round(valid_count / total, 6)
            if total
            else None
        ),
        "min_valid_range": minimum,
        "max_valid_range": maximum,
        "angle_min": finite_number(msg.angle_min),
        "angle_max": finite_number(msg.angle_max),
        "angle_increment": finite_number(
            msg.angle_increment
        ),
        "time_increment": finite_number(
            msg.time_increment
        ),
        "scan_time": finite_number(msg.scan_time),
        "range_min": finite_number(msg.range_min),
        "range_max": finite_number(msg.range_max),
    }


def summarize_tf(msg, bag_timestamp_ns):
    transforms = []

    for transform in msg.transforms:
        tf_ns = stamp_to_ns(transform.header.stamp)

        transforms.append({
            "parent": str(transform.header.frame_id),
            "child": str(transform.child_frame_id),
            "stamp_ns": tf_ns,
            "bag_minus_tf_ms": (
                round(
                    (bag_timestamp_ns - tf_ns) / NS_PER_MS,
                    3,
                )
                if tf_ns > 0
                else None
            ),
            "translation": simplify(
                transform.transform.translation
            ),
            "rotation": simplify(
                transform.transform.rotation
            ),
        })

    return {"transforms": transforms}


def summarize_message(msg, msg_type, bag_timestamp_ns):

    if msg_type == "sensor_msgs/msg/PointCloud2":
        return summarize_pointcloud(msg)

    if msg_type == "livox_ros_driver2/msg/CustomMsg":
        return summarize_livox_custom(
            msg,
            bag_timestamp_ns,
        )

    if msg_type == "sensor_msgs/msg/LaserScan":
        return summarize_scan(msg)

    if msg_type == "tf2_msgs/msg/TFMessage":
        return summarize_tf(msg, bag_timestamp_ns)

    if msg_type == "sensor_msgs/msg/Image":
        return {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "step": int(msg.step),
            "data_bytes": len(msg.data),
        }

    if msg_type == "sensor_msgs/msg/CompressedImage":
        return {
            "format": msg.format,
            "data_bytes": len(msg.data),
        }

    return simplify(msg)


def detect_storage_id(bag_directory):
    metadata_path = bag_directory / "metadata.yaml"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing rosbag metadata: {metadata_path}"
        )

    content = metadata_path.read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"^\s*storage_identifier:\s*['\"]?([\w-]+)",
        content,
        re.MULTILINE,
    )

    if not match:
        raise ValueError(
            "Cannot determine rosbag storage identifier."
        )

    return match.group(1)


def output_paths(bag_directory, output_arg):
    if output_arg:
        timeline = Path(output_arg).expanduser().resolve()
    else:
        match = re.search(
            r"_R(\d+)$",
            bag_directory.parent.name,
            re.IGNORECASE,
        )

        suffix = (
            f"_R{match.group(1)}"
            if match
            else ""
        )

        timeline = (
            bag_directory
            / f"social_timeline{suffix}.jsonl"
        )

    diagnostics = timeline.with_name(
        timeline.stem + "_diagnostics.json"
    )

    return timeline, diagnostics


class IntervalStats:
    """
    Track reception and acquisition intervals.

    No synthetic interpolation is performed.
    """

    def __init__(self, threshold_sec=None):
        self.previous_bag_ns = None
        self.previous_header_ns = None

        self.count = 0
        self.gaps = []
        self.large_gaps = []
        self.header_regressions = 0

        self.threshold_sec = threshold_sec

    def update(self, bag_ns, header_ns=None):
        self.count += 1

        if self.previous_bag_ns is not None:
            gap = (
                bag_ns - self.previous_bag_ns
            ) / NS_PER_SECOND

            if gap >= 0:
                self.gaps.append(gap)

                if (
                    self.threshold_sec is not None
                    and gap > self.threshold_sec
                ):
                    self.large_gaps.append({
                        "at_bag_ns": int(bag_ns),
                        "gap_sec": round(gap, 6),
                    })

        self.previous_bag_ns = bag_ns

        if header_ns is not None and header_ns > 0:
            if (
                self.previous_header_ns is not None
                and header_ns < self.previous_header_ns
            ):
                self.header_regressions += 1

            self.previous_header_ns = header_ns

    def report(self):
        if not self.gaps:
            return {
                "messages": self.count,
                "intervals": 0,
                "header_regressions": self.header_regressions,
                "large_gaps": self.large_gaps,
            }

        sorted_gaps = sorted(self.gaps)

        def percentile(p):
            position = round(
                (len(sorted_gaps) - 1) * p
            )
            return sorted_gaps[position]

        return {
            "messages": self.count,
            "intervals": len(self.gaps),
            "mean_interval_sec": round(
                sum(self.gaps) / len(self.gaps),
                6,
            ),
            "median_interval_sec": round(
                percentile(0.5), 6
            ),
            "p95_interval_sec": round(
                percentile(0.95), 6
            ),
            "max_interval_sec": round(
                sorted_gaps[-1], 6
            ),
            "header_regressions": self.header_regressions,
            "large_gap_count": len(self.large_gaps),
            "large_gaps": self.large_gaps,
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bag",
        help="ROS 2 bag directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional JSONL output path",
    )

    args = parser.parse_args()

    bag_directory = Path(
        args.bag
    ).expanduser().resolve()

    if not bag_directory.is_dir():
        parser.error(
            f"Bag directory not found: {bag_directory}"
        )

    storage_id = detect_storage_id(bag_directory)

    timeline_path, diagnostics_path = output_paths(
        bag_directory,
        args.output,
    )

    timeline_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_directory),
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }

    type_cache = {}
    unavailable_types = {}
    decode_errors = Counter()
    error_examples = {}

    counts = Counter()
    topic_stats = {}
    tf_edge_stats = {}

    first_timestamp_ns = None
    last_timestamp_ns = None

    with timeline_path.open(
        "w",
        encoding="utf-8",
    ) as output:

        while reader.has_next():

            topic, data, bag_ns = reader.read_next()

            # Include even messages whose type cannot be decoded
            # when establishing the recording time span.
            if first_timestamp_ns is None:
                first_timestamp_ns = bag_ns

            last_timestamp_ns = bag_ns

            msg_type = topic_types.get(topic)

            if msg_type is None:
                continue

            if msg_type not in type_cache:
                try:
                    type_cache[msg_type] = get_message(
                        msg_type
                    )
                except Exception as exc:
                    type_cache[msg_type] = None
                    unavailable_types[msg_type] = str(exc)

            message_class = type_cache[msg_type]

            if message_class is None:
                continue

            try:
                msg = deserialize_message(
                    data,
                    message_class,
                )

                record = {
                    "t": round(
                        (
                            bag_ns - first_timestamp_ns
                        ) / NS_PER_SECOND,
                        6,
                    ),
                    "timestamp_ns": int(bag_ns),
                    "topic": topic,
                    "type": msg_type,
                }

                header_info = extract_header(
                    msg,
                    bag_ns,
                )

                record.update(header_info)

                record["msg"] = summarize_message(
                    msg,
                    msg_type,
                    bag_ns,
                )

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                counts[topic] += 1

                if topic not in topic_stats:
                    topic_stats[topic] = IntervalStats(
                        GAP_THRESHOLDS_SEC.get(topic)
                    )

                topic_stats[topic].update(
                    bag_ns,
                    header_info.get("header_stamp_ns"),
                )

                # Track each dynamic TF link independently.
                if (
                    topic == "/tf"
                    and msg_type == "tf2_msgs/msg/TFMessage"
                ):
                    for tf in msg.transforms:

                        parent = tf.header.frame_id
                        child = tf.child_frame_id
                        edge = f"{parent} -> {child}"

                        if edge not in tf_edge_stats:
                            tf_edge_stats[edge] = IntervalStats(
                                TF_EDGE_GAP_THRESHOLD_SEC
                            )

                        tf_edge_stats[edge].update(
                            bag_ns,
                            stamp_to_ns(tf.header.stamp),
                        )

            except Exception as exc:
                decode_errors[topic] += 1

                if topic not in error_examples:
                    error_examples[topic] = str(exc)

    missing_important = [
        topic
        for topic in IMPORTANT_TOPICS
        if counts[topic] == 0
    ]

    diagnostic_report = {
        "bag": str(bag_directory),
        "storage_id": storage_id,
        "timeline": str(timeline_path),
        "first_bag_timestamp_ns": first_timestamp_ns,
        "last_bag_timestamp_ns": last_timestamp_ns,
        "total_extracted_messages": sum(counts.values()),
        "message_counts": dict(counts),
        "topics_present_in_metadata": topic_types,
        "important_topics_missing": missing_important,
        "topic_intervals": {
            topic: stats.report()
            for topic, stats in sorted(topic_stats.items())
        },
        "tf_edge_intervals": {
            edge: stats.report()
            for edge, stats in sorted(tf_edge_stats.items())
        },
        "unavailable_message_types": unavailable_types,
        "decode_errors": dict(decode_errors),
        "decode_error_examples": error_examples,
    }

    with diagnostics_path.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            diagnostic_report,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print()
    print("=" * 64)
    print("G1 SOCIAL NAVIGATION - EXTRACTION REPORT")
    print("=" * 64)

    print(f"Bag             : {bag_directory}")
    print(f"Storage         : {storage_id}")
    print(f"Messages        : {sum(counts.values())}")
    print(f"Timeline        : {timeline_path}")
    print(f"Diagnostics     : {diagnostics_path}")

    print("\nCritical topic coverage:")

    for topic in IMPORTANT_TOPICS:
        count = counts[topic]
        state = "OK" if count else "ABSENT"

        print(f"  {state:6} {topic}: {count}")

    print("\nMaximum reception intervals:")

    for topic in (
        "/scan",
        "/scan_raw",
        "/scan_filtered",
        "/odom",
        "/amcl_pose",
        "/tf",
        "/yolo/detections",
        "/yolo/detections_3d",
    ):
        stats = topic_stats.get(topic)

        if stats and stats.gaps:
            print(
                f"  {topic}: "
                f"{max(stats.gaps):.3f} s"
            )

    print("\nTF links found:")

    for edge in sorted(tf_edge_stats):
        stats = tf_edge_stats[edge]

        max_gap = (
            max(stats.gaps)
            if stats.gaps
            else 0.0
        )

        print(
            f"  {edge}: "
            f"{stats.count} messages, "
            f"max gap {max_gap:.3f} s"
        )

    if unavailable_types:
        print("\nUnavailable ROS message types:")
        for name, error in unavailable_types.items():
            print(f"  {name}: {error}")

    if decode_errors:
        print("\nDecode errors:")
        for topic, count in decode_errors.items():
            print(
                f"  {topic}: {count} "
                f"({error_examples.get(topic, '')})"
            )

    print()
    print("Extraction completed.")


if __name__ == "__main__":
    main()


#python3 ./scripts/extract_social_timeline.py /home/jean/g1_ws/experiments/restaurant_social_awareness/technical/D/S1a/technical_D_S1a_R05/navigation