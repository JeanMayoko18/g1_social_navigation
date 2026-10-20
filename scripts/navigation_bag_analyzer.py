#!/usr/bin/env python3

"""Analyze one G1 Nav2 rosbag2 trial without modifying the bag.

Compatible with ROS 2 Humble rosbag2 SQLite bags. The movement window is
detected from the final robot command (/cmd_vel_rlsar), falling back to
/cmd_vel when necessary.
"""

import argparse
import bisect
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


ODOM_TOPICS = (
    "/odom",
    "/odom_dog_calibrated",
    "/dog_odom",
    "/dog_odom_2d",
)

COMMAND_TOPICS = (
    "/cmd_vel",
    "/cmd_vel_nav_scaled",
    "/cmd_vel_mux",
    "/cmd_vel_rlsar",
)

PLAN_TOPIC = "/plan"
AMCL_TOPIC = "/amcl_pose"
ROSOUT_TOPIC = "/rosout"


@dataclass
class VectorSample:
    time_sec: float
    vx: float
    vy: float
    wz: float


@dataclass
class OdomSample:
    time_sec: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


@dataclass
class AmclSample:
    time_sec: float
    x: float
    y: float
    yaw: float
    sigma_x: float
    sigma_y: float
    sigma_xy: float
    sigma_yaw: float


@dataclass
class LogEvent:
    time_sec: float
    node_name: str
    level: int
    message: str


def quaternion_to_yaw(quaternion) -> float:
    """Convert a geometry_msgs quaternion to planar yaw."""

    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle), math.cos(angle))


def safe_sqrt(value: float) -> float:
    return math.sqrt(max(0.0, float(value)))


def average(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def root_mean_square(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(mean([value * value for value in values]))


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = probability * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_bag(bag_path: Path):
    """Read only the topics required by the navigation analysis."""

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }

    requested_topics = set(ODOM_TOPICS)
    requested_topics.update(COMMAND_TOPICS)
    requested_topics.update({PLAN_TOPIC, AMCL_TOPIC, ROSOUT_TOPIC})

    message_classes = {
        topic: get_message(topic_types[topic])
        for topic in requested_topics
        if topic in topic_types and topic != PLAN_TOPIC
    }

    commands: Dict[str, List[VectorSample]] = {
        topic: [] for topic in COMMAND_TOPICS
    }
    odometry: Dict[str, List[OdomSample]] = {
        topic: [] for topic in ODOM_TOPICS
    }
    amcl: List[AmclSample] = []
    plan_times: List[float] = []
    logs: List[LogEvent] = []
    first_time: Optional[float] = None
    last_time: Optional[float] = None

    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        time_sec = timestamp_ns * 1.0e-9

        if first_time is None:
            first_time = time_sec
        last_time = time_sec

        if topic not in requested_topics:
            continue

        if topic == PLAN_TOPIC:
            plan_times.append(time_sec)
            continue

        message_class = message_classes.get(topic)
        if message_class is None:
            continue

        message = deserialize_message(serialized, message_class)

        if topic in COMMAND_TOPICS:
            commands[topic].append(
                VectorSample(
                    time_sec=time_sec,
                    vx=float(message.linear.x),
                    vy=float(message.linear.y),
                    wz=float(message.angular.z),
                )
            )
            continue

        if topic in ODOM_TOPICS:
            pose = message.pose.pose
            twist = message.twist.twist
            odometry[topic].append(
                OdomSample(
                    time_sec=time_sec,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=quaternion_to_yaw(pose.orientation),
                    vx=float(twist.linear.x),
                    vy=float(twist.linear.y),
                    wz=float(twist.angular.z),
                )
            )
            continue

        if topic == AMCL_TOPIC:
            pose = message.pose.pose
            covariance = message.pose.covariance
            sigma_x = safe_sqrt(covariance[0])
            sigma_y = safe_sqrt(covariance[7])
            amcl.append(
                AmclSample(
                    time_sec=time_sec,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=quaternion_to_yaw(pose.orientation),
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    sigma_xy=math.sqrt(sigma_x * sigma_x + sigma_y * sigma_y),
                    sigma_yaw=safe_sqrt(covariance[35]),
                )
            )
            continue

        if topic == ROSOUT_TOPIC:
            logs.append(
                LogEvent(
                    time_sec=time_sec,
                    node_name=str(message.name),
                    level=int(message.level),
                    message=str(message.msg),
                )
            )

    return {
        "topic_types": topic_types,
        "commands": commands,
        "odometry": odometry,
        "amcl": amcl,
        "plan_times": plan_times,
        "logs": logs,
        "bag_start_sec": first_time,
        "bag_end_sec": last_time,
    }


def is_moving(
    sample: VectorSample,
    linear_threshold: float,
    angular_threshold: float,
) -> bool:
    return (
        math.hypot(sample.vx, sample.vy) >= linear_threshold
        or abs(sample.wz) >= angular_threshold
    )


def movement_window(
    commands: Dict[str, List[VectorSample]],
    linear_threshold: float,
    angular_threshold: float,
) -> Tuple[str, float, float]:
    """Determine movement bounds from the final command to the robot."""

    command_topic = (
        "/cmd_vel_rlsar"
        if commands.get("/cmd_vel_rlsar")
        else "/cmd_vel"
    )
    samples = commands.get(command_topic, [])
    moving_samples = [
        sample
        for sample in samples
        if is_moving(sample, linear_threshold, angular_threshold)
    ]

    if not moving_samples:
        raise RuntimeError(
            f"No movement detected on {command_topic}."
        )

    return (
        command_topic,
        moving_samples[0].time_sec,
        moving_samples[-1].time_sec,
    )


def select_window(samples: Iterable, start: float, end: float) -> List:
    return [
        sample
        for sample in samples
        if start <= sample.time_sec <= end
    ]


def vector_statistics(samples: Sequence) -> Dict:
    """Calculate signed and absolute statistics for vx, vy and wz."""

    result = {"sample_count": len(samples)}

    for axis in ("vx", "vy", "wz"):
        values = [float(getattr(sample, axis)) for sample in samples]
        absolute = [abs(value) for value in values]
        result[axis] = {
            "mean": average(values),
            "mean_absolute": average(absolute),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "maximum_absolute": max(absolute) if absolute else None,
            "rms": root_mean_square(values),
        }

    return result


def count_sign_changes(values: Sequence[float], threshold: float) -> int:
    """Count changes between significant positive and negative values."""

    previous_sign = 0
    changes = 0

    for value in values:
        if value > threshold:
            current_sign = 1
        elif value < -threshold:
            current_sign = -1
        else:
            continue

        if previous_sign and current_sign != previous_sign:
            changes += 1
        previous_sign = current_sign

    return changes


def sign_change_statistics(
    samples: Sequence,
    linear_threshold: float,
    angular_threshold: float,
) -> Dict[str, int]:
    return {
        "vx": count_sign_changes(
            [sample.vx for sample in samples],
            linear_threshold,
        ),
        "vy": count_sign_changes(
            [sample.vy for sample in samples],
            linear_threshold,
        ),
        "wz": count_sign_changes(
            [sample.wz for sample in samples],
            angular_threshold,
        ),
    }


def command_time_statistics(
    samples: Sequence[VectorSample],
    start: float,
    end: float,
    linear_threshold: float,
    angular_threshold: float,
) -> Dict:
    """Integrate stopped and rotation durations inside the movement window."""

    selected = select_window(samples, start, end)
    if len(selected) < 2:
        return {
            "window_duration_sec": max(0.0, end - start),
            "observed_duration_sec": 0.0,
            "stopped_time_sec": None,
            "rotation_time_sec": None,
            "in_place_rotation_time_sec": None,
        }

    periods = [
        second.time_sec - first.time_sec
        for first, second in zip(selected, selected[1:])
        if 0.0 < second.time_sec - first.time_sec <= 1.0
    ]
    normal_period = median(periods) if periods else 0.05
    maximum_period = max(0.20, normal_period * 5.0)

    observed = 0.0
    stopped = 0.0
    rotating = 0.0
    in_place = 0.0

    for current, following in zip(selected, selected[1:]):
        delta = following.time_sec - current.time_sec
        if delta <= 0.0 or delta > maximum_period:
            continue

        observed += delta
        linear_speed = math.hypot(current.vx, current.vy)
        angular_speed = abs(current.wz)

        if (
            linear_speed < linear_threshold
            and angular_speed < angular_threshold
        ):
            stopped += delta

        if angular_speed >= angular_threshold:
            rotating += delta
            if linear_speed < linear_threshold:
                in_place += delta

    return {
        "window_duration_sec": end - start,
        "observed_duration_sec": observed,
        "stopped_time_sec": stopped,
        "stopped_ratio": stopped / observed if observed else None,
        "rotation_time_sec": rotating,
        "rotation_ratio": rotating / observed if observed else None,
        "in_place_rotation_time_sec": in_place,
    }


def odometry_statistics(samples: Sequence[OdomSample]) -> Dict:
    if not samples:
        return {"sample_count": 0}

    path_length = 0.0
    for first, second in zip(samples, samples[1:]):
        path_length += math.hypot(
            second.x - first.x,
            second.y - first.y,
        )

    first = samples[0]
    last = samples[-1]
    displacement = math.hypot(last.x - first.x, last.y - first.y)

    return {
        "sample_count": len(samples),
        "duration_sec": last.time_sec - first.time_sec,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "start": {"x": first.x, "y": first.y, "yaw_rad": first.yaw},
        "end": {"x": last.x, "y": last.y, "yaw_rad": last.yaw},
        "heading_change_rad": wrap_angle(last.yaw - first.yaw),
        "twist": vector_statistics(samples),
    }


def nearest_pairs(reference: Sequence, candidate: Sequence, tolerance: float):
    """Pair samples by nearest bag timestamp."""

    if not reference or not candidate:
        return []

    candidate_times = [sample.time_sec for sample in candidate]
    pairs = []

    for reference_sample in reference:
        index = bisect.bisect_left(candidate_times, reference_sample.time_sec)
        choices = []
        if index < len(candidate):
            choices.append(candidate[index])
        if index > 0:
            choices.append(candidate[index - 1])

        if not choices:
            continue

        closest = min(
            choices,
            key=lambda sample: abs(sample.time_sec - reference_sample.time_sec),
        )
        delta = abs(closest.time_sec - reference_sample.time_sec)
        if delta <= tolerance:
            pairs.append((reference_sample, closest, delta))

    return pairs


def odometry_difference(reference: Sequence[OdomSample], candidate: Sequence[OdomSample]):
    pairs = nearest_pairs(reference, candidate, tolerance=0.03)
    if not pairs:
        return {"matched_samples": 0}

    position_errors = [
        math.hypot(first.x - second.x, first.y - second.y)
        for first, second, _ in pairs
    ]
    yaw_errors = [
        abs(wrap_angle(first.yaw - second.yaw))
        for first, second, _ in pairs
    ]

    result = {
        "matched_samples": len(pairs),
        "timestamp_difference_mean_sec": average(
            [delta for _, _, delta in pairs]
        ),
        "position_error_mean_m": average(position_errors),
        "position_error_rmse_m": root_mean_square(position_errors),
        "position_error_max_m": max(position_errors),
        "yaw_error_mean_rad": average(yaw_errors),
        "yaw_error_max_rad": max(yaw_errors),
    }

    for axis in ("vx", "vy", "wz"):
        errors = [
            getattr(first, axis) - getattr(second, axis)
            for first, second, _ in pairs
        ]
        result[f"{axis}_difference_mean"] = average(errors)
        result[f"{axis}_difference_rmse"] = root_mean_square(errors)
        result[f"{axis}_difference_max_absolute"] = max(
            abs(error) for error in errors
        )

    return result


def command_difference(reference: Sequence[VectorSample], candidate: Sequence[VectorSample]):
    pairs = nearest_pairs(reference, candidate, tolerance=0.10)
    result = {"matched_samples": len(pairs)}

    if not pairs:
        return result

    result["timestamp_difference_mean_sec"] = average(
        [delta for _, _, delta in pairs]
    )

    for axis in ("vx", "vy", "wz"):
        errors = [
            getattr(first, axis) - getattr(second, axis)
            for first, second, _ in pairs
        ]
        result[axis] = {
            "mean_error": average(errors),
            "mae": average([abs(error) for error in errors]),
            "rmse": root_mean_square(errors),
            "maximum_absolute_error": max(abs(error) for error in errors),
        }

    return result


def amcl_statistics(samples: Sequence[AmclSample]) -> Dict:
    if not samples:
        return {"sample_count": 0}

    result = {"sample_count": len(samples)}

    for name in ("sigma_x", "sigma_y", "sigma_xy", "sigma_yaw"):
        values = [getattr(sample, name) for sample in samples]
        result[name] = {
            "mean": average(values),
            "median": median(values),
            "p95": percentile(values, 0.95),
            "maximum": max(values),
        }

    return result


def navigation_log_events(logs: Sequence[LogEvent], start: float, end: float):
    relevant_nodes = re.compile(
        r"controller|planner|bt_navigator|behavior|recover|navigate|waypoint|amcl",
        re.IGNORECASE,
    )
    relevant_messages = re.compile(
        r"succeed|success|reached|abort|failed|failure|cancel|recover|goal",
        re.IGNORECASE,
    )
    success_pattern = re.compile(
        r"succeed|success|goal reached|reached the goal",
        re.IGNORECASE,
    )
    abort_pattern = re.compile(
        r"abort|failed|failure",
        re.IGNORECASE,
    )

    events = []
    terminal_candidates = []

    for event in logs:
        if not start <= event.time_sec <= end:
            continue
        if not (
            relevant_nodes.search(event.node_name)
            or relevant_messages.search(event.message)
        ):
            continue
        if not relevant_messages.search(event.message):
            continue

        classification = "other"
        if success_pattern.search(event.message):
            classification = "success"
            terminal_candidates.append((event.time_sec, classification))
        elif abort_pattern.search(event.message):
            classification = "abort_or_failure"
            terminal_candidates.append((event.time_sec, classification))

        events.append(
            {
                "time_from_movement_start_sec": event.time_sec - start,
                "node": event.node_name,
                "level": event.level,
                "classification": classification,
                "message": event.message,
            }
        )

    terminal_status = (
        terminal_candidates[-1][1]
        if terminal_candidates
        else "unknown_missing_action_status"
    )

    return terminal_status, events


def write_trajectory_csv(
    output_path: Path,
    odometry: Dict[str, List[OdomSample]],
    start: float,
    end: float,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["topic", "time_from_start_sec", "x", "y", "yaw_rad", "vx", "vy", "wz"]
        )
        for topic in ODOM_TOPICS:
            for sample in select_window(odometry.get(topic, []), start, end):
                writer.writerow(
                    [
                        topic,
                        sample.time_sec - start,
                        sample.x,
                        sample.y,
                        sample.yaw,
                        sample.vx,
                        sample.vy,
                        sample.wz,
                    ]
                )


def analyze(arguments) -> Dict:
    data = read_bag(arguments.bag)
    command_topic, movement_start, movement_end = movement_window(
        data["commands"],
        arguments.linear_threshold,
        arguments.angular_threshold,
    )

    movement_commands = select_window(
        data["commands"][command_topic],
        movement_start,
        movement_end,
    )

    command_metrics = {}
    for topic, samples in data["commands"].items():
        selected = select_window(samples, movement_start, movement_end)
        command_metrics[topic] = {
            "statistics": vector_statistics(selected),
            "sign_changes": sign_change_statistics(
                selected,
                arguments.linear_threshold,
                arguments.angular_threshold,
            ),
        }

    odometry_metrics = {}
    selected_odometry = {}
    for topic, samples in data["odometry"].items():
        selected = select_window(samples, movement_start, movement_end)
        selected_odometry[topic] = selected
        odometry_metrics[topic] = {
            **odometry_statistics(selected),
            "sign_changes": sign_change_statistics(
                selected,
                arguments.linear_threshold,
                arguments.angular_threshold,
            ),
        }

    odometry_comparisons = {}
    reference_odom = selected_odometry.get("/odom", [])
    for topic in ODOM_TOPICS:
        if topic == "/odom":
            continue
        comparison_name = "odom_vs_" + topic.strip("/")
        odometry_comparisons[comparison_name] = odometry_difference(
            reference_odom,
            selected_odometry.get(topic, []),
        )

    plan_times = [
        timestamp
        for timestamp in data["plan_times"]
        if movement_start <= timestamp <= movement_end
    ]

    amcl_selected = select_window(
        data["amcl"],
        movement_start,
        movement_end,
    )
    terminal_status, log_events = navigation_log_events(
        data["logs"],
        movement_start,
        movement_end,
    )

    metrics = {
        "schema_version": "1.0",
        "source_bag": str(arguments.bag),
        "bag": {
            "start_sec": data["bag_start_sec"],
            "end_sec": data["bag_end_sec"],
            "duration_sec": (
                data["bag_end_sec"] - data["bag_start_sec"]
                if data["bag_start_sec"] is not None
                else None
            ),
        },
        "movement": {
            "detection_topic": command_topic,
            "start_sec": movement_start,
            "end_sec": movement_end,
            "duration_sec": movement_end - movement_start,
            **command_time_statistics(
                movement_commands,
                movement_start,
                movement_end,
                arguments.linear_threshold,
                arguments.angular_threshold,
            ),
        },
        "commands": command_metrics,
        "cmd_vel_vs_cmd_vel_rlsar": command_difference(
            select_window(
                data["commands"].get("/cmd_vel", []),
                movement_start,
                movement_end,
            ),
            select_window(
                data["commands"].get("/cmd_vel_rlsar", []),
                movement_start,
                movement_end,
            ),
        ),
        "odometry": odometry_metrics,
        "odometry_comparisons": odometry_comparisons,
        "amcl_dispersion": amcl_statistics(amcl_selected),
        "global_plans": {
            "published_plan_count": len(plan_times),
            "estimated_replanning_count": max(0, len(plan_times) - 1),
            "interpretation": (
                "Each /plan publication after the first is counted as a "
                "replanning cycle; action status was absent from these bags."
            ),
        },
        "navigation_result": {
            "status": terminal_status,
            "source": "/rosout heuristic",
            "action_status_available": False,
        },
        "thresholds": {
            "linear_mps": arguments.linear_threshold,
            "angular_radps": arguments.angular_threshold,
        },
    }

    arguments.output.mkdir(parents=True, exist_ok=True)

    with (arguments.output / "navigation_metrics.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(metrics, stream, sort_keys=False)

    with (arguments.output / "rosout_navigation_events.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "source_bag": str(arguments.bag),
                "events": log_events,
            },
            stream,
            sort_keys=False,
        )

    write_trajectory_csv(
        arguments.output / "odometry_trajectories.csv",
        data["odometry"],
        movement_start,
        movement_end,
    )

    return metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        type=Path,
        required=True,
        help="Directory containing metadata.yaml and navigation_0.db3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--angular-threshold",
        type=float,
        default=0.02,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if not arguments.bag.is_dir():
        raise RuntimeError(f"Bag directory not found: {arguments.bag}")

    metrics = analyze(arguments)

    movement = metrics["movement"]
    odom = metrics["odometry"].get("/odom", {})
    result = metrics["navigation_result"]["status"]

    print("=" * 72)
    print("G1 NAVIGATION BAG ANALYSIS")
    print("=" * 72)
    print(f"Bag               : {arguments.bag}")
    print(f"Movement duration : {movement['duration_sec']:.3f} s")
    print(f"Stopped time      : {movement['stopped_time_sec']:.3f} s")
    print(f"Rotation time     : {movement['rotation_time_sec']:.3f} s")
    print(f"/odom path length : {odom.get('path_length_m')} m")
    print(f"Global plans      : {metrics['global_plans']['published_plan_count']}")
    print(f"Result heuristic  : {result}")
    print(f"Output            : {arguments.output}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
