#!/usr/bin/env python3
"""Analyze and plot one or more G1 Nav2 rosbag2 trials (ROS 2 Humble)."""

import argparse
import bisect
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


CMD_TOPICS = (
    "/cmd_vel", "/cmd_vel_nav_scaled", "/cmd_vel_mux", "/cmd_vel_rlsar"
)
ODOM_TOPICS = (
    "/odom", "/odom_dog_calibrated", "/dog_odom", "/dog_odom_2d"
)
AMCL_TOPIC = "/amcl_pose"
PLAN_TOPIC = "/plan"
ROSOUT_TOPIC = "/rosout"
ACTION_STATUS_TOPIC = "/navigate_to_pose/_action/status"
LINEAR_THRESHOLD = 0.01
ANGULAR_THRESHOLD = 0.02
ODOMETRY_FILTER_WINDOW_SEC = 0.50
ODOMETRY_RESAMPLE_INTERVAL_SEC = 0.25
AMCL_INITIAL_CONVERGENCE_SEC = 10.0
PLAN_GRID_RESOLUTION_M = 0.25
PLAN_SIMILARITY_THRESHOLD = 0.80
COMMAND_VX_DIFFERENCE_THRESHOLD = 0.02
COMMAND_WZ_DIFFERENCE_THRESHOLD = 0.05


def yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def as_float(value):
    return None if value is None else float(value)


def read_bag(path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(path), storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    wanted = set(CMD_TOPICS + ODOM_TOPICS)
    wanted.update((AMCL_TOPIC, PLAN_TOPIC, ROSOUT_TOPIC, ACTION_STATUS_TOPIC))
    classes = {
        topic: get_message(types[topic])
        for topic in wanted
        if topic in types
    }
    data = {
        "commands": {topic: [] for topic in CMD_TOPICS},
        "odometry": {topic: [] for topic in ODOM_TOPICS},
        "amcl": [], "plans": [], "logs": [], "statuses": [],
        "topics": types, "start": None, "end": None,
    }

    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        t = stamp_ns * 1e-9
        data["start"] = t if data["start"] is None else data["start"]
        data["end"] = t
        if topic not in wanted:
            continue
        cls = classes.get(topic)
        if cls is None:
            continue
        msg = deserialize_message(raw, cls)

        if topic == PLAN_TOPIC:
            points = [
                (float(pose.pose.position.x), float(pose.pose.position.y))
                for pose in msg.poses
            ]
            data["plans"].append((t, points))
        elif topic in CMD_TOPICS:
            data["commands"][topic].append(
                (t, float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            )
        elif topic in ODOM_TOPICS:
            p, tw = msg.pose.pose, msg.twist.twist
            data["odometry"][topic].append(
                (t, float(p.position.x), float(p.position.y), yaw(p.orientation),
                 float(tw.linear.x), float(tw.linear.y), float(tw.angular.z))
            )
        elif topic == AMCL_TOPIC:
            p, c = msg.pose.pose, msg.pose.covariance
            sx, sy = math.sqrt(max(0.0, c[0])), math.sqrt(max(0.0, c[7]))
            data["amcl"].append(
                (t, float(p.position.x), float(p.position.y), yaw(p.orientation),
                 sx, sy, math.hypot(sx, sy), math.sqrt(max(0.0, c[35])))
            )
        elif topic == ROSOUT_TOPIC:
            data["logs"].append((t, str(msg.name), int(msg.level), str(msg.msg)))
        elif topic == ACTION_STATUS_TOPIC:
            for status in msg.status_list:
                goal_id = bytes(status.goal_info.goal_id.uuid).hex()
                goal_stamp = (
                    float(status.goal_info.stamp.sec)
                    + float(status.goal_info.stamp.nanosec) * 1.0e-9
                )
                data["statuses"].append(
                    (t, goal_stamp, goal_id, int(status.status))
                )
    return data


def window_rows(rows, start, end):
    return [row for row in rows if start <= row[0] <= end]


def moving(row):
    return math.hypot(row[1], row[2]) >= LINEAR_THRESHOLD or abs(row[3]) >= ANGULAR_THRESHOLD


def movement_window(commands):
    topic = "/cmd_vel_rlsar" if commands["/cmd_vel_rlsar"] else "/cmd_vel"
    active = [row for row in commands[topic] if moving(row)]
    if not active:
        raise RuntimeError(f"No movement detected on {topic}")
    return topic, active[0][0], active[-1][0]


def axis_stats(rows, columns=(1, 2, 3)):
    result = {}
    for name, column in zip(("vx", "vy", "wz"), columns):
        values = np.asarray([row[column] for row in rows], dtype=float)
        result[name] = {
            "mean": as_float(np.mean(values)) if values.size else None,
            "mean_absolute": as_float(np.mean(np.abs(values))) if values.size else None,
            "minimum": as_float(np.min(values)) if values.size else None,
            "maximum": as_float(np.max(values)) if values.size else None,
            "maximum_absolute": as_float(np.max(np.abs(values))) if values.size else None,
            "p95_absolute": as_float(np.percentile(np.abs(values), 95)) if values.size else None,
            "rms": as_float(np.sqrt(np.mean(values ** 2))) if values.size else None,
        }
    return result


def sign_changes(values, threshold):
    previous, count = 0, 0
    for value in values:
        sign = 1 if value > threshold else -1 if value < -threshold else 0
        if not sign:
            continue
        if previous and sign != previous:
            count += 1
        previous = sign
    return count


def command_time_metrics(rows):
    if len(rows) < 2:
        return {
            "observed_sec": 0.0,
            "stopped_sec": None,
            "rotation_sec": None,
            "in_place_rotation_sec": None,
            "translating_straight_sec": None,
            "translating_with_rotation_sec": None,
        }
    dt_values = np.diff([row[0] for row in rows])
    normal_dt = float(np.median(dt_values[dt_values > 0]))
    maximum_dt = max(0.20, normal_dt * 5.0)
    observed = stopped = rotating = in_place = 0.0
    translating_straight = translating_with_rotation = 0.0
    for first, second in zip(rows, rows[1:]):
        dt = second[0] - first[0]
        if dt <= 0.0 or dt > maximum_dt:
            continue
        observed += dt
        linear = math.hypot(first[1], first[2])
        angular = abs(first[3])
        if linear < LINEAR_THRESHOLD and angular < ANGULAR_THRESHOLD:
            stopped += dt
        elif linear < LINEAR_THRESHOLD and angular >= ANGULAR_THRESHOLD:
            in_place += dt
        elif linear >= LINEAR_THRESHOLD and angular >= ANGULAR_THRESHOLD:
            translating_with_rotation += dt
        else:
            translating_straight += dt

        if angular >= ANGULAR_THRESHOLD:
            rotating += dt
    return {
        "observed_sec": observed,
        "stopped_sec": stopped,
        "stopped_ratio": stopped / observed if observed else None,
        "rotation_sec": rotating,
        "rotation_ratio": rotating / observed if observed else None,
        "in_place_rotation_sec": in_place,
        "in_place_rotation_ratio": in_place / observed if observed else None,
        "translating_straight_sec": translating_straight,
        "translating_with_rotation_sec": translating_with_rotation,
        "exclusive_time_composition": {
            "stopped_sec": stopped,
            "in_place_rotation_sec": in_place,
            "translating_straight_sec": translating_straight,
            "translating_with_rotation_sec": translating_with_rotation,
        },
    }


def path_length(rows):
    """Return the planar length of a sampled trajectory."""

    return sum(
        math.hypot(second[1] - first[1], second[2] - first[2])
        for first, second in zip(rows, rows[1:])
    )


def smooth_odometry(rows, window_sec=ODOMETRY_FILTER_WINDOW_SEC):
    """Apply a centered moving average to odometry position and twist."""

    if len(rows) < 3:
        return list(rows)

    periods = np.diff([row[0] for row in rows])
    periods = periods[periods > 0.0]
    if not periods.size:
        return list(rows)

    window = max(3, int(round(window_sec / float(np.median(periods)))))
    if window % 2 == 0:
        window += 1
    maximum_window = len(rows) if len(rows) % 2 else len(rows) - 1
    window = min(window, maximum_window)
    if window < 3:
        return list(rows)

    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    source = np.asarray(rows, dtype=float)
    filtered = source.copy()

    for column in (1, 2, 4, 5, 6):
        padded = np.pad(source[:, column], (pad, pad), mode="edge")
        filtered[:, column] = np.convolve(padded, kernel, mode="valid")

    unwrapped_yaw = np.unwrap(source[:, 3])
    padded_yaw = np.pad(unwrapped_yaw, (pad, pad), mode="edge")
    filtered[:, 3] = [
        wrap(value)
        for value in np.convolve(padded_yaw, kernel, mode="valid")
    ]

    return [tuple(row) for row in filtered]


def resample_by_time(rows, interval_sec=ODOMETRY_RESAMPLE_INTERVAL_SEC):
    """Keep trajectory samples separated by a minimum time interval."""

    if not rows:
        return []

    selected = [rows[0]]
    last_time = rows[0][0]
    for row in rows[1:-1]:
        if row[0] - last_time >= interval_sec:
            selected.append(row)
            last_time = row[0]
    if len(rows) > 1 and selected[-1][0] != rows[-1][0]:
        selected.append(rows[-1])
    return selected


def odom_metrics(rows):
    if not rows:
        return {"samples": 0}
    distance = path_length(rows)
    filtered_rows = smooth_odometry(rows)
    resampled_rows = resample_by_time(filtered_rows)
    filtered_distance = path_length(resampled_rows)
    integrated = sum(
        math.hypot(a[4], a[5]) * (b[0] - a[0])
        for a, b in zip(rows, rows[1:])
        if 0.0 < b[0] - a[0] <= 0.20
    )
    first, last = rows[0], rows[-1]
    displacement = math.hypot(last[1] - first[1], last[2] - first[2])
    filtered_displacement = (
        math.hypot(
            resampled_rows[-1][1] - resampled_rows[0][1],
            resampled_rows[-1][2] - resampled_rows[0][2],
        )
        if resampled_rows
        else 0.0
    )
    return {
        "samples": len(rows),
        "path_length_m": distance,
        "filtered_path_length_m": filtered_distance,
        "filter_window_sec": ODOMETRY_FILTER_WINDOW_SEC,
        "resample_interval_sec": ODOMETRY_RESAMPLE_INTERVAL_SEC,
        "twist_integrated_distance_m": integrated,
        "displacement_m": displacement,
        "filtered_displacement_m": filtered_displacement,
        "path_efficiency": (
            filtered_displacement / filtered_distance
            if filtered_distance > 0.0
            else None
        ),
        "heading_change_rad": wrap(last[3] - first[3]),
        "start": {"x": first[1], "y": first[2], "yaw": first[3]},
        "end": {"x": last[1], "y": last[2], "yaw": last[3]},
        "velocity": axis_stats(rows, columns=(4, 5, 6)),
        "filtered_velocity": axis_stats(filtered_rows, columns=(4, 5, 6)),
        "sign_changes": {
            "vx": sign_changes([r[4] for r in rows], LINEAR_THRESHOLD),
            "vy": sign_changes([r[5] for r in rows], LINEAR_THRESHOLD),
            "wz": sign_changes([r[6] for r in rows], ANGULAR_THRESHOLD),
        },
    }


def nearest_pairs(reference, candidate, tolerance):
    if not reference or not candidate:
        return []
    times = [row[0] for row in candidate]
    pairs = []
    for row in reference:
        i = bisect.bisect_left(times, row[0])
        choices = candidate[max(0, i - 1):min(len(candidate), i + 1)]
        if choices:
            match = min(choices, key=lambda x: abs(x[0] - row[0]))
            if abs(match[0] - row[0]) <= tolerance:
                pairs.append((row, match))
    return pairs


def odom_comparison(reference, candidate):
    pairs = nearest_pairs(reference, candidate, 0.03)
    if not pairs:
        return {"matched_samples": 0}
    p0, q0 = pairs[0]
    absolute_position = [math.hypot(a[1] - b[1], a[2] - b[2]) for a, b in pairs]
    relative_position = [
        math.hypot((a[1] - p0[1]) - (b[1] - q0[1]),
                   (a[2] - p0[2]) - (b[2] - q0[2]))
        for a, b in pairs
    ]
    yaw_error = [abs(wrap((a[3] - p0[3]) - (b[3] - q0[3]))) for a, b in pairs]
    return {
        "matched_samples": len(pairs),
        "absolute_position_rmse_m": float(np.sqrt(np.mean(np.square(absolute_position)))),
        "relative_position_rmse_m": float(np.sqrt(np.mean(np.square(relative_position)))),
        "relative_position_max_m": max(relative_position),
        "relative_yaw_rmse_rad": float(np.sqrt(np.mean(np.square(yaw_error)))),
    }


def command_comparison(cmd, output):
    pairs = nearest_pairs(cmd, output, 0.10)
    result = {"matched_samples": len(pairs)}
    for name, col in (("vx", 1), ("vy", 2), ("wz", 3)):
        errors = np.asarray([a[col] - b[col] for a, b in pairs], dtype=float)
        result[name] = {
            "mae": float(np.mean(np.abs(errors))) if errors.size else None,
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
            "maximum_absolute": float(np.max(np.abs(errors))) if errors.size else None,
        }
    anomalous_times = [
        first[0]
        for first, second in pairs
        if (
            abs(first[1] - second[1]) >= COMMAND_VX_DIFFERENCE_THRESHOLD
            or abs(first[3] - second[3]) >= COMMAND_WZ_DIFFERENCE_THRESHOLD
        )
    ]
    result["difference_thresholds"] = {
        "vx_mps": COMMAND_VX_DIFFERENCE_THRESHOLD,
        "wz_radps": COMMAND_WZ_DIFFERENCE_THRESHOLD,
    }
    result["significant_difference_samples"] = len(anomalous_times)
    result["significant_difference_ratio"] = (
        len(anomalous_times) / len(pairs)
        if pairs
        else None
    )
    result["first_significant_difference_time"] = (
        anomalous_times[0]
        if anomalous_times
        else None
    )
    result["last_significant_difference_time"] = (
        anomalous_times[-1]
        if anomalous_times
        else None
    )
    return result


def amcl_metrics(rows):
    result = {"samples": len(rows)}
    for name, col in (("sigma_x_m", 4), ("sigma_y_m", 5),
                      ("sigma_xy_m", 6), ("sigma_yaw_rad", 7)):
        values = np.asarray([row[col] for row in rows], dtype=float)
        result[name] = {
            "mean": float(np.mean(values)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
        }
        if rows:
            steady_values = np.asarray(
                [
                    row[col]
                    for row in rows
                    if row[0] - rows[0][0] >= AMCL_INITIAL_CONVERGENCE_SEC
                ],
                dtype=float,
            )
            result[name]["steady_state_mean"] = (
                float(np.mean(steady_values))
                if steady_values.size
                else None
            )
            result[name]["steady_state_p95"] = (
                float(np.percentile(steady_values, 95))
                if steady_values.size
                else None
            )
    result["initial_convergence_excluded_sec"] = AMCL_INITIAL_CONVERGENCE_SEC
    return result


def classify_navigation_log(row):
    """Classify one ROS log without confusing recovery failures with goal failure."""

    _, node, _, message = row
    node = node.lstrip("/")
    text = message.lower()

    if node == "bt_navigator" and "goal succeeded" in text:
        return "GOAL_SUCCEEDED"
    if node == "controller_server" and "reached the goal" in text:
        return "GOAL_REACHED_SUPPORTING_EVIDENCE"
    if node == "bt_navigator" and any(
        phrase in text
        for phrase in (
            "goal failed",
            "navigation failed",
            "failed to navigate",
            "goal aborted",
        )
    ):
        return "GOAL_ABORTED"
    if node == "bt_navigator" and any(
        phrase in text
        for phrase in ("goal canceled", "goal cancelled")
    ):
        return "GOAL_CANCELED"
    if "aborting handle" in text or text.endswith(" failed"):
        if node in {
            "controller_server",
            "planner_server",
            "behavior_server",
        }:
            return "INTERNAL_ACTION_FAILURE"
    if node.startswith("bt_navigator_") and (
        "timed out" in text or "failed to get result" in text
    ):
        return "BT_INTERNAL_FAILURE"
    if node == "behavior_server" and "completed successfully" in text:
        return "RECOVERY_SUCCEEDED"
    if node == "planner_server" and (
        "failed to create plan" in text or "failed to generate" in text
    ):
        return "PLANNER_FAILURE"
    return None


def goal_result(data, start, analysis_end):
    """Resolve the final top-level NavigateToPose state and supporting evidence."""

    names = {
        0: "UNKNOWN",
        1: "ACCEPTED",
        2: "EXECUTING",
        3: "CANCELING",
        4: "SUCCEEDED",
        5: "CANCELED",
        6: "ABORTED",
    }
    statuses = [
        row
        for row in data["statuses"]
        if start <= row[0] <= analysis_end
    ]
    classified_events = []
    for row in data["logs"]:
        if not start <= row[0] <= analysis_end:
            continue
        category = classify_navigation_log(row)
        if category:
            classified_events.append((*row, category))

    if statuses:
        goal_order = {}
        for bag_time, goal_stamp, goal_id, status in statuses:
            if not goal_id or set(goal_id) == {"0"}:
                continue
            goal_order.setdefault(goal_id, (goal_stamp, bag_time))

        if goal_order:
            selected_goal_id = max(
                goal_order,
                key=lambda goal_id: goal_order[goal_id],
            )
            selected_statuses = [
                row
                for row in statuses
                if row[2] == selected_goal_id
            ]
            terminal = [row for row in selected_statuses if row[3] in (4, 5, 6)]
            if terminal:
                selected = terminal[-1]
                return {
                    "state": names[selected[3]],
                    "source": "action_status",
                    "goal_id": selected_goal_id,
                    "terminal_evidence": names[selected[3]],
                }, classified_events

    terminal_events = [
        row
        for row in classified_events
        if row[4] in {"GOAL_SUCCEEDED", "GOAL_ABORTED", "GOAL_CANCELED"}
    ]
    if terminal_events:
        selected = terminal_events[-1]
        state = {
            "GOAL_SUCCEEDED": "SUCCEEDED",
            "GOAL_ABORTED": "ABORTED",
            "GOAL_CANCELED": "CANCELED",
        }[selected[4]]
        return {
            "state": state,
            "source": "rosout_terminal_event",
            "goal_id": None,
            "terminal_evidence": selected[3],
        }, classified_events

    supporting_success = [
        row
        for row in classified_events
        if row[4] == "GOAL_REACHED_SUPPORTING_EVIDENCE"
    ]
    if supporting_success:
        return {
            "state": "SUCCEEDED",
            "source": "rosout_supporting_evidence",
            "goal_id": None,
            "terminal_evidence": supporting_success[-1][3],
        }, classified_events

    return {
        "state": "UNKNOWN_NO_TERMINAL_RESULT",
        "source": "no_top_level_terminal_event",
        "goal_id": None,
        "terminal_evidence": None,
    }, classified_events


def plan_cell_set(points):
    """Quantize a path into occupied cells for geometry comparison."""

    return {
        (
            int(round(x / PLAN_GRID_RESOLUTION_M)),
            int(round(y / PLAN_GRID_RESOLUTION_M)),
        )
        for x, y in points
    }


def plan_metrics(plans, duration_sec):
    """Separate plan publications from significant path geometry changes."""

    significant_changes = 0
    similarities = []
    previous_cells = None

    for _, points in plans:
        cells = plan_cell_set(points)
        if previous_cells and cells:
            similarity = len(previous_cells & cells) / float(
                min(len(previous_cells), len(cells))
            )
            similarities.append(similarity)
            if similarity < PLAN_SIMILARITY_THRESHOLD:
                significant_changes += 1
        previous_cells = cells

    return {
        "published": len(plans),
        "publication_rate_hz": len(plans) / duration_sec if duration_sec > 0.0 else None,
        "significant_geometry_changes": significant_changes,
        "geometry_similarity_threshold": PLAN_SIMILARITY_THRESHOLD,
        "grid_resolution_m": PLAN_GRID_RESOLUTION_M,
        "mean_consecutive_similarity": (
            float(np.mean(similarities))
            if similarities
            else None
        ),
    }


def configured_speed(label):
    match = re.search(r"vx[_-]?(\d{3})", label.lower())
    return int(match.group(1)) / 100.0 if match else None


def route_identifier(path):
    """Extract the route identifier from a campaign path when available."""

    match = re.search(r"corridor_start_goal_\d+", str(path))
    return match.group(0) if match else "UNKNOWN_ROUTE"


def analyze_trial(label, bag_path, output_root):
    data = read_bag(bag_path)
    command_topic, start, end = movement_window(data["commands"])
    command_rows = {k: window_rows(v, start, end) for k, v in data["commands"].items()}
    odom_rows = {k: window_rows(v, start, end) for k, v in data["odometry"].items()}
    amcl_rows = window_rows(data["amcl"], start, end)
    goal, events = goal_result(data, start, data["end"])
    plans = [row for row in data["plans"] if start <= row[0] <= end]
    command_difference = command_comparison(
        command_rows["/cmd_vel"], command_rows["/cmd_vel_rlsar"]
    )
    first_difference_time = command_difference.get(
        "first_significant_difference_time"
    )
    last_difference_time = command_difference.get(
        "last_significant_difference_time"
    )
    if first_difference_time is not None:
        command_difference["first_significant_difference_sec"] = (
            first_difference_time - start
        )
        command_difference["last_significant_difference_sec"] = (
            last_difference_time - start
        )
    command_difference.pop("first_significant_difference_time", None)
    command_difference.pop("last_significant_difference_time", None)
    event_counts = {}
    for event in events:
        event_counts[event[4]] = event_counts.get(event[4], 0) + 1

    metrics = {
        "label": label,
        "source_bag": str(bag_path),
        "route_id": route_identifier(bag_path),
        "configured_speed_mps": configured_speed(label),
        "bag_duration_sec": data["end"] - data["start"],
        "movement": {
            "detection_topic": command_topic,
            "duration_sec": end - start,
            **command_time_metrics(command_rows[command_topic]),
        },
        "commands": {
            topic: {
                "statistics": axis_stats(rows),
                "sign_changes": {
                    "vx": sign_changes([r[1] for r in rows], LINEAR_THRESHOLD),
                    "vy": sign_changes([r[2] for r in rows], LINEAR_THRESHOLD),
                    "wz": sign_changes([r[3] for r in rows], ANGULAR_THRESHOLD),
                },
            }
            for topic, rows in command_rows.items()
        },
        "cmd_vel_vs_cmd_vel_rlsar": command_difference,
        "odometry": {topic: odom_metrics(rows) for topic, rows in odom_rows.items()},
        "odometry_comparisons": {
            f"odom_vs_{topic.lstrip('/').replace('/', '_')}": odom_comparison(
                odom_rows["/odom"], rows
            )
            for topic, rows in odom_rows.items() if topic != "/odom"
        },
        "amcl_dispersion": amcl_metrics(amcl_rows),
        "global_plans": plan_metrics(plans, end - start),
        "navigation_event_counts": event_counts,
        "goal": goal,
    }

    out = output_root / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "navigation_metrics.yaml").open("w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    with (out / "rosout_goal_events.yaml").open("w") as f:
        yaml.safe_dump({"events": [dict(time=r[0], node=r[1], level=r[2], message=r[3], category=r[4])
                                   for r in events]}, f, sort_keys=False)
    write_trial_report(out / "analysis_report.md", metrics)
    write_trajectory_csv(out / "odometry_trajectories.csv", odom_rows, start)
    plot_trial(label, metrics, command_rows, odom_rows, amcl_rows, events, start, out)
    return metrics, odom_rows


def format_value(value, digits=3):
    """Format an optional numerical value for a Markdown report."""

    return "N/A" if value is None else f"{value:.{digits}f}"


def write_trial_report(path, metrics):
    """Write an evidence-based interpretation for one navigation trial."""

    movement = metrics["movement"]
    odom = metrics["odometry"]["/odom"]
    command = metrics["commands"]["/cmd_vel"]["statistics"]
    measured = odom["velocity"]
    filtered = odom["filtered_velocity"]
    comparison = metrics["cmd_vel_vs_cmd_vel_rlsar"]
    amcl = metrics["amcl_dispersion"]
    plans = metrics["global_plans"]
    event_counts = metrics["navigation_event_counts"]
    goal = metrics["goal"]
    configured = metrics["configured_speed_mps"]
    command_ratio = (
        100.0 * command["vx"]["mean_absolute"] / configured
        if configured
        else None
    )
    measured_ratio = (
        100.0 * filtered["vx"]["mean_absolute"] / configured
        if configured
        else None
    )

    lines = [
        f"# Navigation Trial Analysis — {metrics['label']}",
        "",
        "## Experimental identity",
        "",
        f"- Route: `{metrics['route_id']}`",
        f"- Configured maximum forward speed: {format_value(configured, 2)} m/s",
        f"- Final navigation goal state: `{goal['state']}`",
        f"- Goal-state evidence source: `{goal['source']}`",
        "",
        "## Main findings",
        "",
        f"- Actual movement window: {format_value(movement['duration_sec'], 2)} s.",
        f"- Raw `/odom` path length: {format_value(odom['path_length_m'], 2)} m.",
        f"- Filtered `/odom` path length: {format_value(odom['filtered_path_length_m'], 2)} m.",
        f"- Net displacement: {format_value(odom['displacement_m'], 2)} m.",
        f"- Filtered path efficiency: {format_value(odom['path_efficiency'], 3)}.",
        f"- Mean commanded `|vx|`: {format_value(command['vx']['mean_absolute'])} m/s "
        f"({format_value(command_ratio, 1)}% of the configured maximum).",
        f"- Mean filtered measured `|vx|`: {format_value(filtered['vx']['mean_absolute'])} m/s "
        f"({format_value(measured_ratio, 1)}% of the configured maximum).",
        f"- Raw measured `|vx|` P95: {format_value(measured['vx']['p95_absolute'])} m/s.",
        f"- In-place rotation time: {format_value(movement['in_place_rotation_sec'], 2)} s.",
        f"- Translating with angular correction: "
        f"{format_value(movement['translating_with_rotation_sec'], 2)} s.",
        f"- Commanded `wz` sign changes: "
        f"{metrics['commands']['/cmd_vel']['sign_changes']['wz']}.",
        "",
        "## Command-chain integrity",
        "",
        f"- `/cmd_vel` versus `/cmd_vel_rlsar` `vx` RMSE: "
        f"{format_value(comparison['vx']['rmse'], 5)} m/s.",
        f"- `/cmd_vel` versus `/cmd_vel_rlsar` `wz` RMSE: "
        f"{format_value(comparison['wz']['rmse'], 5)} rad/s.",
        f"- Samples above the significant-difference thresholds: "
        f"{comparison['significant_difference_samples']}.",
        "",
        "## Localization and planning",
        "",
        f"- AMCL combined position uncertainty P95: "
        f"{format_value(amcl['sigma_xy_m']['p95'])} m.",
        f"- AMCL steady-state combined position uncertainty P95: "
        f"{format_value(amcl['sigma_xy_m'].get('steady_state_p95'))} m.",
        f"- Global-plan publications: {plans['published']}.",
        f"- Significant global-plan geometry changes: "
        f"{plans['significant_geometry_changes']}.",
        f"- Internal action failures: "
        f"{event_counts.get('INTERNAL_ACTION_FAILURE', 0)}.",
        f"- Behavior-tree internal failures: "
        f"{event_counts.get('BT_INTERNAL_FAILURE', 0)}.",
        f"- Successful recovery actions: "
        f"{event_counts.get('RECOVERY_SUCCEEDED', 0)}.",
        "",
        "## Interpretation limits",
        "",
        "- Raw odometry maxima are sensitive to gait oscillations and must not be interpreted as sustained navigation speed.",
        "- Rotation time includes forward motion with angular correction; in-place rotation is reported separately.",
        "- A missing top-level terminal event produces `UNKNOWN_NO_TERMINAL_RESULT`; internal recovery failures do not determine the final goal state.",
    ]

    if goal["state"] == "UNKNOWN_NO_TERMINAL_RESULT":
        lines.extend(
            (
                "",
                "## Goal-state conclusion",
                "",
                "The bag contains no reliable top-level terminal result. Internal planner, controller, or recovery failures may have occurred, but they do not prove that the NavigateToPose goal was aborted.",
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(path, results):
    """Write a comparison report and explicitly state experimental validity."""

    labels = list(results)
    routes = {results[label]["route_id"] for label in labels}
    same_route = len(routes) == 1 and "UNKNOWN_ROUTE" not in routes
    lines = [
        "# Navigation Trial Comparison",
        "",
        "## Comparison validity",
        "",
        f"- Same physical route: `{'YES' if same_route else 'NO'}`.",
        f"- Speed effect isolated: `{'YES' if same_route else 'NO'}`.",
    ]
    if not same_route:
        lines.append(
            "- Conclusion: this comparison is descriptive only because the trials use different routes. Duration, distance, rotation, replanning, and goal outcome cannot be attributed to speed alone."
        )

    lines.extend(("", "## Trial summary", ""))
    lines.append(
        "| Trial | Route | Speed | Goal | Movement | Filtered path | Displacement | In-place rotation | `wz` sign changes |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for label in labels:
        trial = results[label]
        movement = trial["movement"]
        odom = trial["odometry"]["/odom"]
        lines.append(
            f"| {label} | {trial['route_id']} | "
            f"{format_value(trial['configured_speed_mps'], 2)} m/s | "
            f"{trial['goal']['state']} | "
            f"{format_value(movement['duration_sec'], 2)} s | "
            f"{format_value(odom['filtered_path_length_m'], 2)} m | "
            f"{format_value(odom['displacement_m'], 2)} m | "
            f"{format_value(movement['in_place_rotation_sec'], 2)} s | "
            f"{trial['commands']['/cmd_vel']['sign_changes']['wz']} |"
        )

    lines.extend(
        (
            "",
            "## Goal-state interpretation",
            "",
        )
    )
    for label in labels:
        goal = results[label]["goal"]
        lines.append(
            f"- `{label}`: `{goal['state']}` from `{goal['source']}`."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trajectory_csv(path, odometry, start):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("topic", "time_sec", "x", "y", "yaw", "vx", "vy", "wz"))
        for topic, rows in odometry.items():
            for row in rows:
                writer.writerow((topic, row[0] - start, *row[1:]))


def finish_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trial(label, metrics, commands, odometry, amcl, events, start, out):
    colors = {"/odom": "black", "/odom_dog_calibrated": "tab:blue",
              "/dog_odom": "tab:orange", "/dog_odom_2d": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for topic, rows in odometry.items():
        if not rows:
            continue
        x0, y0 = rows[0][1], rows[0][2]
        ax.plot([r[1] - x0 for r in rows], [r[2] - y0 for r in rows],
                label=topic, linewidth=1.5, color=colors[topic])
    ax.set(title=f"Normalized odometry trajectories — {label}", xlabel="Relative x (m)",
           ylabel="Relative y (m)")
    ax.axis("equal"); ax.grid(True, alpha=.3); ax.legend()
    finish_figure(fig, out / "trajectory_odometry.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    speed = metrics["configured_speed_mps"]
    for axis, name, col, unit in zip(axes, ("vx", "vy", "wz"), (1, 2, 3),
                                     ("m/s", "m/s", "rad/s")):
        for topic in ("/cmd_vel", "/cmd_vel_rlsar"):
            rows = commands[topic]
            if rows:
                axis.plot([r[0] - start for r in rows], [r[col] for r in rows],
                          label=topic, linewidth=1.0)
        rows = odometry["/odom"]
        if rows:
            filtered_rows = smooth_odometry(rows)
            axis.plot(
                [r[0] - start for r in rows],
                [r[col + 3] for r in rows],
                label="/odom raw",
                linewidth=.5,
                alpha=.20,
            )
            axis.plot(
                [r[0] - start for r in filtered_rows],
                [r[col + 3] for r in filtered_rows],
                label=f"/odom filtered ({ODOMETRY_FILTER_WINDOW_SEC:.2f} s)",
                linewidth=1.4,
            )
        if name == "vx" and speed is not None:
            axis.axhline(speed, color="red", linestyle="--", label="configured maximum")
        axis.set_ylabel(f"{name} ({unit})"); axis.grid(True, alpha=.3); axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time since movement start (s)")
    axes[0].set_title(f"Commanded and measured velocities — {label}")
    finish_figure(fig, out / "velocities.png")

    pairs = nearest_pairs(commands["/cmd_vel"], commands["/cmd_vel_rlsar"], .10)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, name, col in zip(axes, ("vx", "vy", "wz"), (1, 2, 3)):
        axis.plot([a[0] - start for a, _ in pairs], [a[col] - b[col] for a, b in pairs])
        axis.axhline(0, color="black", linewidth=.7)
        threshold = (
            COMMAND_VX_DIFFERENCE_THRESHOLD
            if name == "vx"
            else COMMAND_WZ_DIFFERENCE_THRESHOLD
            if name == "wz"
            else None
        )
        if threshold is not None:
            axis.axhline(threshold, color="red", linestyle="--", linewidth=.8)
            axis.axhline(-threshold, color="red", linestyle="--", linewidth=.8)
        axis.set_ylabel(f"Δ{name}"); axis.grid(True, alpha=.3)
    axes[-1].set_xlabel("Time (s)"); axes[0].set_title(f"/cmd_vel − /cmd_vel_rlsar — {label}")
    finish_figure(fig, out / "command_difference.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if amcl:
        time = [r[0] - start for r in amcl]
        axes[0].plot(time, [r[4] for r in amcl], label="σx")
        axes[0].plot(time, [r[5] for r in amcl], label="σy")
        axes[0].plot(time, [r[6] for r in amcl], label="σxy")
        axes[1].plot(time, [math.degrees(r[7]) for r in amcl], label="σyaw")
        for axis in axes:
            axis.axvspan(
                0.0,
                AMCL_INITIAL_CONVERGENCE_SEC,
                color="tab:gray",
                alpha=.18,
                label="initial convergence window",
            )
    axes[0].set_ylabel("Position uncertainty (m)"); axes[1].set_ylabel("Yaw uncertainty (deg)")
    axes[1].set_xlabel("Time (s)")
    for axis in axes: axis.grid(True, alpha=.3); axis.legend()
    axes[0].set_title(f"AMCL uncertainty — {label}")
    finish_figure(fig, out / "amcl_dispersion.png")

    movement, odom = metrics["movement"], metrics["odometry"]["/odom"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    distance_values = [
        odom.get("path_length_m", 0),
        odom.get("filtered_path_length_m", 0),
        odom.get("displacement_m", 0),
    ]
    bars = axes[0].bar(
        ("Raw path", "Filtered path", "Displacement"),
        distance_values,
    )
    axes[0].bar_label(bars, fmt="%.2f")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("Distance interpretation")
    axes[0].tick_params(axis="x", rotation=18)

    composition = movement["exclusive_time_composition"]
    composition_labels = (
        "Stopped",
        "In-place rotation",
        "Straight translation",
        "Translation + angular correction",
    )
    composition_values = (
        composition["stopped_sec"],
        composition["in_place_rotation_sec"],
        composition["translating_straight_sec"],
        composition["translating_with_rotation_sec"],
    )
    bottom = 0.0
    for name, value in zip(composition_labels, composition_values):
        axes[1].bar(label, value, bottom=bottom, label=name)
        bottom += value
    axes[1].set_ylabel("Observed time (s)")
    axes[1].set_title("Exclusive motion composition")
    axes[1].legend(fontsize=8)

    goal_state = metrics["goal"]["state"]
    goal_color = (
        "tab:green" if goal_state == "SUCCEEDED"
        else "tab:red" if goal_state in {"ABORTED", "CANCELED"}
        else "tab:gray"
    )
    axes[2].bar((label,), (1.0,), color=goal_color)
    axes[2].text(0, .5, goal_state, ha="center", va="center",
                 color="white", weight="bold", wrap=True)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_title("Top-level navigation result")

    for axis in axes[:2]:
        axis.grid(True, axis="y", alpha=.3)
    fig.suptitle(f"Trial evidence summary — {label}")
    finish_figure(fig, out / "trial_summary.png")

    event_categories = (
        "PLANNER_FAILURE",
        "INTERNAL_ACTION_FAILURE",
        "BT_INTERNAL_FAILURE",
        "RECOVERY_SUCCEEDED",
        "GOAL_REACHED_SUPPORTING_EVIDENCE",
        "GOAL_SUCCEEDED",
        "GOAL_ABORTED",
        "GOAL_CANCELED",
    )
    category_y = {name: index for index, name in enumerate(event_categories)}
    category_colors = {
        "PLANNER_FAILURE": "tab:orange",
        "INTERNAL_ACTION_FAILURE": "tab:red",
        "BT_INTERNAL_FAILURE": "tab:purple",
        "RECOVERY_SUCCEEDED": "tab:blue",
        "GOAL_REACHED_SUPPORTING_EVIDENCE": "tab:green",
        "GOAL_SUCCEEDED": "tab:green",
        "GOAL_ABORTED": "tab:red",
        "GOAL_CANCELED": "tab:gray",
    }
    fig, axis = plt.subplots(figsize=(12, 6))
    for category in event_categories:
        times = [
            row[0] - start
            for row in events
            if row[4] == category
        ]
        if times:
            axis.scatter(
                times,
                [category_y[category]] * len(times),
                label=category,
                color=category_colors[category],
                s=35,
            )
    axis.set_yticks(list(category_y.values()), list(category_y.keys()))
    axis.set_xlabel("Time since movement start (s)")
    axis.set_title(
        f"Navigation event timeline — {label} — final state: {goal_state}"
    )
    axis.grid(True, axis="x", alpha=.3)
    axis.axvline(movement["duration_sec"], color="black", linestyle="--",
                 label="last movement command")
    axis.legend(loc="best", fontsize=8)
    finish_figure(fig, out / "navigation_event_timeline.png")


def plot_comparison(results, trajectories, output):
    labels = list(results)
    routes = {results[label]["route_id"] for label in labels}
    same_route = len(routes) == 1 and "UNKNOWN_ROUTE" not in routes
    validity_note = (
        "Same route: speed comparison can be evaluated"
        if same_route
        else "Different routes: descriptive comparison only"
    )
    duration = [results[x]["movement"]["duration_sec"] for x in labels]
    raw_distance = [results[x]["odometry"]["/odom"].get("path_length_m", 0) for x in labels]
    filtered_distance = [results[x]["odometry"]["/odom"].get("filtered_path_length_m", 0) for x in labels]
    displacement = [results[x]["odometry"]["/odom"].get("displacement_m", 0) for x in labels]
    stopped = [results[x]["movement"].get("stopped_sec") or 0 for x in labels]
    in_place = [results[x]["movement"].get("in_place_rotation_sec") or 0 for x in labels]
    straight = [results[x]["movement"].get("translating_straight_sec") or 0 for x in labels]
    angular_translation = [results[x]["movement"].get("translating_with_rotation_sec") or 0 for x in labels]
    amcl_p95 = [results[x]["amcl_dispersion"]["sigma_xy_m"].get("steady_state_p95") or 0 for x in labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bars = axes[0, 0].bar(labels, duration)
    axes[0, 0].bar_label(bars, fmt="%.2f")
    axes[0, 0].set_ylabel("Time (s)")
    axes[0, 0].set_title("Actual movement window")

    x = np.arange(len(labels))
    width = .25
    axes[0, 1].bar(x - width, raw_distance, width, label="Raw path")
    axes[0, 1].bar(x, filtered_distance, width, label="Filtered path")
    axes[0, 1].bar(x + width, displacement, width, label="Displacement")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Distance (m)")
    axes[0, 1].set_title("Raw distance versus robust distance")
    axes[0, 1].legend(fontsize=8)

    bottom = np.zeros(len(labels))
    for name, values in (
        ("Stopped", stopped),
        ("In-place rotation", in_place),
        ("Straight translation", straight),
        ("Translation + angular correction", angular_translation),
    ):
        axes[1, 0].bar(labels, values, bottom=bottom, label=name)
        bottom += np.asarray(values)
    axes[1, 0].set_ylabel("Observed time (s)")
    axes[1, 0].set_title("Exclusive motion composition")
    axes[1, 0].legend(fontsize=8)

    bars = axes[1, 1].bar(labels, amcl_p95)
    axes[1, 1].bar_label(bars, fmt="%.3f")
    axes[1, 1].set_ylabel("Position uncertainty (m)")
    axes[1, 1].set_title("Steady-state AMCL combined P95")

    for axis in axes.flat:
        axis.grid(True, axis="y", alpha=.3)
    fig.suptitle(f"Navigation evidence comparison\n{validity_note}")
    finish_figure(fig, output / "comparison_summary.png")

    configured = [results[x]["configured_speed_mps"] for x in labels]
    commanded_mean = [
        results[x]["commands"]["/cmd_vel"]["statistics"]["vx"]["mean_absolute"]
        for x in labels
    ]
    filtered_mean_vx = [
        results[x]["odometry"]["/odom"]["filtered_velocity"]["vx"]["mean_absolute"]
        for x in labels
    ]
    measured_p95_vx = [
        results[x]["odometry"]["/odom"]["velocity"]["vx"]["p95_absolute"]
        for x in labels
    ]

    x = np.arange(len(labels)); width = .20
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, configured, width, label="Configured maximum")
    ax.bar(x - .5 * width, commanded_mean, width, label="Commanded mean |vx|")
    ax.bar(x + .5 * width, filtered_mean_vx, width, label="Filtered measured mean |vx|")
    ax.bar(x + 1.5 * width, measured_p95_vx, width, label="Raw measured |vx| P95")
    ax.set_xticks(x, labels); ax.set_ylabel("m/s")
    ax.set_title("Forward-speed realization without raw maxima")
    ax.grid(True, axis="y", alpha=.3); ax.legend()
    finish_figure(fig, output / "comparison_velocity.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    for label in labels:
        rows = trajectories[label]["/odom"]
        if rows:
            x0, y0 = rows[0][1], rows[0][2]
            ax.plot([r[1] - x0 for r in rows], [r[2] - y0 for r in rows], label=label)
    ax.axis("equal"); ax.grid(True, alpha=.3); ax.legend()
    ax.set(title=f"Normalized /odom trajectories\n{validity_note}", xlabel="Relative x (m)", ylabel="Relative y (m)")
    finish_figure(fig, output / "comparison_trajectories.png")

    states = [results[x]["goal"]["state"] for x in labels]
    colors = [
        "tab:green" if state == "SUCCEEDED"
        else "tab:red" if state in {"ABORTED", "CANCELED"}
        else "tab:gray"
        for state in states
    ]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(labels, [1] * len(labels), color=colors)
    for bar, state in zip(bars, states):
        ax.text(bar.get_x() + bar.get_width()/2, .5, state, ha="center", va="center", color="white", weight="bold")
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title("Final top-level navigation goal state")
    finish_figure(fig, output / "comparison_goal_state.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag", action="append", required=True, metavar="LABEL=PATH",
        help="Repeat once per trial, e.g. --bag vx_012=/path/to/navigation",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, trajectories = {}, {}
    for specification in args.bag:
        if "=" not in specification:
            raise RuntimeError(f"Invalid --bag {specification!r}; expected LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"Bag directory not found: {path}")
        print(f"Analyzing {label}: {path}")
        metrics, odometry = analyze_trial(label, path, args.output)
        results[label], trajectories[label] = metrics, odometry

    routes = {trial["route_id"] for trial in results.values()}
    same_route = len(routes) == 1 and "UNKNOWN_ROUTE" not in routes
    comparison = {
        "comparison_validity": {
            "same_route": same_route,
            "speed_effect_isolated": same_route,
            "interpretation": (
                "Speed comparison can be evaluated because all trials use the same route."
                if same_route
                else "Descriptive comparison only: trials use different routes, so differences cannot be attributed to speed alone."
            ),
        },
        "trials": results,
    }
    with (args.output / "comparison_metrics.yaml").open("w") as f:
        yaml.safe_dump(comparison, f, sort_keys=False)
    write_comparison_report(args.output / "comparison_report.md", results)
    plot_comparison(results, trajectories, args.output)
    print(f"Results written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
