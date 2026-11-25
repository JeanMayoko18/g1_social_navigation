
#!/usr/bin/env python3

"""Extract a chronological JSONL timeline from a G1 experiment rosbag."""

import argparse
from array import array
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re
import math
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ================================================================
# COMMON TOPICS: EXPERIMENTAL CONFIGURATIONS A, C, D
# ================================================================

COMMON_TOPICS = {
    # Goal and navigation.
    "/goal_pose",
    "/navigate_to_pose/_action/feedback",
    "/navigate_to_pose/_action/status",
    "/navigate_to_pose_internal/_action/goal",
    "/navigate_to_pose_internal/_action/cancel",
    "/navigate_to_pose_internal/_action/feedback",
    "/navigate_to_pose_internal/_action/status",
    "/navigate_to_pose_internal/_action/result",
    "/follow_path/_action/goal",
    "/follow_path/_action/cancel",
    "/follow_path/_action/feedback",
    "/follow_path/_action/status",
    "/follow_path/_action/result",
    "/compute_path_to_pose/_action/status",
    "/compute_path_to_pose/_action/feedback",

    # Movement commands.
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_nav_scaled",
    "/cmd_vel_mux",
    "/cmd_vel_rlsar",

    # Odometry and localization.
    "/odom",
    "/odom_dog_calibrated",
    "/dog_odom",
    "/dog_odom_2d",
    "/amcl_pose",

    # TF.
    "/tf",
    "/tf_static",

    # Planning.
    "/plan",
    "/plan_smoothed",
    "/local_plan",
    "/received_global_plan",
    "/transformed_global_plan",

    # Maps and costmaps.
    "/map",
    "/map_runtime",
    "/initialpose",
    "/global_costmap/costmap",
    "/global_costmap/costmap_updates",
    "/local_costmap/costmap",
    "/local_costmap/costmap_updates",

    # Sensor and system diagnostics.
    "/scan",
    "/scan_raw",
    "/scan_filtered",
    "/clock",
    "/diagnostics",
    "/diagnostics_agg",
    "/parameter_events",
    "/rosout",
}


# ================================================================
# SOCIAL TOPICS: CONFIGURATIONS C, D
# ================================================================

SOCIAL_TOPICS = {
    # Person detection and tracking.
    "/yolo/detections",
    "/yolo/detections_tracked",
    "/yolo/detections_3d",
    "/yolo/detections_3d_base",
    "/social_perception/detections_source",
    "/social_perception/detections_3d",
    "/social_perception/detections_3d_odom",
    "/social_perception/detections_identified_odom",
    "/social_perception/tracked_persons",
    "/social_perception/tracked_persons_odom",
    "/social_perception/social_zones",
    "/social_perception/social_state",

    # Social decisions.
    "/social_navigation/navigation_goal_context",
    "/social_navigation/navigation_state",
    "/social_navigation/robot_motion_state",
    "/social_navigation/decision",
    "/social_navigation/episode_state",
    "/social_navigation/collaborative_decision",

    # Trajectory generation and validation.
    "/social_navigation/trajectory_generation_request",
    "/social_navigation/candidate_trajectory",
    "/social_navigation/trajectory_social_validation",
    "/social_navigation/trajectory_generation_result",
    "/social_navigation/trajectory_generation_status_request",

    # Observation and reassessment.
    "/social_navigation/observation_health",
    "/social_navigation/episode_retirement",
    "/social_navigation/reassessment_request",
    "/social_navigation/reassessment_result",

    # Navigation execution gate.
    "/social_navigation/gate_command_result",

    # Directional constraints and Nav2 filter.
    "/social_navigation/directional_constraint_request",
    "/social_navigation/directional_constraint_status",
    "/social_navigation/directional_filter_info",
    "/social_navigation/directional_keepout_mask",

    # Explainability and speech.
    "/explainability/decision_request",
    "/explainability/context",
    "/explainability/text",
    "/explainability/speech_status",
    "/explainability/speech_cancel_request",
    "/explainability/pc_audio_fallback/request",
    "/explainability/pc_audio_fallback/status",
}


# ================================================================
# HRI TOPICS: CONFIGURATION D
# ================================================================

HRI_TOPICS = {
    # Structured dialogue lifecycle.
    "/hri/dialogue_request",
    "/hri/human_intent",
    "/hri/dialogue_control",
    "/hri/dialogue_control_result",

    # Speech recognition turn control.
    # These are optional if the deployment uses other topic names.
    "/hri/asr_turn_command",
    "/hri/asr_turn_status",
    "/hri/asr_listening_enabled",

    # Dialogue manager.
    "/dialogue_manager/robot_speech",
    "/dialogue_manager/closed_captions",
    "/dialogue_manager/currently_waiting_for_chatbot_response",

    # Human speech and intent.
    "/intents",
    "/humans/persons/tracked",
    "/humans/bodies/tracked",
    "/humans/faces/tracked",
    "/humans/voices/tracked",
    "/humans/voices/pc_user/speech",
}


# High-value topics for reconstructing the system state.
# Their absence is reported prominently, but does not prevent export.
CORE_TOPICS = {
    "A": {
        "/odom",
        "/cmd_vel",
        "/navigate_to_pose/_action/status",
    },
    "C": {
        "/social_navigation/decision",
        "/social_navigation/collaborative_decision",
        "/social_navigation/navigation_state",
        "/explainability/speech_status",
    },
    "D": {
        "/hri/dialogue_request",
        "/hri/human_intent",
        "/hri/dialogue_control",
        "/hri/dialogue_control_result",
        "/social_navigation/directional_constraint_request",
        "/social_navigation/directional_constraint_status",
        "/explainability/speech_status",
    },
}

GRID_TOPICS = {
    "/map",
    "/map_runtime",
    "/global_costmap/costmap",
    "/global_costmap/costmap_updates",
    "/local_costmap/costmap",
    "/local_costmap/costmap_updates",
    "/social_navigation/directional_keepout_mask",
}


def detect_configuration(bag_path):
    """Infer A, C or D from the experiment directory structure."""
    for part in reversed(bag_path.parts):
        if part in {"A", "C", "D"}:
            return part

        match = re.search(
            r"(?:^|_)(" + r"A|C|D" + r")_(?:S1a|S1b|S2|S3|S4|S5)",
            part,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

    return None


def selected_topic_set(configuration):
    """Build the topics relevant to the recorded configuration."""
    if configuration == "A":
        return set(COMMON_TOPICS)

    if configuration == "C":
        return COMMON_TOPICS | SOCIAL_TOPICS

    if configuration == "D":
        return COMMON_TOPICS | SOCIAL_TOPICS | HRI_TOPICS

    # Unknown configuration: extract everything recognized.
    return COMMON_TOPICS | SOCIAL_TOPICS | HRI_TOPICS



def simplify(value):
    """
    Convert ROS messages into strict JSON-compatible data.

    Preserve non-finite floating-point values as explicit strings:
    NaN, Infinity, -Infinity.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (str, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"

        if math.isinf(value):
            return (
                "Infinity"
                if value > 0
                else "-Infinity"
            )

        return value

    if isinstance(value, (bytes, bytearray, array)):
        return [simplify(item) for item in value]

    if hasattr(value, "get_fields_and_field_types"):
        return {
            field: simplify(getattr(value, field))
            for field in value.get_fields_and_field_types()
        }

    # NumPy scalar values, including numpy.float32/float64.
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
            if scalar is not value:
                return simplify(scalar)
        except (ValueError, TypeError):
            pass

    # ROS arrays, including NumPy-backed sequences.
    if hasattr(value, "tolist"):
        return simplify(value.tolist())

    if isinstance(value, dict):
        return {
            str(key): simplify(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [simplify(item) for item in value]

    if isinstance(value, Sequence):
        return [simplify(item) for item in value]

    return str(value)


def compact_grid(msg):
    """
    Summarize a grid without writing every cell.

    Keep the geometry metadata and a fingerprint of the cells.
    This option is for compact timeline inspection, not for
    reconstructing the complete map.
    """
    data = getattr(msg, "data", None)

    if data is None:
        return simplify(msg)

    result = simplify(msg)
    original = result.get("data", [])

    raw = bytes(
        int(value) & 0xFF
        for value in original
    )

    counts = Counter(int(value) for value in original)

    result["data"] = {
        "encoding": "compact_grid_summary",
        "cell_count": len(original),
        "value_counts": {
            str(key): count
            for key, count in sorted(counts.items())
        },
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

    return result


def output_file_path(bag_path, requested_output):
    """Honor -o; otherwise place the output in the bag directory."""
    if requested_output:
        path = Path(requested_output).expanduser().resolve()
        return path

    match = re.search(
        r"_R(\d+)$",
        bag_path.parent.name,
        re.IGNORECASE,
    )

    if match:
        name = f"social_timeline_R{match.group(1)}.jsonl"
    else:
        name = "social_timeline.jsonl"

    return bag_path / name


def main():
    parser = argparse.ArgumentParser(
        description="Extract a G1 social navigation timeline."
    )

    parser.add_argument(
        "bag",
        help="Path to the rosbag2 directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL path; defaults to the bag directory",
    )

    parser.add_argument(
        "--configuration",
        choices=("A", "C", "D"),
        default=None,
        help="Override automatic experimental mode detection",
    )

    parser.add_argument(
        "--compact-grids",
        action="store_true",
        help="Store grid summaries instead of full cell arrays",
    )

    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()

    if not bag_path.is_dir():
        parser.error(f"Bag directory does not exist: {bag_path}")

    configuration = (
        args.configuration
        or detect_configuration(bag_path)
    )

    output_path = output_file_path(
        bag_path,
        args.output,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }

    expected_topics = selected_topic_set(configuration)

    available_topics = expected_topics & set(topic_types)
    missing_topics = expected_topics - set(topic_types)

    required = CORE_TOPICS["A"].copy()

    if configuration in {"C", "D"}:
        required |= CORE_TOPICS["C"]

    if configuration == "D":
        required |= CORE_TOPICS["D"]

    if configuration is None:
        required = set()

    critical_missing = missing_topics & required
    optional_missing = missing_topics - required

    print()
    print("=" * 65)
    print("G1 SOCIAL NAVIGATION TIMELINE EXTRACTION")
    print("=" * 65)
    print(f"Bag:           {bag_path}")
    print(f"Configuration: {configuration or 'UNKNOWN'}")
    print(f"Output:        {output_path}")
    print(f"Compact grids: {args.compact_grids}")
    print()

    print(f"Selected topics found: {len(available_topics)}")

    for topic in sorted(available_topics):
        print(f"  {topic}")

    if critical_missing:
        print("\nWARNING: Core analysis topics absent from bag:")

        for topic in sorted(critical_missing):
            print(f"  {topic}")

    if optional_missing:
        print(
            "\nOther topics absent from bag "
            f"({len(optional_missing)}):"
        )

        for topic in sorted(optional_missing):
            print(f"  {topic}")

    if not available_topics:
        raise RuntimeError(
            "No recognized G1 experiment topics found in bag."
        )

    # Skip unrelated bag topics at the rosbag reader level.
    storage_filter = rosbag2_py.StorageFilter(
        topics=sorted(available_topics)
    )

    reader.set_filter(storage_filter)

    type_cache = {}
    topic_counts = Counter()

    first_timestamp = None
    last_timestamp = None
    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output:

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic not in available_topics:
                continue

            message_type = topic_types[topic]

            if message_type not in type_cache:
                type_cache[message_type] = get_message(
                    message_type
                )

            msg = deserialize_message(
                data,
                type_cache[message_type],
            )

            if first_timestamp is None:
                first_timestamp = timestamp

            last_timestamp = timestamp

            if (
                args.compact_grids
                and topic in GRID_TOPICS
            ):
                message_data = compact_grid(msg)
            else:
                message_data = simplify(msg)

            record = {
                "t": round(
                    (timestamp - first_timestamp) / 1e9,
                    6,
                ),
                "timestamp_ns": timestamp,
                "topic": topic,
                "type": message_type,
                "msg": message_data,
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

            count += 1
            topic_counts[topic] += 1

    duration = (
        (last_timestamp - first_timestamp) / 1e9
        if first_timestamp is not None
        else 0.0
    )

    print()
    print("=" * 65)
    print(f"Messages extracted: {count}")
    print(f"Duration:           {duration:.3f} s")
    print(f"Topics with data:   {len(topic_counts)}")
    print(f"Output:             {output_path}")
    print("=" * 65)

    # Separate lightweight report for detecting silent topics.
    report_path = output_path.with_suffix(
        ".topic_counts.json"
    )

    report = {
        "bag": str(bag_path),
        "configuration": configuration,
        "output": str(output_path),
        "duration_seconds": round(duration, 6),
        "total_messages": count,
        "compact_grids": args.compact_grids,
        "core_topics_missing": sorted(critical_missing),
        "other_topics_missing": sorted(optional_missing),
        "topic_message_counts": dict(
            sorted(topic_counts.items())
        ),
        "available_without_messages": sorted(
            available_topics - set(topic_counts)
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Topic report:       {report_path}")


if __name__ == "__main__":
    main()


"""
 python3 ./scripts/extract_social_timeline_old.py /home/jean/g1_ws/experiments/restaurant_social_awareness/technical/D/S1a/technical_D_S1a_R13/navigation

 python3 ~/g1_ws/scripts/extract_social_timeline_old.py \
  ~/g1_ws/experiments/restaurant_social_awareness/participant/D/S1a/participant_D_S1a_R02/navigation

python3 ~/g1_ws/scripts/extract_social_timeline_old.py \
~/g1_ws/experiments/restaurant_social_awareness/technical/D/S1a/technical_D_S1a_R13/navigation \
--compact-grids
===========================================================================
python3 - <<'PY'
import json
from pathlib import Path

root = Path.home() / (
    "g1_ws/experiments/restaurant_social_awareness/"
    "participant/D/S1a/participant_D_S1a_R02/navigation"
)

source = root / "social_timeline_R02.jsonl"
target = root / "negotiation_R02.jsonl"

topics = {
    "/hri/dialogue_request",
    "/hri/human_intent",
    "/hri/dialogue_control",
    "/hri/dialogue_control_result",
    "/explainability/text",
    "/explainability/speech_status",
    "/social_navigation/collaborative_decision",
    "/social_navigation/episode_state",
    "/social_navigation/navigation_state",
    "/social_navigation/directional_constraint_request",
    "/social_navigation/directional_constraint_status",
    "/social_navigation/gate_command_result",
}

count = 0

with source.open(encoding="utf-8") as inp, \
     target.open("w", encoding="utf-8") as out:

    for line in inp:
        event = json.loads(line)

        if event["topic"] not in topics:
            continue

        out.write(json.dumps(
            event,
            ensure_ascii=False,
        ) + "\n")

        count += 1

print(f"Événements extraits : {count}")
print(f"Fichier : {target}")
PY

python3 plot_navigation_sequence.py negotiation_R02.jsonl --out R02_figures
==================================================================
python3 - <<'PY'
import json
from pathlib import Path
from collections import Counter

root = Path.home() / (
    "g1_ws/experiments/restaurant_social_awareness/"
    "technical/D/S1a/technical_D_S1a_R22/navigation"
)

source = root / "social_timeline_R22.jsonl"
target = root / "nav2_motion_R22.jsonl"

motion_topics = {
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_nav_scaled",
    "/cmd_vel_mux",
    "/cmd_vel_rlsar",
    "/odom",
    "/odom_dog_calibrated",
    "/dog_odom",
    "/dog_odom_2d",
    "/amcl_pose",
    "/plan",
    "/plan_smoothed",
    "/local_plan",
    "/received_global_plan",
    "/transformed_global_plan",
    "/navigate_to_pose_internal/_action/feedback",
    "/navigate_to_pose_internal/_action/status",
    "/compute_path_to_pose/_action/feedback",
    "/compute_path_to_pose/_action/status",
    "/follow_path/_action/feedback",
    "/follow_path/_action/status",
}

grid_topics = {
    "/global_costmap/costmap",
    "/global_costmap/costmap_updates",
    "/local_costmap/costmap",
    "/local_costmap/costmap_updates",
    "/social_navigation/directional_keepout_mask",
}

counts = Counter()
extracted = 0

with source.open(encoding="utf-8") as inp, \
     target.open("w", encoding="utf-8") as out:

    for line in inp:
        event = json.loads(line)
        topic = event.get("topic", "")

        counts[topic] += 1

        if topic in motion_topics:
            out.write(line)
            extracted += 1

print("\n=== TOPICS DE MOUVEMENT ===")
for topic in sorted(motion_topics):
    print(f"{counts[topic]:>8}  {topic}")

print("\n=== COSTMAPS ET MASQUE ===")
for topic in sorted(grid_topics):
    print(f"{counts[topic]:>8}  {topic}")

print("\n=== EXTRACTION ===")
print("Messages extraits :", extracted)
print("Fichier :", target)

missing = [
    topic for topic in motion_topics
    if counts[topic] == 0
]

if missing:
    print(
        "\nATTENTION : certains topics sont absents "
        "du JSONL source. Vérifier le bag original."
    )
PY


"""