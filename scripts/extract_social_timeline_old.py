#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = {
    # Goal / état navigation
    "/goal_pose",
    "/social_navigation/navigation_state",

    # Social / collaboration
    "/social_navigation/decision",
    "/social_navigation/episode_state",
    "/social_navigation/collaborative_decision",
    "/social_navigation/trajectory_generation_request",
    "/social_navigation/candidate_trajectory",
    "/social_navigation/trajectory_social_validation",
    "/social_perception/social_zones",
    "/social_perception/tracked_persons",

    # Explainability / speech
    "/explainability/context",
    "/explainability/text",
    "/explainability/speech_status",

    # Nav2 actions
    "/navigate_to_pose/_action/feedback",
    "/navigate_to_pose/_action/status",
    "/navigate_to_pose_internal/_action/feedback",
    "/navigate_to_pose_internal/_action/status",
    "/follow_path/_action/status",

    # Commandes de mouvement — IMPORTANT
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_nav_scaled",
    "/cmd_vel_mux",
    "/cmd_vel_rlsar",

    # Localisation
    "/odom",
    "/amcl_pose",

    # Plans
    "/plan",
    "/plan_smoothed",
    "/local_plan",
    "/received_global_plan",
    "/transformed_global_plan",

    # Diagnostic ROS
    "/rosout",
}

OPTIONAL_TOPICS = {
    # Dialogue-specific.
    "/hri/dialogue_request",
    "/hri/human_intent",
    "/humans/voices/tracked",
    "/humans/voices/pc_user/speech",
    "/intents",

    # Only used when remote audio fallback is actually exercised.
    "/explainability/pc_audio_fallback/request",
    "/explainability/pc_audio_fallback/status",

    # Nav2 configurations may not publish all diagnostic plan topics.
    "/plan_smoothed",
    "/local_plan",
    "/received_global_plan",
    "/transformed_global_plan",

    # Alternative odometry sources depend on deployment.
    "/odom_dog_calibrated",
    "/dog_odom",
    "/dog_odom_2d",
}


def simplify(value):
    """Convert a ROS message recursively to JSON-compatible data."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [simplify(v) for v in value]

    if hasattr(value, "get_fields_and_field_types"):
        result = {}
        for field in value.get_fields_and_field_types():
            result[field] = simplify(getattr(value, field))
        return result

    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to rosbag2 directory")
    parser.add_argument(
        "-o",
        "--output",
        default="social_timeline.jsonl",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    #output_path = Path(args.output).expanduser().resolve()

    bag_path = Path(args.bag).expanduser().resolve()

    # Extract trial number from the bag directory name.
    # Accepted examples:
    #   C_S1a_trial_2
    #   trial_2
    #   test_trial_03

    match = re.search(
        r"_R(\d+)$",
        bag_path.parent.name,
        re.IGNORECASE,
    )

    if match:
        trial_number = match.group(1)
        output_name = f"social_timeline_R{trial_number}.jsonl"

    else:
        # Fallback when the bag directory has no explicit trial number.
        output_name = "social_timeline.jsonl"

    output_path = bag_path / output_name

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

    selected_topics = TOPICS.intersection(topic_types)

    print("Topics found:")
    for topic in sorted(selected_topics):
        print(f"  {topic}: {topic_types[topic]}")

    missing = TOPICS.difference(topic_types)

    required_missing = (
        missing
        - OPTIONAL_TOPICS
    )

    optional_missing = (
        missing
        & OPTIONAL_TOPICS
    )

    if required_missing:
        print("\nREQUIRED topics absent from bag:")
        for topic in sorted(required_missing):
            print(f"  {topic}")

    if optional_missing:
        print("\nOptional / mode-dependent topics absent from bag:")
        for topic in sorted(optional_missing):
            print(f"  {topic}")

    type_cache = {}

    count = 0
    first_timestamp = None

    with output_path.open("w", encoding="utf-8") as output:
        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic not in selected_topics:
                continue

            if first_timestamp is None:
                first_timestamp = timestamp

            msg_type_name = topic_types[topic]

            if msg_type_name not in type_cache:
                type_cache[msg_type_name] = get_message(msg_type_name)

            msg = deserialize_message(
                data,
                type_cache[msg_type_name],
            )

            record = {
                "t": round(
                    (timestamp - first_timestamp) / 1e9,
                    6,
                ),
                "timestamp_ns": timestamp,
                "topic": topic,
                "type": msg_type_name,
                "msg": simplify(msg),
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print()
    print(f"Messages extracted: {count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

#python3 ./scripts/extract_social_timeline.py /home/jean/g1_ws/experiments/restaurant_social_awareness/technical/D/S1a/technical_D_S1a_R05/navigation