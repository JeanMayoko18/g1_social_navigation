#!/usr/bin/env python3

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message


TOPICS = {
    "/social_navigation/decision": "events",
    "/social_navigation/episode_state": "episode_states",
    "/social_navigation/navigation_state": "navigation_states",
    "/explainability/decision_request": "explanation_requests",
    "/explainability/context": "contexts",
    "/explainability/text": "explanations",
    "/explainability/speech_status": "speech_status",
}


def flatten(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False)
        else:
            result[name] = value
    return result


def stamp_ns(stamp):
    if not isinstance(stamp, dict):
        return None
    if "sec" not in stamp or "nanosec" not in stamp:
        return None
    value = int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])
    return value if value != 0 else None


def write_csv(path, rows):
    fields = list(dict.fromkeys(
        ["bag_time_ns", "bag_elapsed_s"]
        + [key for row in rows for key in row]
    ))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    metadata_file = bag / "metadata.yaml"
    if not metadata_file.is_file():
        raise SystemExit(f"metadata.yaml introuvable : {bag}")

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    info = metadata["rosbag2_bagfile_information"]
    if info["storage_identifier"] != "sqlite3":
        raise SystemExit("Ce script attend un bag SQLite3.")

    records = []
    classes = {}
    first_bag_time = None

    placeholders = ",".join("?" for _ in TOPICS)
    query = f"""
        SELECT m.timestamp, t.name, t.type, t.serialization_format, m.data
        FROM messages AS m
        JOIN topics AS t ON m.topic_id = t.id
        WHERE t.name IN ({placeholders})
        ORDER BY m.timestamp, m.id
    """

    for relative_path in info["relative_file_paths"]:
        db = (bag / relative_path).resolve()
        connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
        try:
            first = connection.execute(
                "SELECT MIN(timestamp) FROM messages"
            ).fetchone()[0]
            if first is not None:
                first_bag_time = (
                    first if first_bag_time is None
                    else min(first_bag_time, first)
                )

            for timestamp, topic, type_name, serialization, payload in (
                connection.execute(query, tuple(TOPICS))
            ):
                if serialization != "cdr":
                    raise RuntimeError(
                        f"Sérialisation non prise en charge : {serialization}"
                    )
                if type_name not in classes:
                    classes[type_name] = get_message(type_name)
                message = deserialize_message(payload, classes[type_name])
                data = message_to_ordereddict(message)
                records.append((timestamp, topic, data))
        finally:
            connection.close()

    if not records:
        raise SystemExit("Aucun message des topics sélectionnés.")

    records.sort(key=lambda item: item[0])

    output = (
        args.output.expanduser().resolve()
        if args.output else bag.parent / "social_export"
    )
    # Évite d'écraser une analyse précédente.
    output.mkdir(parents=True, exist_ok=False)

    tables = {name: [] for name in TOPICS.values()}

    for timestamp, topic, data in records:
        row = {
            "bag_time_ns": timestamp,
            "bag_elapsed_s": (timestamp - first_bag_time) / 1e9,
            **flatten(data),
        }

        if topic == "/explainability/text":
            published = stamp_ns(data.get("header", {}).get("stamp"))
            context = stamp_ns(data.get("context_stamp"))
            row["context_to_text_s"] = ""
            row["latency_status"] = "missing_or_zero_stamp"

            if published is not None and context is not None:
                delta = (published - context) / 1e9
                if delta >= 0:
                    row["context_to_text_s"] = delta
                    row["latency_status"] = "requires_clock_validation"
                else:
                    row["latency_status"] = "negative_check_clocks_or_stamps"

        tables[TOPICS[topic]].append(row)

    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)

    texts = tables["explanations"]
    social_texts = [
        row for row in texts
        if row.get("speech_kind") == "SOCIAL_EXPLANATION"
    ]
    write_csv(output / "social_explanations.csv", social_texts)

    with (output / "explanations.txt").open("w", encoding="utf-8") as stream:
        stream.write(
            "EXPORT DES TEXTES ET DE LEURS MESSAGES ASSOCIES\n"
            "Temps du bag et timestamps source sont conservés séparément.\n"
            "Les associations utilisent decision_id ; aucune ligne n'est "
            "supprimée en cas de répétition.\n"
            "context_to_text_s n'est pas le temps d'inférence LLM seul.\n"
        )

        for index, text in enumerate(texts, 1):
            decision_id = text.get("decision_id", "")
            stream.write(f"\n{'=' * 70}\nTEXTE {index}\n")
            stream.write(json.dumps(text, ensure_ascii=False, indent=2))
            stream.write("\n")

            for name in (
                "events", "explanation_requests", "contexts", "speech_status"
            ):
                matches = [
                    row for row in tables[name]
                    if decision_id and row.get("decision_id") == decision_id
                ]
                stream.write(f"\n{name} : {len(matches)} correspondance(s)\n")
                for row in matches:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, indent=2) + "\n"
                    )

    summary = {
        "bag": str(bag),
        "counts": {name: len(rows) for name, rows in tables.items()},
        "speech_kind_counts": dict(Counter(
            row.get("speech_kind") or "UNCLASSIFIED" for row in texts
        )),
        "social_explanation_count": len(social_texts),
        "social_provider_counts": dict(Counter(
            row.get("provider", "") for row in social_texts
        )),
        "social_cache_count": sum(
            bool(row.get("used_cache")) for row in social_texts
        ),
        "social_fallback_count": sum(
            bool(row.get("used_fallback")) for row in social_texts
        ),
        "speech_state_counts": dict(Counter(
            row.get("state", "") for row in tables["speech_status"]
        )),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nExport terminé : {output}")


if __name__ == "__main__":
    main()
