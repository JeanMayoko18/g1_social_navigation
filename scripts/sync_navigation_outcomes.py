
#!/usr/bin/env python3
"""Synchronize per-trial outcomes and aggregate success counts."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# Experimenter-confirmed successful trial numbers.
# All other trials in these groups were reported as unsuccessful.
ODOM_SUCCESSES = {
    ("corridor_with_doors", 0.12): {3},
    ("corridor_with_doors", 0.30): {1, 3, 4, 5},
    ("corridor", 0.12): {1, 2},
    ("corridor", 0.30): {1, 2, 3, 5},
}

ODOM_TRIAL_COUNTS = {
    ("corridor_with_doors", 0.12): 3,
    ("corridor_with_doors", 0.30): 5,
    ("corridor", 0.12): 3,
    ("corridor", 0.30): 5,
}

TRIAL_PATTERN = re.compile(
    r"^L-shaped_(corridor_with_doors|corridor)"
    r"_start_goal_1(_v12)?__(.+)__trial_(\d+)$"
)

VALID_OUTCOMES = {
    "SUCCEEDED",
    "ABORTED",
    "CANCELED",
    "FAILED",
    "TIMEOUT",
}


def parse_trial_id(trial_id):
    match = TRIAL_PATTERN.fullmatch(trial_id)

    if match is None:
        raise ValueError(f"Unsupported trial_id: {trial_id}")

    scenario, v12, configuration, trial_number = match.groups()

    speed = 0.12 if v12 else 0.30

    return (
        scenario,
        speed,
        configuration,
        int(trial_number),
    )


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, columns, rows):
    # Write a temporary file first to avoid partial output.
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def synchronize(annotations_path, counts_path):
    columns, annotations = read_csv(annotations_path)

    if not columns or not {"trial_id", "outcome"} <= set(columns):
        raise ValueError("Invalid navigation annotations CSV.")

    original_other_outcomes = {}
    odom_seen = defaultdict(set)
    groups = defaultdict(list)
    seen_ids = set()

    for row in annotations:
        trial_id = row["trial_id"].strip()

        if trial_id in seen_ids:
            raise ValueError(f"Duplicate trial: {trial_id}")

        seen_ids.add(trial_id)

        scenario, speed, configuration, index = parse_trial_id(
            trial_id
        )

        key = (scenario, speed, configuration)

        current_outcome = row["outcome"].strip()

        if configuration == "Odometry-only":
            group_key = (scenario, speed)

            if group_key not in ODOM_SUCCESSES:
                raise ValueError(
                    f"Unexpected Odometry-only group: {group_key}"
                )

            if not 1 <= index <= ODOM_TRIAL_COUNTS[group_key]:
                raise ValueError(
                    f"Unexpected trial number: {trial_id}"
                )

            odom_seen[group_key].add(index)

            expected = (
                "SUCCEEDED"
                if index in ODOM_SUCCESSES[group_key]
                else "ABORTED"
            )

            if current_outcome not in {
                "",
                "UNKNOWN",
                expected,
            }:
                raise ValueError(
                    f"Conflicting status for {trial_id}: "
                    f"{current_outcome} != {expected}"
                )

            row["outcome"] = expected

            if "evidence" in row:
                row["evidence"] = (
                    "Experimenter-confirmed per-trial "
                    "outcomes, 2026-10-22"
                )

        else:
            # Preserve calibrated, dog, and g1_nav exactly.
            original_other_outcomes[trial_id] = current_outcome

        outcome = row["outcome"].strip()

        if outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Missing or invalid outcome: "
                f"{trial_id}: {outcome}"
            )

        groups[key].append(outcome)

    # All 16 Odometry-only trials must be present.
    for key, total in ODOM_TRIAL_COUNTS.items():
        expected_indices = set(range(1, total + 1))

        if odom_seen[key] != expected_indices:
            raise ValueError(
                f"Incomplete Odometry-only trials for {key}: "
                f"{sorted(odom_seen[key])}"
            )

    # Explicitly protect the recorded g1_nav cancellation.
    canceled_trial = (
        "L-shaped_corridor_with_doors_start_goal_1"
        "__g1_nav__trial_02"
    )

    if original_other_outcomes.get(canceled_trial) != "CANCELED":
        raise ValueError(
            "The original g1_nav CANCELED status "
            "is missing or modified."
        )

    # Retain existing group order and provenance.
    count_columns, old_counts = read_csv(counts_path)

    required = {
        "scenario",
        "speed_mps",
        "configuration",
        "successes",
        "trials",
        "source",
    }

    if not count_columns or not required <= set(count_columns):
        raise ValueError("Invalid success counts CSV.")

    updated_counts = []
    seen_groups = set()

    for row in old_counts:
        key = (
            row["scenario"].strip(),
            round(float(row["speed_mps"]), 2),
            row["configuration"].strip(),
        )

        if key in seen_groups:
            raise ValueError(f"Duplicate group: {key}")

        seen_groups.add(key)

        if key not in groups:
            raise ValueError(
                f"Missing annotation group: {key}"
            )

        outcomes = groups[key]

        successes = outcomes.count("SUCCEEDED")
        total = len(outcomes)

        if int(row["trials"]) != total:
            raise ValueError(
                f"Trial count mismatch for {key}"
            )

        # For configurations other than Odometry-only,
        # the original aggregates must still agree.
        if (
            key[2] != "Odometry-only"
            and int(row["successes"]) != successes
        ):
            raise ValueError(
                f"Existing success count mismatch: {key}"
            )

        row["successes"] = str(successes)
        row["trials"] = str(total)

        if key[2] == "Odometry-only":
            row["source"] = (
                "Experimenter-confirmed individual "
                "trial annotations, 2026-10-22"
            )

        updated_counts.append(row)

    if seen_groups != set(groups):
        raise ValueError(
            f"Aggregate groups missing: {set(groups) - seen_groups}"
        )

    # Only write after all validations pass.
    write_csv(annotations_path, columns, annotations)
    write_csv(counts_path, count_columns, updated_counts)

    print("\nOUTCOMES BY CONFIGURATION")

    for configuration in (
        "calibrated",
        "Odometry-only",
        "dog",
        "g1_nav",
    ):
        selected = [
            row["outcome"]
            for row in annotations
            if parse_trial_id(row["trial_id"])[2]
            == configuration
        ]

        counts = {
            status: selected.count(status)
            for status in (
                "SUCCEEDED",
                "ABORTED",
                "CANCELED",
            )
            if selected.count(status)
        }

        print(f"{configuration}: {counts}")

    print("\nGenerated:")
    print(annotations_path)
    print(counts_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("navigation_annotations.csv"),
    )

    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("success_counts.csv"),
    )

    args = parser.parse_args()

    synchronize(
        args.annotations,
        args.counts,
    )


if __name__ == "__main__":
    main()

"""
python3 sync_navigation_outcomes.py \
  --annotations g1_navigation_analysis_v2/config/navigation_annotations.csv \
  --counts g1_navigation_analysis_v2/config/success_counts.csv

"""