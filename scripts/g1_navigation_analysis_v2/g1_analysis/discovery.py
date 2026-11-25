"""Resolve explicit campaign metadata; never infer speed from an arbitrary label."""
import csv
from pathlib import Path
import re

# Experimenter-confirmed individual outcomes for Odometry-only.
ODOM_ONLY_RESULTS = {
    "L-shaped_corridor_with_doors_start_goal_1_v12": {
        1: "ABORTED",
        2: "ABORTED",
        3: "SUCCEEDED",
    },
    "L-shaped_corridor_with_doors_start_goal_1": {
        1: "SUCCEEDED",
        2: "ABORTED",
        3: "SUCCEEDED",
        4: "SUCCEEDED",
        5: "SUCCEEDED",
    },
    "L-shaped_corridor_start_goal_1_v12": {
        1: "SUCCEEDED",
        2: "SUCCEEDED",
        3: "ABORTED",
    },
    "L-shaped_corridor_start_goal_1": {
        1: "SUCCEEDED",
        2: "SUCCEEDED",
        3: "SUCCEEDED",
        4: "ABORTED",
        5: "SUCCEEDED",
    },
}

def discover(root, config):
    trials = []
    for campaign, metadata in config['campaigns'].items():
        for configuration in config['configurations']:
            folder = root / campaign / configuration
            for bag in sorted(folder.glob('*/navigation')):
                if not (bag / 'metadata.yaml').is_file():
                    continue
                match = re.search(r'_trial_(\d+)$', bag.parent.name)
                if not match:
                    continue
                trials.append(dict(trial_id=f'{campaign}__{configuration}__trial_{match[1]}', original_trial_name=bag.parent.name, campaign=campaign,
                                   configuration=configuration, trial_number=match[1],
                                   bag_path=str(bag.resolve()), **metadata))
    if len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('Duplicate trial IDs: resolve before analysis.')
    return trials


def annotations(path):
    """Load annotations and apply confirmed Odometry-only outcomes."""
    if path is None:
        return {}

    with Path(path).open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    result = {}

    names = {
        "Reached": "SUCCEEDED",
        "Aborted": "ABORTED",
        "Canceled": "CANCELED",
        "Cancelled": "CANCELED",
        "Unknown": "UNKNOWN",
    }

    for row in rows:
        if "trial_id" not in row:
            required = (
                "campaign",
                "config",
                "trial",
                "status",
            )

            if not all(k in row for k in required):
                raise ValueError(
                    "Annotations require trial_id/outcome "
                    "or campaign/config/trial/status."
                )

            row = dict(
                trial_id=(
                    f"{row['campaign']}__"
                    f"{row['config']}__trial_{row['trial']}"
                ),
                outcome=names.get(
                    row["status"],
                    row["status"].upper(),
                ),
                evidence="User annotation: " + str(path),
            )

        trial_id = row["trial_id"].strip()

        if trial_id in result:
            raise ValueError(
                "Duplicate annotation trial ID: " + trial_id
            )

        # Apply only explicitly documented Odometry-only results.
        if "__Odometry-only__trial_" in trial_id:
            campaign, number = trial_id.rsplit(
                "__Odometry-only__trial_", 1
            )

            try:
                trial_number = int(number)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Odometry-only trial: {trial_id}"
                ) from exc

            outcomes = ODOM_ONLY_RESULTS.get(campaign)

            if outcomes is None or trial_number not in outcomes:
                raise ValueError(
                    f"Missing confirmed outcome for {trial_id}"
                )

            expected = outcomes[trial_number]

            existing = str(
                row.get("outcome") or ""
            ).strip().upper()

            if existing not in {
                "",
                "UNKNOWN",
                expected,
            }:
                raise ValueError(
                    f"Conflicting outcome for {trial_id}: "
                    f"{existing} != {expected}"
                )

            row["outcome"] = expected
            row["evidence"] = (
                "Experimenter-confirmed individual outcome, "
                "2026-10-22"
            )

        result[trial_id] = row

    return result


def write_annotation_template(path, trials):
    """Create annotations with confirmed Odometry-only outcomes."""

    path = Path(path)

    fields = [
        "trial_id",
        "goal_id",
        "outcome",
        "evidence",
        "start_s",
        "end_s",
        "goal_x",
        "goal_y",
        "goal_yaw_rad",
        "settings_id",
        "notes",
    ]

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            row = {"trial_id": trial["trial_id"]}

            if trial["configuration"] == "Odometry-only":
                campaign = trial["campaign"]
                trial_number = int(trial["trial_number"])

                outcomes = ODOM_ONLY_RESULTS.get(campaign)

                if outcomes is None or trial_number not in outcomes:
                    raise ValueError(
                        "Missing confirmed Odometry-only outcome: "
                        f"{trial['trial_id']}"
                    )

                row["outcome"] = outcomes[trial_number]
                row["evidence"] = (
                    "Experimenter-confirmed individual outcome, "
                    "2026-10-22"
                )

            writer.writerow(row)