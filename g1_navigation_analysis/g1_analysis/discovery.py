"""Discover navigation trials and manage explicit trial annotations."""

import csv
import re
from pathlib import Path


def discover(root, config):
    """Discover bags with campaign-scoped unique trial identifiers."""
    root = Path(root).expanduser()
    trials = []
    seen = {}

    for campaign, metadata in config["campaigns"].items():
        for configuration in config["configurations"]:
            directory = root / campaign / configuration

            for bag in sorted(directory.glob("*/navigation")):
                if not (bag / "metadata.yaml").is_file():
                    continue

                original_trial_name = bag.parent.name
                match = re.search(
                    r"_trial_(\d+)$",
                    original_trial_name,
                )

                if match is None:
                    print(
                        "[WARNING] Skipping unsupported trial "
                        f"directory: {bag.parent}"
                    )
                    continue

                #trial_number = int(match.group(1))
                # Preserve the original identifier: "004" and "04" are distinct.
                trial_number = match.group(1)

                trial_id = (
                    f"{campaign}__"
                    f"{configuration}__"
                    f"trial_{trial_number}"
                )

                resolved_bag = str(bag.resolve())

                if trial_id in seen:
                    raise ValueError(
                        "Multiple bags have the same campaign, "
                        "configuration, and trial number.\n"
                        f"Trial ID: {trial_id}\n"
                        f"First bag: {seen[trial_id]}\n"
                        f"Second bag: {resolved_bag}"
                    )

                seen[trial_id] = resolved_bag

                trials.append({
                    **metadata,
                    "trial_id": trial_id,
                    "campaign": campaign,
                    "configuration": configuration,
                    "trial_number": trial_number,
                    "original_trial_name": original_trial_name,
                    "bag_path": resolved_bag,
                })

    return trials


def annotations(path):
    """Load optional annotations indexed by unique trial ID.

    Empty cells remain empty strings. Duplicate trial IDs are rejected
    to prevent one annotation silently replacing another.
    """
    if path is None:
        return {}

    path = Path(path).expanduser()

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        reader = csv.DictReader(stream)

        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            raise ValueError(
                f"Annotation file requires a 'trial_id' column: {path}"
            )

        result = {}

        for line_number, row in enumerate(reader, start=2):
            # Ignore completely empty rows.
            if not any(
                str(value).strip()
                for value in row.values()
                if value is not None
            ):
                continue

            if None in row:
                raise ValueError(
                    f"Malformed CSV row at {path}:{line_number}. "
                    "Check delimiters and quote fields containing commas."
                )

            cleaned = {
                key: (value or "").strip()
                for key, value in row.items()
            }

            trial_id = cleaned["trial_id"]

            if not trial_id:
                raise ValueError(
                    f"Missing trial_id at {path}:{line_number}"
                )

            if trial_id in result:
                raise ValueError(
                    f"Duplicate annotation trial_id: {trial_id}"
                )

            result[trial_id] = cleaned

    return result


def write_annotation_template(path, trials):
    """Write an empty annotation template without overwriting a file.

    start_s and end_s use absolute timestamps in seconds.
    Manual goal coordinates use the map frame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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

    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            writer.writerow({
                "trial_id": trial["trial_id"],
            })