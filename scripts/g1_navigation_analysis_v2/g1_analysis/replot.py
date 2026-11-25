"""Regenerate paper figures from saved metrics, without rereading ROS bags."""
import csv
import json
from pathlib import Path
from .aggregate import summarize, contrasts
from .command_metrics import command_metrics
from .discovery import annotations, write_annotation_template
from .outcomes import ALLOWED
from .reporting import csv_file, json_file
from .paper_plots import campaign_figures

TEXT_FIELDS = {'trial_id','trial_number','configuration','campaign','scenario','bag_path',
               'trial','config','status','goal_reference','data_source','original_trial_name','automatic_outcome','outcome','outcome_source','goal_id',
               'target_source','window_source','settings_id','command_detection_topic'}


def load_metrics(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        if 'trial_id' not in row:
            row['trial_id'] = f"{row['campaign']}__{row['config']}__trial_{row['trial']}"
            row['trial_number'] = row['trial']
            row['configuration'] = row['config']
            row['speed_mps'] = row['speed_m_s']
            row['scenario'] = {'Fire doors':'corridor_with_doors','Office corridor':'corridor'}.get(row['scenario'],row['scenario'])
            row['outcome'] = {'Reached':'SUCCEEDED','Aborted':'ABORTED','Canceled':'CANCELED'}.get(row['status'],'UNKNOWN')
            for key in ('raw_odom_amcl_rmse_m','raw_odom_amcl_final_m','used_odom_amcl_rmse_m','used_odom_amcl_final_m','initial_plan_deviation_rmse_m','commanded_stop_count'):
                row['legacy_'+key] = row.get(key,'')
            row['outcome_source'] = 'original_csv_annotation'
        for key,value in list(row.items()):
            if key not in TEXT_FIELDS:
                if value=='':
                    row[key]=None
                else:
                    try:
                        row[key]=float(value)
                    except ValueError:
                        pass
        row['outcome']=row.get('outcome') or 'UNKNOWN'
    if len({r['trial_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate trial IDs in input CSV.')
    return rows


def load_series(folder):
    streams={}
    for name in ('cmd_vel','cmd_vel_nav_scaled','cmd_vel_mux','cmd_vel_rlsar'):
        path=folder/('series_'+name+'.csv')
        if not path.is_file():
            continue
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        streams['/'+name]=[dict(t=float(r['t']),vx=float(r['vx']),vy=float(r['vy']),wz=float(r['wz'])) for r in rows]
    return streams


def export_replot_annotations(path, rows, manual):
    """Export effective trial outcomes with their provenance."""
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

    with Path(path).open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for result in rows:
            trial_id = result["trial_id"]
            annotation = manual.get(trial_id, {})

            record = {
                field: annotation.get(field, "") or ""
                for field in fields
            }

            record["trial_id"] = trial_id
            record["outcome"] = result["outcome"]

            record["evidence"] = (
                result.get("outcome_evidence")
                or annotation.get("evidence")
                or (
                    "Inherited from saved trial_metrics.csv; "
                    f"source={result.get('outcome_source', 'unknown')}"
                )
            )

            record["goal_id"] = (
                annotation.get("goal_id")
                or result.get("goal_id")
                or ""
            )

            record["settings_id"] = (
                annotation.get("settings_id")
                or result.get("settings_id")
                or ""
            )

            writer.writerow(record)


def export_replot_success_counts(path, rows):
    """Aggregate resolved trial outcomes without imputing UNKNOWN."""
    groups = {}

    known_outcomes = {
        "SUCCEEDED",
        "ABORTED",
        "CANCELED",
        "FAILED",
        "TIMEOUT",
    }

    for row in rows:
        key = (
            str(row["scenario"]),
            round(float(row["speed_mps"]), 2),
            str(row["configuration"]),
        )

        groups.setdefault(key, []).append(
            str(row.get("outcome") or "UNKNOWN").strip().upper()
        )

    records = []

    for key in sorted(groups):
        outcomes = groups[key]

        if any(status not in known_outcomes for status in outcomes):
            print(
                "WARNING: incomplete outcome group excluded "
                f"from success counts: {key}"
            )
            continue

        scenario, speed, configuration = key

        records.append({
            "scenario": scenario,
            "speed_mps": speed,
            "configuration": configuration,
            "successes": outcomes.count("SUCCEEDED"),
            "trials": len(outcomes),
            "source": (
                "Resolved individual trial outcomes "
                "from replot input and documented annotations"
            ),
        })

    fields = [
        "scenario",
        "speed_mps",
        "configuration",
        "successes",
        "trials",
        "source",
    ]

    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    return Path(path)


def run_replot(source, output, config, annotation_path=None):
    """
    Recreate publication figures from saved metrics.

    Apply documented per-trial annotations, preserve terminal
    outcome distinctions, regenerate summaries and success counts,
    and never require ROS bags for scalar replotting.
    """
    source = Path(source)
    output = Path(output)

    csv_path = (
        source / "trial_metrics.csv"
        if source.is_dir()
        else source
    )

    rows = load_metrics(csv_path)
    manual = annotations(annotation_path)

    available_ids = {
        row["trial_id"] for row in rows
    }

    invalid = set(manual) - available_ids

    if invalid:
        print(
            f"Annotation file includes {len(invalid)} trial IDs "
            "outside this input; these are not added to metrics."
        )

    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory must be empty."
        )

    output.mkdir(parents=True, exist_ok=True)

    methods_source = (
        Path(__file__).resolve().parents[1] / "METHODS.md"
    )

    (output / "METHODS.md").write_text(
        methods_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    recovered = 0
    corrected_outcomes = 0

    for row in rows:
        trial_id = row["trial_id"]

        # --------------------------------------------------
        # Recover command metrics from saved command series
        # --------------------------------------------------
        folder = (
            source / "trials" / trial_id
            if source.is_dir()
            else None
        )

        if folder is not None and folder.is_dir():
            streams = load_series(folder)

            if streams:
                recovered_metrics = command_metrics(
                    streams,
                    config["analysis"],
                )

                row.update({
                    key: value
                    for key, value in recovered_metrics.items()
                    if key not in row or row[key] is None
                })

                recovered += 1

        # --------------------------------------------------
        # Resolve documented outcome annotations
        # --------------------------------------------------
        annotation = manual.get(trial_id, {})

        original_outcome = str(
            row.get("outcome") or "UNKNOWN"
        ).strip().upper()

        state = str(
            annotation.get("outcome") or ""
        ).strip().upper()

        if state:
            if (
                state not in ALLOWED
                or not str(
                    annotation.get("evidence") or ""
                ).strip()
            ):
                raise ValueError(
                    "Outcome annotation needs a valid "
                    "state and evidence: " + trial_id
                )

            # Preserve an existing recorded cancellation.
            # In particular, g1_nav CANCELED must not
            # silently become ABORTED.
            if (
                original_outcome == "CANCELED"
                and state != "CANCELED"
            ):
                raise ValueError(
                    "Refusing to overwrite recorded "
                    f"CANCELED outcome: {trial_id}"
                )

            if original_outcome != state:
                corrected_outcomes += 1

            row.update(
                outcome=state,
                outcome_source="annotation",
                outcome_evidence=annotation["evidence"],
            )

        if str(
            annotation.get("settings_id") or ""
        ).strip():
            row["settings_id"] = annotation["settings_id"]

        # Scalar replotting cannot recompute time windows
        # or spatial goal references.
        if any(
            str(annotation.get(key) or "").strip()
            for key in (
                "start_s",
                "end_s",
                "goal_x",
                "goal_y",
                "goal_yaw_rad",
                "goal_id",
            )
        ):
            raise ValueError(
                "Time/goal annotations require a full "
                "bag analysis, not --results: " + trial_id
            )

    # ------------------------------------------------------
    # Verify outcome integrity
    # ------------------------------------------------------
    counts = {}

    for row in rows:
        configuration = row["configuration"]
        status = row["outcome"]

        counts.setdefault(configuration, {})
        counts[configuration][status] = (
            counts[configuration].get(status, 0) + 1
        )

    print("\nOUTCOMES BY CONFIGURATION")

    for configuration in config["configurations"]:
        print(
            f"{configuration}: "
            f"{counts.get(configuration, {})}"
        )

    # ------------------------------------------------------
    # Recalculate all aggregate analysis results
    # ------------------------------------------------------
    summaries = summarize(rows)

    csv_file(
        output / "trial_metrics.csv",
        rows,
    )

    csv_file(
        output / "group_summary.csv",
        summaries,
    )

    csv_file(
        output / "command_ablation.csv",
        contrasts(rows, config["analysis"]),
    )

    # Produce a reference annotation template.
    write_annotation_template(
        output / "annotations_template.csv",
        rows,
    )

    # Export effective, not merely original, annotations.
    export_replot_annotations(
        output / "annotations_used.csv",
        rows,
        manual,
    )

    # Derive success counts from resolved individual results.
    generated_counts_path = export_replot_success_counts(
        output / "success_counts.csv",
        rows,
    )

    json_file(
        output / "replot_provenance.json",
        {
            "source": str(source.resolve()),
            "command_series_recovered": recovered,
            "corrected_outcomes": corrected_outcomes,
            "outcomes_unknown": sum(
                row["outcome"] == "UNKNOWN"
                for row in rows
            ),
            "analysis_config": config,
            "metric_definitions": (
                "Inherited scalar metrics from source CSV; "
                "command metrics recomputed where cached "
                "series exist; outcomes updated only through "
                "documented annotations."
            ),
        },
    )

    # ------------------------------------------------------
    # Regenerate publication figures
    # ------------------------------------------------------
    campaign_figures(
        rows,
        summaries,
        config,
        output / "figures",
        success_counts_path=generated_counts_path,
    )

    # Dedicated success-count figures, generated from the
    # same resolved trial outcomes.
    from .aggregate_outcomes import export_declared_counts

    export_declared_counts(
        generated_counts_path,
        output / "figures",
    )

    print(
        f"\nPaper figures regenerated for {len(rows)} trials."
    )

    print(
        f"Command series recovered: {recovered}"
    )

    print(
        f"Outcome corrections applied: {corrected_outcomes}"
    )

    print(
        "Unknown outcomes: "
        f"{sum(row['outcome'] == 'UNKNOWN' for row in rows)}"
    )

    print(
        f"Generated success counts: {generated_counts_path}"
    )

    return 0
