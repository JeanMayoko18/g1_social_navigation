"""Orchestration only: discovery -> reader -> metrics -> aggregate -> exports."""
import argparse
import hashlib
import sys
import csv
from pathlib import Path
import yaml
from . import __version__
from .discovery import discover, annotations, write_annotation_template
from .reader import read_bag
from .metrics import analyze
from .aggregate import summarize, contrasts
from .reporting import json_file, csv_file, report



def export_annotations_used(path, trials, manual):
    """
    Export the effective annotations used by the analysis.

    Preserve original annotations for all configurations,
    including g1_nav CANCELED.
    """
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
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            trial_id = trial["trial_id"]
            annotation = manual.get(trial_id, {})

            row = {
                field: annotation.get(field, "")
                for field in fields
            }
            row["trial_id"] = trial_id
            writer.writerow(row)


def export_resolved_success_counts(path, trials, results):
    """
    Derive aggregate counts from resolved trial outcomes.

    Never guess missing trial outcomes. Only export complete
    groups with known outcomes.
    """
    expected = {}
    for trial in trials:
        key = (
            trial["scenario"],
            round(float(trial["speed_mps"]), 2),
            trial["configuration"],
        )
        expected.setdefault(key, set()).add(
            trial["trial_id"]
        )

    observed = {}
    for result in results:
        key = (
            result["scenario"],
            round(float(result["speed_mps"]), 2),
            result["configuration"],
        )
        observed.setdefault(key, []).append(result)

    rows = []
    skipped = []

    for key in sorted(expected):
        group = observed.get(key, [])
        expected_ids = expected[key]

        observed_ids = {
            row["trial_id"] for row in group
        }

        outcomes = [
            str(row.get("outcome") or "UNKNOWN")
            .strip().upper()
            for row in group
        ]

        if (
            len(group) != len(expected_ids)
            or observed_ids != expected_ids
            or any(
                status not in {
                    "SUCCEEDED",
                    "ABORTED",
                    "CANCELED",
                    "FAILED",
                    "TIMEOUT",
                }
                for status in outcomes
            )
        ):
            skipped.append(key)
            continue

        scenario, speed, configuration = key

        rows.append({
            "scenario": scenario,
            "speed_mps": speed,
            "configuration": configuration,
            "successes": outcomes.count("SUCCEEDED"),
            "trials": len(group),
            "source": (
                "Resolved individual trial outcomes "
                "(action status or documented annotation)"
            ),
        })

    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario",
                "speed_mps",
                "configuration",
                "successes",
                "trials",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for key in skipped:
        print(
            "WARNING: success counts not exported "
            f"for incomplete/unknown group: {key}",
            file=sys.stderr,
        )

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'g1_ws/experiments')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/campaign.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotations', '--status-csv', dest='annotations', type=Path, default=Path(__file__).resolve().parents[1]/'config/navigation_annotations.csv')
    parser.add_argument('--success-counts',type=Path,help='Optional aggregate outcome CSV; plotted separately, never assigned to trial IDs.')
    parser.add_argument('--results', '--result', dest='results', type=Path, help='Replot saved metrics without ROS or bag discovery.')
    parser.add_argument('--inventory-only',action='store_true',help='No ROS imports or bag deserialization.')
    parser.add_argument('--trial',action='append',help='Exact trial ID; repeat to select multiple trials.')
    parser.add_argument('--no-trial-plots',action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['analysis']['time_basis'] != 'bag':
        parser.error('Version 2 publication metrics require time_basis: bag.')
    if args.results:
        from .replot import run_replot
        return run_replot(args.results.expanduser(), args.output.expanduser(),
                          config, args.annotations, args.success_counts)
    trials = discover(args.root.expanduser(), config)

    if args.trial:
        missing = set(args.trial)-{t['trial_id'] for t in trials}
        if missing:
            parser.error('Unknown trial IDs: '+', '.join(sorted(missing)))
        trials = [t for t in trials if t['trial_id'] in args.trial]
    if not trials:
        parser.error('No navigation/metadata.yaml found in configured campaign directories.')
    # Refuse stale result mixing; outputs are reproducible new runs.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty. Choose a new run directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'METHODS.md').write_text((Path(__file__).resolve().parents[1]/'METHODS.md').read_text())
    csv_file(args.output/'inventory.csv',trials)
    write_annotation_template(args.output/'annotations_template.csv',trials)
    json_file(args.output/'run_config.json',dict(version=__version__,configuration=config,
              source_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
              command_line=sys.argv,root=str(args.root.resolve())))
    print(f'Discovered {len(trials)} trials.')
    if args.inventory_only:
        return 0
    manual = annotations(args.annotations)

    export_annotations_used(
        args.output / "annotations_used.csv",
        trials,
        manual,
    )
    from .plots import trial_figures
    from .paper_plots import campaign_figures
    results,failures = [],[]
    for index,trial in enumerate(trials,1):
        print(f"[{index}/{len(trials)}] {trial['trial_id']}",flush=True)
        try:
            data = read_bag(Path(trial['bag_path']),config['analysis'])
            metrics,streams = analyze(trial,data,config['analysis'],manual.get(trial['trial_id'],{}))
            folder = args.output/'trials'/trial['trial_id']
            folder.mkdir(parents=True,exist_ok=True)
            json_file(folder/'metrics.json',metrics)
            results.append(metrics)
            for topic, rows in streams.items():
                if rows:
                    csv_file(folder / ('series_' + topic.strip('/').replace('/', '_') + '.csv'), rows)
            if not args.no_trial_plots:
                try:
                    trial_figures(metrics,streams,folder)
                except Exception as error:
                    metrics['warnings'].append(f'Figure generation failed: {error}')
                    json_file(folder/'metrics.json',metrics)
        except Exception as error:
            failures.append(dict(trial_id=trial['trial_id'],error=f'{type(error).__name__}: {error}'))
            print(f"ERROR: {trial['trial_id']}: {error}",file=sys.stderr)
    from .count_validation import validate_counts
    validate_counts(results, args.success_counts)
    summary = summarize(results)
    csv_file(args.output/'trial_metrics.csv',results)
    csv_file(args.output/'group_summary.csv',summary)
    csv_file(args.output/'command_ablation.csv',contrasts(results,config['analysis']))
    json_file(args.output/'processing_errors.json',failures)
    report(args.output/'REPORT.md',results,failures)

    # Preserve counts derived exclusively from resolved individual outcomes.
    generated_counts_path = export_resolved_success_counts(
        args.output / "success_counts_resolved.csv",
        trials,
        results,
    )

    # Explicitly supplied aggregate counts complement individual annotations.
    # They never modify individual trial outcomes.
    if args.success_counts is not None:
        source_counts_path = args.success_counts.expanduser().resolve()

        if not source_counts_path.is_file():
            raise FileNotFoundError(
                f"Success counts file not found: {source_counts_path}"
            )

        selected_counts_path = args.output / "success_counts_used.csv"
        selected_counts_path.write_bytes(source_counts_path.read_bytes())

        print(
            "Using supplied aggregate success counts: "
            f"{source_counts_path}",
            flush=True,
        )
    else:
        selected_counts_path = generated_counts_path

    campaign_figures(
        results,
        summary,
        config,
        args.output / "figures",
        success_counts_path=selected_counts_path,
    )

    from .aggregate_outcomes import export_declared_counts

    export_declared_counts(
        selected_counts_path,
        args.output / "figures",
    )
    print(f'Output: {args.output.resolve()}')
    return 2 if failures else 0
