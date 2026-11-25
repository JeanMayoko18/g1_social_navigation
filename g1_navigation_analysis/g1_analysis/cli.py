"""Orchestration only: discovery -> reader -> metrics -> aggregate -> exports."""
import argparse
import hashlib
import sys
from pathlib import Path
import yaml
from . import __version__
from .discovery import discover, annotations, write_annotation_template
from .reader import read_bag
from .metrics import analyze
from .aggregate import summarize, contrasts
from .reporting import json_file, csv_file, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'g1_ws/experiments')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/campaign.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--inventory-only',action='store_true',help='No ROS imports or bag deserialization.')
    parser.add_argument('--trial',action='append',help='Exact trial ID; repeat to select multiple trials.')
    parser.add_argument('--no-trial-plots',action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['analysis']['time_basis'] not in ('bag','header'):
        parser.error('time_basis must be bag or header')
    trials = discover(args.root.expanduser(),config)
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
    csv_file(args.output/'inventory.csv',trials)
    write_annotation_template(args.output/'annotations_template.csv',trials)
    json_file(args.output/'run_config.json',dict(version=__version__,configuration=config,
              source_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
              command_line=sys.argv,root=str(args.root.resolve())))
    print(f'Discovered {len(trials)} trials.')
    if args.inventory_only:
        return 0
    manual = annotations(args.annotations)
    if args.annotations:
        (args.output/'annotations_used.csv').write_bytes(args.annotations.read_bytes())
    from .plots import campaign_figures, trial_figures
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
    summary = summarize(results)
    csv_file(args.output/'trial_metrics.csv',results)
    csv_file(args.output/'group_summary.csv',summary)
    csv_file(args.output/'command_ablation.csv',contrasts(results,config['analysis']))
    json_file(args.output/'processing_errors.json',failures)
    report(args.output/'REPORT.md',results,failures)
    campaign_figures(results,summary,config,args.output/'figures')
    print(f'Output: {args.output.resolve()}')
    return 2 if failures else 0
