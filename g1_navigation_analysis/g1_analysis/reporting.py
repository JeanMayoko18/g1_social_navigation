"""Machine-readable exports and a human-readable audit report."""
import csv
import json
from pathlib import Path


def json_file(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_file(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = sorted({k for row in rows for k,v in row.items() if not isinstance(v,(dict,list))})
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def report(path, results, failures):
    lines = ['# G1 navigation analysis', '',
             f'Analyzed trials: {len(results)}. Processing errors: {len(failures)}.', '',
             '## Interpretation', '',
             '- The four configurations are compared within each scenario and nominal speed.',
             '- Unknown outcomes are reported separately; success intervals use known outcomes only.',
             '- Canceled missions count as non-success, not automatically as technical failures.',
             '- Terminal residuals use an estimated AMCL pose, not independent physical ground truth.',
             '- Odometry-AMCL consistency uses initial SE(2) alignment of each trajectory; it is not ATE.',
             '- Plan deviation is point-to-segment distance from AMCL samples to the first plan in the analysis window.',
             '- Plan publications are not counted as independent replanning decisions.',
             '- Command comparisons use nearest reception times by default; they are approximate transport diagnostics.',
             '- Compensation contrasts are unpaired. Trial numbers do not establish experimental pairing.',
             '- Continuous contrasts use successful missions only and may be affected by survival selection.',
             '- Bootstrap intervals are descriptive, especially with small samples or uniform binary outcomes.',
             '- A shared nonempty settings_id is a user declaration, not automatic verification of matching parameters.',
             '- Wall clearance, jerk, causal abort diagnosis, and independent localization accuracy are not computed.',
             '', '## Trial audit', '']
    for trial in results:
        lines.extend([f"### {trial['trial_id']}",f"Outcome: {trial['outcome']} ({trial['outcome_source']}).",''])
        lines.extend('- '+w for w in trial['warnings'])
        lines.append('')
    if failures:
        lines += ['## Processing failures — excluded from summaries, must be resolved', '']
        lines += [f"- {f['trial_id']}: {f['error']}" for f in failures]
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
