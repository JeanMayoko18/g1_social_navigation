"""Validate aggregate counts without assigning trial outcomes."""
from collections import defaultdict
from .paper_plots import load_declared_success_counts


def validate_counts(rows, path):
    if path is None:
        return
    declared = load_declared_success_counts(path)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scenario'], round(float(row['speed_mps']), 2), row['configuration'])].append(row)
    for key, group in groups.items():
        if key not in declared:
            continue
        successes, total = declared[key]
        known = [r for r in group if r.get('outcome', 'UNKNOWN') != 'UNKNOWN']
        reached = sum(r['outcome'] == 'SUCCEEDED' for r in known)
        failures = len(known) - reached
        if len(group) != total or not reached <= successes <= total - failures:
            raise ValueError(f'Aggregate counts conflict with trial outcomes for {key}')
