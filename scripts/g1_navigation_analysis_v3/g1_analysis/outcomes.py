"""Conservative goal selection and explicit manual evidence overrides."""
NAMES = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}
ALLOWED = set(NAMES.values()) | {'UNKNOWN', 'TIMEOUT', 'FAILED'}


def resolve(data, annotation):
    selected = annotation.get('goal_id', '').strip()
    candidates = {}
    for row in data['statuses']:
        gid = row['goal_id']
        if not gid or set(gid) == {'0'}:
            continue
        candidates.setdefault(gid, []).append(row)
    if not selected:
        # Exclude old terminal-only entries retained in status arrays.
        eligible = [gid for gid, rows in candidates.items() if any(
            r['status'] in (1,2,3) or
            data['bag_start'] <= r['goal_stamp'] <= data['bag_end'] for r in rows)]
        selected = eligible[0] if len(eligible) == 1 else None
    rows = candidates.get(selected, [])
    terminal = sorted((r for r in rows if r['status'] in NAMES), key=lambda r: r['t'])
    automatic = NAMES[terminal[0]['status']] if terminal else 'UNKNOWN'
    starts = [r['t'] for r in rows if r['status'] in (1,2)]
    result = dict(outcome=automatic, automatic_outcome=automatic,
                  outcome_source='action_status' if terminal else 'unavailable',
                  goal_id=selected, candidate_goal_count=len(candidates),
                  start_s=min(starts) if starts else None,
                  end_s=terminal[0]['t'] if terminal else None)
    override = annotation.get('outcome', '').strip().upper()
    if override:
        if override not in ALLOWED or not annotation.get('evidence', '').strip():
            raise ValueError('Annotated outcome requires a valid outcome and evidence.')
        result.update(outcome=override, outcome_source='annotation', evidence=annotation['evidence'])
    for field in ('start_s', 'end_s'):
        if annotation.get(field, '').strip():
            result[field] = float(annotation[field])
    if result['start_s'] is not None and result['end_s'] is not None:
        if result['end_s'] < result['start_s']:
            raise ValueError('Terminal time precedes mission start.')
    return result
