"""Regression checks against the supplied R02 and R03 logs."""
import copy
import sys
from pathlib import Path
from plot_navigation_sequence import read_records, find_sequences, GATE, INTENT, NAV

records, _ = read_records(Path(sys.argv[1]))
sequences, _ = find_sequences(records)
assert len(sequences) == 1
seq = sequences[0]
assert seq['goal_id'] == '1e0e09c4df351deeeb5e5b1713dc8af7'
assert abs(seq['resume']['t'] - seq['confirmation']['t'] - .257964) < 1e-6
assert abs(seq['success']['t'] - seq['navigation']['t'] - 69.066932) < 1e-6
assert seq['start_decision']['msg']['requires_replan'] is False

def rejects_change(event, field, value):
    changed = copy.deepcopy(records)
    for row in changed:
        if row['source_line'] == seq[event]['source_line']:
            row['msg'][field] = value
    assert not find_sequences(changed)[0], (event, field)

rejects_change('confirmation', 'request_id', 'unrelated_request')
rejects_change('confirmation', 'person_id', 'different_person')
rejects_change('confirmation', 'interaction_id', 'different_interaction')
rejects_change('start_accepted', 'status', 'REJECTED')
rejects_change('start_decision', 'navigation_authorized', False)
rejects_change('success', 'navigation_state', 'ABORTED')
rejects_change('resume', 'navigation_goal_id', 'different_goal')
if len(sys.argv) > 2:
    incomplete, _ = read_records(Path(sys.argv[2]))
    assert not find_sequences(incomplete)[0]
print('PASS: R02 times and correlation; mismatched identity, failed START, goal reset, missing success and R03 rejected.')
