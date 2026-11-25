#!/usr/bin/env python3
"""Audit navigation outcomes directly from ROS 2 bags without annotation overrides.

Read-only on source bags. Python 3.8+, PyYAML and a sourced ROS 2 installation.
Only a single current goal with an observed active state followed by one consistent
terminal state is automatically classified. Logs are exported as supporting evidence,
never promoted to action success. Old CSV annotations are compared AFTER extraction.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

CAMPAIGNS = (
    'L-shaped_corridor_start_goal_1',
    'L-shaped_corridor_with_doors_start_goal_1',
    'L-shaped_corridor_start_goal_1_v12',
    'L-shaped_corridor_with_doors_start_goal_1_v12',
)
CONFIGURATIONS = ('calibrated', 'Odometry-only', 'dog', 'g1_nav')
STATES = {0:'UNKNOWN', 1:'ACCEPTED', 2:'EXECUTING', 3:'CANCELING',
          4:'SUCCEEDED', 5:'CANCELED', 6:'ABORTED'}
ALIASES = {'REACHED':'SUCCEEDED', 'CANCELLED':'CANCELED'}


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def discover(root):
    trials = []
    for campaign in CAMPAIGNS:
        for configuration in CONFIGURATIONS:
            for bag in sorted((root/campaign/configuration).glob('*/navigation')):
                if not (bag/'metadata.yaml').is_file():
                    continue
                match = re.search(r'_trial_(\d+)$', bag.parent.name)
                if not match:
                    continue
                token = match.group(1)
                trials.append(dict(trial_id=f'{campaign}__{configuration}__trial_{token}',
                    campaign=campaign, configuration=configuration, trial_number=token,
                    bag_path=str(bag.resolve())))
    if len({r['trial_id'] for r in trials}) != len(trials):
        raise ValueError('Duplicate trial IDs; resolve them before auditing.')
    return trials


def previous_annotations(path):
    if path is None:
        return {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if 'trial_id' in row:
            key, value = row['trial_id'], row.get('outcome', '')
        else:
            key = f"{row['campaign']}__{row['config']}__trial_{row['trial']}"
            value = row['status']
        value = value.strip().upper()
        if value and value not in ALIASES and value not in set(STATES.values()) | {'FAILED','TIMEOUT'}:
            raise ValueError('Unknown annotation status: '+value)
        if key in result:
            raise ValueError('Duplicate previous annotation: '+key)
        result[key] = ALIASES.get(value, value)
    return result


def classify(events, selected_topic=None):
    """Pure decision rule. Old annotations and command/pose values are not inputs."""
    topics = sorted({r['topic'] for r in events})
    result = dict(automatic_outcome='UNKNOWN', reason='', evidence_topic='', goal_id='',
                  first_active_time_ns='', terminal_time_ns='', current_goal_count=0)
    if selected_topic:
        topic = selected_topic
    elif len(topics) == 1:
        topic = topics[0]
    elif len(topics) > 1:
        result['reason'] = 'MULTIPLE_STATUS_STREAMS_SELECT_TOPIC'
        return result
    else:
        result['reason'] = 'NO_STATUS_ENTRIES'
        return result
    result['evidence_topic'] = topic
    goals = defaultdict(list)
    for row in events:
        if row['topic'] == topic and row['goal_id'] and set(row['goal_id']) != {'0'}:
            goals[row['goal_id']].append(row)
    # Terminal-only entries can be stale status-array history. Do not classify them.
    current = {gid:rows for gid,rows in goals.items() if any(r['status'] in (1,2,3) for r in rows)}
    result['current_goal_count'] = len(current)
    if len(current) != 1:
        result['reason'] = 'MULTIPLE_CURRENT_GOALS' if current else 'NO_CURRENT_GOAL_ACTIVE_STATE'
        return result
    gid, rows = next(iter(current.items()))
    rows = sorted(rows, key=lambda r:r['bag_time_ns'])
    start = min(r['bag_time_ns'] for r in rows if r['status'] in (1,2,3))
    terminal = [r for r in rows if r['status'] in (4,5,6)]
    result.update(goal_id=gid, first_active_time_ns=start)
    if not terminal:
        result['reason'] = 'NO_TERMINAL_STATUS'
        return result
    if len({r['status'] for r in terminal}) != 1:
        result['reason'] = 'CONFLICTING_TERMINAL_STATES'
        return result
    end = terminal[0]['bag_time_ns']
    if end < start or any(r['status'] in (1,2,3) and r['bag_time_ns'] >= end for r in rows):
        result['reason'] = 'INCONSISTENT_STATE_ORDER'
        return result
    result.update(automatic_outcome=STATES[terminal[0]['status']],
                  terminal_time_ns=end, reason='ACTIVE_THEN_TERMINAL_ACTION_STATUS')
    return result


def extract(trial, folder, requested_topic=None):
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

    path = Path(trial['bag_path'])
    metadata_bytes = (path/'metadata.yaml').read_bytes()
    info = yaml.safe_load(metadata_bytes)['rosbag2_bagfile_information']
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(path), storage_id=info['storage_identifier']),
                ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {t.name:t.type for t in reader.get_all_topics_and_types()}
    status_topics = {t for t,typ in types.items() if typ=='action_msgs/msg/GoalStatusArray'
                     and ('navigate_to_pose' in t or t==requested_topic)}
    log_topics = {t for t,typ in types.items() if typ=='rcl_interfaces/msg/Log' and t.endswith('/rosout')}
    wanted = status_topics | log_topics
    classes = {t:get_message(types[t]) for t in wanted}
    # Filtering avoids deserializing or loading every large LiDAR record where supported.
    if wanted:
        try:
            from rosbag2_py import StorageFilter
            reader.set_filter(StorageFilter(topics=sorted(wanted)))
        except (ImportError, AttributeError, TypeError):
            pass
    events, logs = [], []
    counts = Counter()
    last_state = {}
    if wanted:
        while reader.has_next():
            topic, raw, ns = reader.read_next()
            if topic not in wanted:
                continue
            counts[topic] += 1
            msg = deserialize_message(raw, classes[topic])
            if topic in status_topics:
                for entry in msg.status_list:
                    gid = bytes(entry.goal_info.goal_id.uuid).hex()
                    status = int(entry.status)
                    # Preserve state changes; repeated terminal arrays are not new outcomes.
                    key = (topic,gid)
                    if last_state.get(key) == status:
                        continue
                    last_state[key] = status
                    stamp = entry.goal_info.stamp
                    events.append(dict(topic=topic, goal_id=gid, status=status,
                        state=STATES.get(status,'INVALID'), bag_time_ns=int(ns),
                        goal_stamp_ns=int(stamp.sec)*1000000000+int(stamp.nanosec)))
            else:
                text = msg.msg.lower()
                if any(word in text for word in ('goal','abort','cancel','succeed','reached','failed')):
                    logs.append(dict(topic=topic,bag_time_ns=int(ns),node=msg.name,
                                     level=int(msg.level),message=msg.msg))
    events.sort(key=lambda r:r['bag_time_ns'])
    write_csv(folder/'status_events.csv', events,
              ['topic','goal_id','status','state','bag_time_ns','goal_stamp_ns'])
    write_csv(folder/'supporting_logs.csv', logs,['topic','bag_time_ns','node','level','message'])
    topic_rows = []
    for entry in info.get('topics_with_message_count',[]):
        tm = entry['topic_metadata']
        if tm['name'] in wanted or any(s in tm['name'] for s in ('navigate_to_pose','evaluation')):
            topic_rows.append(dict(topic=tm['name'],type=tm['type'],
                metadata_messages=entry['message_count'],read_messages=counts.get(tm['name'],0)))
    write_csv(folder/'topic_inventory.csv',topic_rows,
              ['topic','type','metadata_messages','read_messages'])
    result = classify(events, requested_topic)
    result.update(status_topics=';'.join(sorted(status_topics)),
        status_messages=sum(counts[t] for t in status_topics),status_event_count=len(events),
        supporting_log_count=len(logs), metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
    if not status_topics:
        result['reason']='NO_NAVIGATE_TO_POSE_STATUS_TOPIC'
    (folder/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'g1_ws/experiments')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,help='Optional previous CSV, used only after independent extraction.')
    parser.add_argument('--status-topic',help='Explicit NavigateToPose status topic when multiple streams exist.')
    args=parser.parse_args()
    trials=discover(args.root.expanduser())
    if not trials:
        parser.error('No bags discovered in the four configured campaigns.')
    # Validate imports before starting the campaign; no ROS daemon or robot is needed.
    try:
        import rosbag2_py
        import rclpy.serialization
        import yaml
    except ImportError as error:
        parser.error('Source your ROS 2 environment first: '+str(error))
    previous=previous_annotations(args.previous.expanduser() if args.previous else None)
    output=args.output.expanduser()
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be a new or empty directory.')
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(command_line=sys.argv,root=str(args.root.expanduser().resolve()),
                  outcome_rule='single current goal, active then one terminal state',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  previous_csv_sha256=hashlib.sha256(args.previous.expanduser().read_bytes()).hexdigest() if args.previous else None)
    (output/'audit_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    results=[]
    for i,trial in enumerate(trials,1):
        print(f"[{i}/{len(trials)}] {trial['trial_id']}",flush=True)
        folder=output/'trials'/trial['trial_id']; folder.mkdir(parents=True)
        try:
            decision=extract(trial,folder,args.status_topic)
        except Exception as error:
            decision=dict(automatic_outcome='UNKNOWN',reason='READ_ERROR',error=str(error))
            (folder/'error.txt').write_text(type(error).__name__+': '+str(error)+'\n')
        old=previous.get(trial['trial_id'],'')
        automatic=decision['automatic_outcome']
        comparison=('NO_PREVIOUS_ANNOTATION' if not old or old=='UNKNOWN' else
                    'NOT_VERIFIABLE' if automatic=='UNKNOWN' else
                    'AGREEMENT' if old==automatic else 'DISAGREEMENT')
        result=dict(trial,**decision,previous_annotation=old,comparison=comparison)
        results.append(result)
        print(f"  {automatic}: {decision['reason']} | {comparison}",flush=True)
        # Checkpoint after every bag, so partial work survives interruption.
        fields=list(dict.fromkeys(k for r in results for k in r))
        write_csv(output/'outcome_audit.csv',results,fields)
    review=[r for r in results if r['automatic_outcome']=='UNKNOWN' or r['comparison']=='DISAGREEMENT']
    write_csv(output/'needs_review.csv',review,list(dict.fromkeys(k for r in results for k in r)))
    # Candidate CSV is NEVER loaded by the analyzer automatically. Review first.
    candidates=[dict(trial_id=r['trial_id'],outcome=r['automatic_outcome'] if r['automatic_outcome']!='UNKNOWN' else '',
         evidence=f"ROS action status; topic={r.get('evidence_topic','')}; goal_id={r.get('goal_id','')}; bag_time_ns={r.get('terminal_time_ns','')}" if r['automatic_outcome']!='UNKNOWN' else '',
         notes=r['reason']) for r in results]
    write_csv(output/'automatic_annotations_candidate.csv',candidates,['trial_id','outcome','evidence','notes'])
    counts=Counter(r['automatic_outcome'] for r in results)
    comparisons=Counter(r['comparison'] for r in results)
    unmatched=sorted(set(previous)-{r['trial_id'] for r in trials})
    lines=['# Navigation outcome audit','',f'Discovered trials: {len(trials)}',
        'Automatic outcomes: '+str(dict(counts)), 'Annotation comparison: '+str(dict(comparisons)),
        f'Trials requiring review: {len(review)}',f'Previous IDs absent from discovery: {len(unmatched)}','',
        'Old annotations never influence automatic outcomes. UNKNOWN is not failure.',
        'Terminal-only historical statuses and multiple current goals require review.',
        'Action outcomes establish the software result, not independent physical goal accuracy.',
        'Supporting ROS logs are exported but never converted automatically into success.',
        'Only the four named navigation campaigns and four configurations are discovered.',
        'Candidate annotations require review before use in paper analysis.','']
    lines.extend(unmatched)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'+ '\n'.join(lines[:7]))
    print('Output: '+str(output.resolve()))
    return 2 if any(r['reason']=='READ_ERROR' for r in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
