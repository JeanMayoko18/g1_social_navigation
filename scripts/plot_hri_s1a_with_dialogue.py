#!/usr/bin/env python3
"""Create a poster-ready HRI case-study figure from a ROS 2 rosbag.

Run inside an environment where ROS 2 and your custom message packages are sourced.
Uses bag receive timestamps (not synchronized video timestamps).
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INTEREST = {
    '/goal_pose',
    '/navigate_to_pose_internal/_action/goal',
    '/navigate_to_pose_internal/_action/status',
    '/navigate_to_pose/_action/status',
    '/odom', '/dog_odom_2d',
    '/social_navigation/decision',
    '/social_navigation/collaborative_decision',
    '/social_navigation/navigation_state',
    '/social_navigation/episode_state',
    '/social_navigation/trajectory_social_validation',
    '/social_navigation/reassessment_result',
    '/explainability/text', '/explainability/speech_status',
    '/hri/dialogue_request', '/hri/human_intent',
    '/dialogue_manager/robot_speech', '/dialogue_manager/closed_captions',
    '/humans/voices/pc_user/speech', '/intents',
    '/hri/asr_turn_status', '/explainability/context',
}


def primitive(obj, depth=0):
    """Convert ROS messages to JSON-like values, even custom interfaces."""
    if depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f'<{len(obj)} bytes>'
    if isinstance(obj, (list, tuple)):
        return [primitive(x, depth + 1) for x in list(obj)[:40]]
    if hasattr(obj, 'tolist'):
        return primitive(obj.tolist(), depth + 1)
    if hasattr(obj, 'get_fields_and_field_types'):
        data = {key: primitive(getattr(obj, key), depth + 1)
                for key in obj.get_fields_and_field_types()}
        # Preserve constants from the actual installed ROS message definition.
        # Never guess that an integer such as 2 or 3 means REPLAN or RESUME.
        constants = {}
        for name in set(dir(type(obj))) | set(dir(type(type(obj)))):
            if name.isupper() and not name.startswith('_'):
                value = getattr(type(obj), name, None)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    constants[name] = value
        if constants:
            data['__enum_constants__'] = constants
        return data
    return str(obj)


def flat_text(obj):
    if isinstance(obj, dict):
        return ' '.join(f'{k} {flat_text(v)}' for k, v in obj.items())
    if isinstance(obj, list):
        return ' '.join(flat_text(v) for v in obj)
    return str(obj)


def get_field(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data:
            return data[key]
    return None


def extract_odom(data):
    """Return x,y,linear planar speed, angular speed or None."""
    try:
        pose = data['pose']['pose'] if isinstance(data['pose'], dict) and 'pose' in data['pose'] else data['pose']
        twist = data['twist']['twist'] if isinstance(data['twist'], dict) and 'twist' in data['twist'] else data['twist']
        pos = pose['position']
        lin = twist['linear']
        ang = twist['angular']
        return (float(pos['x']), float(pos['y']),
                math.hypot(float(lin['x']), float(lin['y'])),
                abs(float(ang['z'])))
    except (KeyError, TypeError, ValueError):
        return None


def terminal_status(data):
    """Action status codes: 4 succeeded, 5 canceled, 6 aborted."""
    if not isinstance(data, dict):
        return []
    arr = data.get('status_list', [])
    out = []
    if isinstance(arr, list):
        for item in arr:
            if isinstance(item, dict):
                status = item.get('status')
                if status in (4, 5, 6):
                    out.append((status, flat_text(item.get('goal_info', {}))))
    return out


def read_bag(bag_dir, extra_topics=()):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError('ROS 2 Python libraries not found. Source /opt/ros/<distro>/setup.bash and your ~/g1_ws/install/setup.bash first.') from exc

    if not (bag_dir / 'metadata.yaml').exists():
        raise FileNotFoundError(f'No metadata.yaml in {bag_dir}; give the navigation bag DIRECTORY, not its parent.')
    # rosbag2 storage: sqlite3 (.db3) or MCAP (.mcap), as declared by metadata.
    import yaml
    metadata = yaml.safe_load((bag_dir / 'metadata.yaml').read_text())
    storage = metadata.get('rosbag2_bagfile_information', {}).get('storage_identifier', 'sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    # Discover supervisory topics rather than requiring two fixed names.
    usable = {topic for topic in types if topic in INTEREST or topic in extra_topics
              or topic.startswith(('/social_navigation/', '/hri/', '/explainability/'))}
    message_classes = {}
    for topic in usable:
        try:
            message_classes[topic] = get_message(types[topic])
        except Exception as exc:
            print(f'WARNING: cannot load {types[topic]} for {topic}: {exc}', file=sys.stderr)
    rows = defaultdict(list)
    skipped = defaultdict(int)
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in message_classes:
            continue
        try:
            msg = deserialize_message(serialized, message_classes[topic])
            data = primitive(msg)
            rows[topic].append((timestamp_ns / 1e9, data))
        except Exception:
            skipped[topic] += 1
    if skipped:
        print('WARNING: deserialization errors:', dict(skipped), file=sys.stderr)
    return rows, types


# Only structured event fields are interpreted. Free-form explanations/reasons
# can mention a future action and must not be treated as executed commands.
EVENT_FIELDS = {
    'command', 'cmd', 'action', 'decision', 'state', 'status', 'phase', 'event',
    'event_type', 'command_type', 'decision_type', 'current_state', 'current_phase',
    'navigation_state', 'navigation_command', 'gate_command', 'supervisor_command',
    'requested_action', 'requested_command', 'selected_action', 'data',
}
ALIASES = {
    'WAIT': 'WAIT', 'WAITING': 'WAIT',
    'REPLAN': 'REPLAN', 'REPLANNING': 'REPLAN', 'REPLAN_REQUESTED': 'REPLAN',
    'REPLAN_STARTED': 'REPLAN', 'REPLAN_COMPLETED': 'REPLAN',
    'RESUME': 'RESUME', 'RESUMING': 'RESUME', 'RESUMED': 'RESUME',
    'RESUME_REQUESTED': 'RESUME', 'RESUME_NAVIGATION': 'RESUME',
    'REPLAN_NAVIGATION': 'REPLAN', 'WAITING_FOR_HUMAN': 'WAIT',
    # START is deliberately distinct: an initial goal start is not a resumption.
    'START': 'START', 'STOP': 'STOP',
}
PREFIXES = ('COMMAND_', 'CMD_', 'ACTION_', 'DECISION_', 'STATE_', 'STATUS_',
            'PHASE_', 'EVENT_', 'NAVIGATION_', 'GATE_')


def event_symbol(value):
    if not isinstance(value, str):
        return None
    token = re.sub(r'[\s-]+', '_', value.strip().upper())
    # Accept explicitly namespaced constants, but never a prose substring.
    token = token.split('::')[-1].split('.')[-1]
    for _ in range(4):
        if token in ALIASES:
            return ALIASES[token]
        prefix = next((x for x in PREFIXES if token.startswith(x)), None)
        if not prefix:
            break
        token = token[len(prefix):]
    return None


def event_fields(data, path='', ancestors=(), selected_paths=()):
    """Yield (field path, raw value, sibling constants) from current fields."""
    if isinstance(data, str):
        try:
            decoded = json.loads(data)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            yield from event_fields(decoded, path, ancestors, selected_paths)
        elif not path or path.split('.')[-1].lower() in EVENT_FIELDS | {'data'}:
            yield path or 'data', data, {}
    elif isinstance(data, dict):
        constants = data.get('__enum_constants__', {})
        for key, value in data.items():
            key_lower = key.lower()
            if key.startswith('__') or key_lower.startswith(('previous', 'prev_', 'last_', 'history')):
                continue
            child = f'{path}.{key}' if path else key
            # std_msgs/String can contain a serialized structured decision.
            if key_lower == 'data' and isinstance(value, str) and value.lstrip().startswith(('{', '[')):
                yield from event_fields(value, child, ancestors, selected_paths)
            elif (child in selected_paths or key_lower in EVENT_FIELDS or key_lower.endswith(('_command', '_state', '_phase'))) and not isinstance(value, (dict, list)):
                yield child, value, constants
            elif isinstance(value, (dict, list)) or (key_lower == 'data' and isinstance(value, str)):
                yield from event_fields(value, child, ancestors + (key_lower,), selected_paths)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from event_fields(value, f'{path}[{i}]', ancestors, selected_paths)


def resolve_event(topic, field, value, constants, overrides):
    """Return a category and its provenance, or a diagnostic reason."""
    configured = overrides.get(f'{topic}|{field}', {})
    if str(value) in configured:
        return configured[str(value)], 'explicit_mapping'
    direct = event_symbol(value)
    if direct:
        return direct, 'symbolic_value'
    # Match numeric enums through message constants. Resolve conflicting enums
    # only when their family matches the field (e.g. COMMAND_* vs STATE_*).
    names = [name for name, number in constants.items()
             if type(number) is type(value) and number == value and event_symbol(name)]
    leaf = field.split('.')[-1].lower()
    family = ('STATE_', 'STATUS_', 'PHASE_') if any(x in leaf for x in ('state', 'status', 'phase')) else (
              'COMMAND_', 'CMD_', 'ACTION_', 'DECISION_')
    if 'event' in leaf:
        family = ('EVENT_',)
    preferred = [name for name in names if name.startswith(family)]
    # A STATE_* constant must not decode a command field merely because the
    # integer happens to match. Bare REPLAN/RESUME constants remain supported.
    names = preferred or [name for name in names if not name.startswith(PREFIXES)]
    categories = {event_symbol(name) for name in names}
    if len(categories) == 1:
        return categories.pop(), 'ros_constant:' + '|'.join(sorted(names))
    return None, 'ambiguous_enum' if len(categories) > 1 else 'unrecognized_value'


def detect_events(rows, t0, t1, event_map=None, diagnostics=None):
    events = []
    event_map = event_map or {}
    diagnostic_rows = diagnostics if diagnostics is not None else []

    def add(t, typ, topic, detail='', field='', evidence='publication', raw_value=''):
        if t0 <= t <= t1:
            events.append({'time_s': round(t - t0, 3), 'event': typ,
                           'topic': topic, 'field': field, 'evidence': evidence,
                           'raw_value': str(raw_value), 'detail': detail[:1200]})

    for topic, messages in rows.items():
        if topic not in ('/social_navigation/decision', '/social_navigation/collaborative_decision') and not (
            topic.startswith(('/social_navigation/', '/hri/')) or any(k.startswith(topic + '|') for k in event_map)
        ):
            continue
        last_states = {}
        selected_paths = {key.split('|', 1)[1] for key in event_map if key.startswith(topic + '|')}
        for t, data in sorted(messages, key=lambda row: row[0]):
            for field, value, constants in event_fields(data, selected_paths=selected_paths):
                category, resolution = resolve_event(topic, field, value, constants, event_map)
                leaf = field.split('.')[-1].lower()
                is_state = any(name in leaf for name in ('state', 'status', 'phase'))
                is_request = ('requested' in leaf or 'REQUESTED' in str(value).upper()
                              or topic.startswith('/hri/') or 'request' in topic)
                evidence = 'state_observation' if is_state else ('request' if is_request else 'command_or_event')
                repeated = is_state and field in last_states and last_states[field] == value
                if is_state:
                    last_states[field] = value
                # Include observations outside the plot window for diagnosis.
                diagnostic_rows.append({'time_unix_s': t, 'time_s': round(t-t0, 3),
                    'topic': topic, 'field': field, 'raw_value': str(value),
                    'resolved_event': category or '', 'resolution': resolution,
                    'inside_window': t0 <= t <= t1, 'repeated_state': repeated})
                if category and not repeated:
                    # A state first observed mid-run is not claimed as a command.
                    add(t, category, topic, json.dumps(data, ensure_ascii=False), field,
                        evidence + ';' + resolution, value)

    for t, data in rows.get('/explainability/text', []):
        content = get_field(data, 'data', 'text', 'utterance', 'message', 'explanation')
        speech = flat_text(data if content is None else content).strip()
        if speech and speech.lower() not in ('none', 'null', '{}', '[]'):
            add(t, 'EXPLAIN', '/explainability/text', speech, evidence='text_publication')

    for t, data in rows.get('/explainability/speech_status', []):
        value = get_field(data, 'status', 'state', 'data')
        if isinstance(value, str) and value.upper() in ('STARTED', 'COMPLETED', 'FAILED'):
            add(t, 'SPEECH_' + value.upper(), '/explainability/speech_status', flat_text(data))

    for topic in ('/navigate_to_pose_internal/_action/status', '/navigate_to_pose/_action/status'):
        seen = set()
        for t, data in sorted(rows.get(topic, []), key=lambda row: row[0]):
            current = set(terminal_status(data))
            for status, goal_id in sorted(current - seen):
                add(t, {4:'GOAL_SUCCEEDED', 5:'GOAL_CANCELED', 6:'GOAL_ABORTED'}[status],
                    topic, goal_id, evidence='action_status')
            seen |= current

    # Preserve different sources/evidence instead of silently merging them.
    # Only suppress near-identical repeats of the same command on one field.
    events.sort(key=lambda e: (e['time_s'], e['event'], e['topic'], e['field']))
    result, last = [], {}
    for event in events:
        key = tuple(event[k] for k in ('event', 'topic', 'field', 'raw_value', 'evidence'))
        command = event['event'] in ('WAIT', 'REPLAN', 'RESUME', 'START', 'STOP')
        previous = last.get(key)
        if command and previous is not None and event['time_s'] - previous <= .25:
            continue
        last[key] = event['time_s']
        result.append(event)
    return result


def motion_metrics(rows, t0, t1, max_gap=1.0):
    source = '/odom' if rows.get('/odom') else '/dog_odom_2d'
    samples = []
    for t, data in rows.get(source, []):
        if not (t0 <= t <= t1):
            continue
        vals = extract_odom(data)
        if vals is not None and all(map(math.isfinite, vals)):
            samples.append((t, *vals))
    samples.sort()
    if len(samples) < 2:
        return None, None, source, 0, []
    path = 0.0
    stationary = 0.0
    segments = []
    run_start = None
    last_still_end = None
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > max_gap:
            if run_start is not None and last_still_end is not None and last_still_end-run_start >= 0.5:
                stationary += last_still_end-run_start
                segments.append((run_start-t0, last_still_end-t0))
            run_start = last_still_end = None
            continue
        jump = math.hypot(b[1] - a[1], b[2] - a[2])
        if jump <= 1.0:  # reject implausible single-sample localization jumps
            path += jump
        # Stationary if speed < 0.03 m/s and |omega_z| < 0.05 rad/s for >=0.5 s.
        still = a[3] < 0.03 and a[4] < 0.05
        if still:
            if run_start is None:
                run_start = a[0]
            last_still_end = b[0]
        elif run_start is not None:
            if last_still_end is not None and last_still_end-run_start >= 0.5:
                stationary += last_still_end-run_start
                segments.append((run_start-t0, last_still_end-t0))
            run_start = last_still_end = None
    if run_start is not None and last_still_end is not None and last_still_end-run_start >= 0.5:
        stationary += last_still_end-run_start
        segments.append((run_start-t0, last_still_end-t0))
    return path, stationary, source, len(samples), segments


def choose_window(rows, mode='goal-to-end'):
    all_ts = [t for topic in rows.values() for t, _ in topic]
    if not all_ts:
        raise ValueError('No relevant messages found in bag.')
    goal_ts = [t for topic in ('/goal_pose', '/navigate_to_pose_internal/_action/goal')
               for t, _ in rows.get(topic, [])]
    t0 = min(goal_ts) if goal_ts else min(all_ts)
    if mode == 'bag':
        return min(all_ts), max(all_ts), 'all decoded relevant messages'
    if mode == 'goal-to-end':
        return t0, max(all_ts), 'first available goal → last decoded relevant message'
    # Optional legacy window; action statuses may include unrelated goals.
    successes = []
    for topic in ('/navigate_to_pose/_action/status', '/navigate_to_pose_internal/_action/status'):
        seen = set()
        for t, data in rows.get(topic, []):
            current = set(terminal_status(data))
            if t >= t0 and any(status == 4 for status, _ in current - seen):
                successes.append(t)
            seen = current
    if successes:
        t1 = max(successes)
        note = 'first public/available goal → last observed SUCCEEDED status'
    else:
        t1 = max(all_ts)
        note = 'first goal (or first recorded topic) → last recorded relevant message; NO confirmed success'
    if t1 <= t0:
        t1 = max(all_ts)
    return t0, t1, note


def extract_dialogue(rows, t0, t1):
    """Export observed messages, not inferred spoken conversations."""
    mapping = [
        ('/explainability/text', 'ROBOT_TEXT'),
        ('/dialogue_manager/robot_speech', 'ROBOT_SPEECH_TOPIC'),
        ('/dialogue_manager/closed_captions', 'CAPTION'),
        ('/hri/dialogue_request', 'DIALOGUE_REQUEST'),
        ('/humans/voices/pc_user/speech', 'HUMAN_SPEECH'),
        ('/hri/human_intent', 'HUMAN_INTENT'),
        ('/intents', 'INTENT'),
        ('/hri/asr_turn_status', 'ASR_STATUS'),
    ]
    def content(data):
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in ('utterance', 'transcript', 'recognized_text', 'speech',
                      'text', 'caption', 'sentence', 'data', 'message', 'content',
                      'question', 'prompt', 'intent'):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for k in ('request', 'response', 'result'):
                v = data.get(k)
                if isinstance(v, dict):
                    inner = content(v)
                    if inner:
                        return inner
        return json.dumps(data, ensure_ascii=False)[:600]

    entries=[]
    for topic, kind in mapping:
        for t, data in rows.get(topic, []):
            if t0 <= t <= t1:
                msg=content(data).strip()
                if msg and msg.lower() not in ('null','none','{}','[]'):
                    entries.append({'time_s': round(t-t0,3), 'source':kind,
                                    'topic':topic, 'text':msg[:1200]})
    return sorted(entries,key=lambda x:x['time_s'])


def make_figure(output, duration, path, stationary, events, segments, note, dialogue):
    import textwrap
    ink='#183746'; teal='#246C80'; orange='#CE893D'; blue='#5279BA'; green='#23846B'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,11.5),facecolor='white')
    gs=fig.add_gridspec(3,1,height_ratios=[2.6,2.9,1.3], hspace=0.42,
                        left=0.10,right=0.97,top=0.89,bottom=0.095)
    fig.text(0.10,0.965,'Configuration D | Social Encounter Case Study',
             color=ink,fontsize=22,fontweight='bold',va='top')
    fig.text(0.10,0.925,'A. Timeline     •     B. Recorded communication     •     C. Robot response',
             color=teal,fontsize=13,fontweight='bold')

    ax=fig.add_subplot(gs[0]); ax.set_facecolor('#F5F8F9')
    ymap={'WAIT':4,'EXPLAIN':3,'REPLAN':2,'RESUME':1,'START':0,'SPEECH_COMPLETED':-1}
    colors={'WAIT':orange,'EXPLAIN':blue,'REPLAN':teal,
            'RESUME':green,'START':'#6A7685','SPEECH_COMPLETED':'#8A64A8'}
    shown=set()
    for e in events:
        et=e['event']
        if et not in ymap: continue
        t=e['time_s']; y=ymap[et]
        ax.vlines(t,y-.23,y+.23,color=colors[et],linewidth=2)
        marker = 's' if e.get('evidence', '').startswith('state_observation') else (
            '^' if e.get('evidence', '').startswith('request') else 'o')
        ax.scatter([t],[y],s=52,color=colors[et],marker=marker,zorder=4)
        if et not in shown:
            shown.add(et)
            ax.annotate(f'{t:.2f} s',(t,y),xytext=(5,9),textcoords='offset points',
                        fontsize=9,color=ink)
    for a,b in segments:
        ax.axvspan(a,b,color=orange,alpha=.09)
    ax.set_yticks(list(ymap.values()),list(ymap.keys()))
    ax.set_xlim(0,max(1,duration*1.015)); ax.set_ylim(-1.6,4.65)
    ax.grid(axis='x',color='#D9E2E5')
    ax.set_xlabel('Seconds from start of analysis window')
    ax.spines[['top','right','left']].set_visible(False)
    ax.text(.99,1.04,'Shading: stationary | Square: state | Triangle: request | Circle: command/publication',transform=ax.transAxes,
            ha='right',color='#5E7079',fontsize=9)

    ax2=fig.add_subplot(gs[1]);ax2.axis('off')
    ax2.set_title('B. Recorded communication (selected messages)',loc='left',
                  fontsize=15,color=teal,fontweight='bold',pad=12)
    # Keep chronological representative utterances; show exactly which topic produced them.
    candidates=[d for d in dialogue if d['source'] in
                ('ROBOT_TEXT','HUMAN_SPEECH','HUMAN_INTENT','ROBOT_SPEECH_TOPIC','DIALOGUE_REQUEST')]
    unique=[]
    for d in candidates:
        if any(prev['source']==d['source'] and prev['text']==d['text']
               and abs(prev['time_s']-d['time_s']) < 2 for prev in unique):
            continue
        unique.append(d)
    chosen=unique[:6]
    if not chosen:
        ax2.text(.02,.72,'No text or dialogue messages decoded in selected analysis window.',
                 transform=ax2.transAxes,color=ink,fontsize=12)
        ax2.text(.02,.55,'Check dialogue_transcript.csv and topic availability.',
                 transform=ax2.transAxes,color=ink,fontsize=11)
    else:
        for i,d in enumerate(chosen):
            y=.92-i*.145
            speaker=('Human' if d['source'].startswith('HUMAN') else
                     'Robot text' if d['source']=='ROBOT_TEXT' else
                     'Dialogue request' if d['source']=='DIALOGUE_REQUEST' else 'Robot speech topic')
            line=' '.join(d['text'].split())
            line=textwrap.shorten(line,width=100,placeholder='…')
            ax2.text(.02,y,f"{d['time_s']:6.2f} s  {speaker}",transform=ax2.transAxes,
                     color=blue if speaker!='Human' else green,fontsize=10,fontweight='bold',va='top')
            ax2.text(.02,y-.057,line,transform=ax2.transAxes,color=ink,fontsize=10,va='top')
    if len(unique)>len(chosen):
        ax2.text(.02,.005,f'{len(unique)-len(chosen)} additional messages: see dialogue_transcript.csv',
                 transform=ax2.transAxes,color='#536774',fontsize=9)

    ax3=fig.add_subplot(gs[2]);ax3.axis('off')
    ax3.set_title('C. Detected event categories (not a chronological sequence)',loc='left',fontsize=15,
                  color=teal,fontweight='bold',pad=10)
    selected=[e['event'] for e in events if e['event'] in ('WAIT','EXPLAIN','REPLAN','RESUME')]
    steps=[]
    for label in ('WAIT','EXPLAIN','REPLAN','RESUME'):
        if label in selected:
            steps.append(label)
    if not steps:
        ax3.text(.02,.48,'No matching command events found.',transform=ax3.transAxes)
    else:
        width=min(.21,.86/len(steps))
        for i,label in enumerate(steps):
            x=.035+i*(.92/len(steps))
            c={'WAIT':orange,'EXPLAIN':blue,'REPLAN':teal,'RESUME':green}[label]
            box=FancyBboxPatch((x,.38),width,.38,boxstyle='round,pad=.016',
                               transform=ax3.transAxes,facecolor='#F5F8F9',
                               edgecolor=c,linewidth=2)
            ax3.add_patch(box)
            ax3.text(x+width/2,.57,label,transform=ax3.transAxes,ha='center',
                     va='center',fontsize=12,fontweight='bold',color=c)
    pval=f'{path:.2f} m' if path is not None else 'N/A'
    sval=f'{stationary:.2f} s' if stationary is not None else 'N/A'
    fig.text(.10,.038,f'Duration: {duration:.2f} s     |     Stationary: {sval}     |     Path: {pval}',
             fontsize=15,color=ink,fontweight='bold')
    fig.text(.10,.015,'States, requests and commands are distinguished in events.csv. None alone proves physical execution.',
             fontsize=9,color='#5E7079')
    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'hri_case_study_D_S1a.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bag', type=Path, help='ROS 2 bag directory containing metadata.yaml')
    ap.add_argument('--out', type=Path, help='Output directory (default: sibling poster_outputs)')
    ap.add_argument('--t0', type=float, help='Override analysis start in seconds since UNIX epoch')
    ap.add_argument('--t1', type=float, help='Override analysis end in seconds since UNIX epoch')
    ap.add_argument('--window', choices=['goal-to-end', 'bag', 'success'], default='goal-to-end',
                    help='Default keeps late events; success uses the older success-status cutoff.')
    ap.add_argument('--topic', action='append', default=[], help='Additional topic to decode; repeat as needed.')
    ap.add_argument('--event-map', type=Path,
                    help='JSON: {"/topic|field.path": {"raw value": "REPLAN"}}; explicit schema-specific mappings.')
    args=ap.parse_args()
    bag=args.bag.expanduser().resolve()
    output=args.out.expanduser().resolve() if args.out else bag.parent / 'poster_outputs'
    event_map = json.loads(args.event_map.read_text()) if args.event_map else {}
    allowed = {'WAIT', 'REPLAN', 'RESUME', 'START', 'STOP'}
    if not isinstance(event_map, dict) or any(
        not isinstance(k, str) or '|' not in k or not isinstance(v, dict)
        or any(not isinstance(x, str) or x not in allowed for x in v.values())
        for k, v in event_map.items()):
        raise ValueError('event-map must map /topic|field.path to raw-value/category objects.')
    extra_topics = set(args.topic) | {key.split('|', 1)[0] for key in event_map}
    rows, types=read_bag(bag, extra_topics)
    t0,t1,note=choose_window(rows, args.window)
    if args.t0 is not None:
        t0=args.t0
    if args.t1 is not None:
        t1=args.t1
    if args.t0 is not None or args.t1 is not None:
        note += '; explicit --t0/--t1 overrides applied'
    if t1 <= t0:
        raise ValueError('Analysis end must be later than analysis start.')
    diagnostics = []
    events=detect_events(rows,t0,t1,event_map,diagnostics)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'event_detection_diagnostics.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['time_unix_s','time_s','topic','field','raw_value','resolved_event',
                  'resolution','inside_window','repeated_state']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(diagnostics)
    inventory = {topic: {'type': types[topic], 'decoded_messages': len(rows.get(topic, [])),
                         'first_decoded': rows[topic][0][0] if rows.get(topic) else None,
                         'last_decoded': rows[topic][-1][0] if rows.get(topic) else None}
                 for topic in sorted(types)}
    (output / 'topic_inventory.json').write_text(json.dumps(inventory, indent=2))
    samples = {topic: [data for _, data in messages[:3]] for topic, messages in rows.items()
               if topic.startswith(('/social_navigation/', '/hri/')) or topic in extra_topics}
    (output / 'message_samples.json').write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    path,stationary,odom_topic,n,segments=motion_metrics(rows,t0,t1)
    duration=t1-t0
    dialogue=extract_dialogue(rows,t0,t1)
    make_figure(output,duration,path,stationary,events,segments,note,dialogue)
    with (output / 'dialogue_transcript.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['time_s','source','topic','text'])
        writer.writeheader();writer.writerows(dialogue)
    print(f'Dialogue/text entries exported: {len(dialogue)}')
    with (output / 'events.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['time_s','event','topic','field','evidence','raw_value','detail'])
        writer.writeheader(); writer.writerows(events)
    summary={'bag_path':str(bag),'analysis_start_unix_s':t0,'analysis_end_unix_s':t1,
             'analysis_window_note':note,'duration_s':round(duration,3),
             'stationary_s':None if stationary is None else round(stationary,3),
             'path_length_m':None if path is None else round(path,3),
             'odom_topic':odom_topic,'odom_samples':n,
             'decoded_topics':sorted(rows),
             'event_evidence_counts':dict(Counter(e['evidence'].split(';')[0] for e in events)),
             'recognized_observations_outside_window':sum(bool(d['resolved_event']) and not d['inside_window'] for d in diagnostics),
             'counts':{typ:sum(x['event']==typ for x in events)
                       for typ in sorted({e['event'] for e in events})},
             'caveats':['ROS bag receive time, not synchronized video time',
                        'Stationary duration includes all causes of stopping',
                        'Commands do not prove physical execution',
                        'Success status may correspond to a later public/internal goal; review trace',
                        'Path length uses odometry and excludes gaps >1 s / jumps >1 m']}
    (output / 'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Saved:',output)
    print(json.dumps({k:summary[k] for k in ('analysis_window_note','duration_s','stationary_s','path_length_m','odom_samples','counts')},indent=2))
    print('Review events.csv, event_detection_diagnostics.csv and topic_inventory.json before citing results.')
    for kind in ('REPLAN', 'RESUME'):
        if not any(e['event'] == kind for e in events):
            print(f'WARNING: no {kind} recognized. This is NOT proof of absence. '
                  'Inspect diagnostics, message_samples.json, topic availability and message definitions.', file=sys.stderr)

if __name__=='__main__':
    main()
