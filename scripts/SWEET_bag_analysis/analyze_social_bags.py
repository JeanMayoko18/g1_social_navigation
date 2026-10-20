#!/usr/bin/env python3
"""Read-only ROS 2 social-navigation analysis. No ROS runtime or robot commands.

Outputs are descriptive estimates, not automatically validated experimental results.
See README.md for metric definitions, message schemas and explicit trial windows.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg, get_types_from_idl

DEFAULTS = {
    'odom_topic': '/odom',
    'human_topic': '/social_perception/tracked_persons',
    'human_array': 'persons',
    'human_position': 'position',
    'human_id': 'track_id',
    'human_valid_field': None,
    'human_valid_values': [],
    'human_stamp': 'header.stamp',
    'human_frame': 'header.frame_id',
    'base_frames': ['base_footprint', 'base_link', 'pelvis'],
    'max_human_age_s': 0.5,
    'sync_tolerance_s': 0.15,
    'max_gap_s': 0.5,
    'stop_speed_mps': 0.03,
    'stop_yaw_rps': 0.05,
    'minimum_stop_s': 0.5,
    'max_plausible_speed_mps': 1.0,
    'minimum_pose_step_m': 0.0,
    'social_distance_m': None,
    'plot_step_s': 0.1,
    'hri_context_s': 8.0,
    'max_examples': 8,
    'speech_id_fields': ['utterance_id', 'speech_id', 'request_id', 'decision_id', 'context_id'],
    'speech_text_fields': ['text', 'speech', 'utterance', 'transcript', 'data'],
    'speech_status_fields': ['status', 'state'],
    'speech_started_values': ['started', 'speaking', 'playing'],
    'speech_finished_values': ['completed', 'finished', 'done'],
    'event_state_fields': ['state', 'navigation_state', 'status', 'event', 'decision', 'action', 'reason', 'data'],
}
SPEECH_TEXT = {'/explainability/text', '/dialogue_manager/robot_speech'}
HUMAN_TEXT = {'/dialogue_manager/closed_captions', '/humans/voices/pc_user/speech'}
SPEECH_STATUS = {'/explainability/speech_status', '/explainability/pc_audio_fallback/status'}
NAV_STATUS = '/navigate_to_pose/_action/status'
STATUS = {0: 'unknown', 1: 'accepted', 2: 'executing', 3: 'canceling',
          4: 'succeeded', 5: 'canceled', 6: 'aborted'}
COLORS = {'A': '#526779', 'C': '#dc922e', 'D': '#137e76', 'unknown': '#7c698b'}


def value(obj, path, default=None):
    for key in str(path).split('.'):
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def plain(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if not f.name.startswith('__')}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def stamp(d):
    if not isinstance(d, dict):
        return None
    try:
        x = float(d['sec']) + float(d['nanosec']) * 1e-9
        return x if x > 0 else None
    except (KeyError, ValueError, TypeError):
        return None


def number(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def first(d, paths):
    for path in paths:
        x = value(d, path)
        if x is not None and x != '':
            return x
    return None


def text_value(d, fields):
    x = first(d, fields)
    return x.strip() if isinstance(x, str) else ''


def latex(s):
    replace = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(replace.get(c, c) for c in str(s)).replace('\n', ' ')


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def csv_out(path, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def discover(root):
    if root.is_file() and root.suffix.lower() == '.mcap':
        return [root]
    if (root / 'metadata.yaml').is_file():
        return [root]
    bags = sorted(p.parent for p in root.rglob('metadata.yaml'))
    bare_mcap = [p for p in root.rglob('*.mcap') if not any(b == p.parent for b in bags)]
    return bags + sorted(bare_mcap)


def type_store(distro, roots):
    store = get_typestore(getattr(Stores, 'ROS2_' + distro.upper()))
    definitions, warnings = {}, []
    for root in roots:
        root = Path(root).expanduser()
        files = list(root.rglob('*.msg')) + list(root.rglob('*.idl')) if root.exists() else []
        if not files:
            warnings.append(f'No message definitions found under {root}')
        for path in files:
            # Ignore generated/build trees when a source tree was supplied.
            if any(x in path.parts for x in ['.git', 'build', 'log']):
                continue
            try:
                if path.suffix == '.idl':
                    parsed = get_types_from_idl(path.read_text())
                else:
                    parent = path.parent
                    pkg = None
                    for ancestor in [parent, *parent.parents]:
                        if (ancestor / 'package.xml').is_file():
                            pkg = ET.parse(ancestor / 'package.xml').getroot().findtext('name')
                            break
                        if ancestor == root.parent:
                            break
                    if pkg is None and parent.name == 'msg':
                        pkg = parent.parent.name
                    if pkg is None:
                        continue
                    parsed = get_types_from_msg(path.read_text(), f'{pkg}/msg/{path.stem}')
                for key, definition in parsed.items():
                    if key not in store.types:
                        if key in definitions and definitions[key] != definition:
                            warnings.append(f'Conflicting definition for {key}: {path}')
                        else:
                            definitions[key] = definition
            except Exception as exc:
                warnings.append(f'Type parse failed: {path}: {exc}')
    if definitions:
        store.register(definitions)
    return store, warnings


def wanted(topic, msgtype, cfg):
    return (topic == cfg['odom_topic'] or topic == cfg['human_topic'] or
            topic.startswith(('/social_navigation/', '/explainability/', '/hri/', '/dialogue_manager/')) or
            topic in HUMAN_TEXT or topic in ['/intents', '/goal_pose', '/amcl_pose', '/plan', '/plan_smoothed',
                                            '/local_plan', '/received_global_plan'] or
            topic.startswith('/cmd_vel') or topic.endswith('/_action/status') or
            topic == '/social_perception/social_state')


def parse_odom(d, t):
    pos = value(d, 'pose.pose.position')
    vel = value(d, 'twist.twist.linear')
    if not isinstance(pos, dict) or not isinstance(vel, dict):
        return None
    vals = [number(pos.get('x')), number(pos.get('y')), number(vel.get('x')),
            number(vel.get('y')), number(value(d, 'twist.twist.angular.z'))]
    if any(x is None for x in vals):
        return None
    x, y, vx, vy, wz = vals
    return {'t': t, 'x': x, 'y': y, 'speed': math.hypot(vx, vy), 'wz': wz,
            'frame': str(value(d, 'header.frame_id', '')).lstrip('/'),
            'header_stamp': stamp(value(d, 'header.stamp'))}


def read_trial(path, store, cfg, out):
    errors, inventory, schemas = [], [], {}
    odom, humans, events, hri, goals, statuses, commands, plans = [], [], [], [], [], [], [], []
    decoded = Counter()
    last_odom, last_command = -math.inf, defaultdict(lambda: -math.inf)
    last_plan = defaultdict(lambda: -math.inf)
    parse_errors = Counter()
    with AnyReader([path], default_typestore=store) as reader:
        origin = reader.start_time * 1e-9
        end = reader.end_time * 1e-9 - origin
        for c in reader.connections:
            inventory.append({'topic': c.topic, 'type': c.msgtype, 'recorded_count': c.msgcount,
                              'selected_for_decode': wanted(c.topic, c.msgtype, cfg)})
        conns = [c for c in reader.connections if wanted(c.topic, c.msgtype, cfg)]
        with (out / 'event_messages.jsonl').open('w') as raw_events:
            iterator = reader.messages(connections=conns) if conns else []
            for c, ns, raw in iterator:
                t = ns * 1e-9 - origin
                try:
                    msg = reader.deserialize(raw, c.msgtype)
                    d = plain(msg)
                    decoded[c.topic] += 1
                except Exception as exc:
                    key = (c.topic, c.msgtype, str(exc))
                    parse_errors[key] += 1
                    continue
                if c.topic not in schemas:
                    schemas[c.topic] = {'type': c.msgtype, 'sample': d}
                if c.topic == cfg['odom_topic']:
                    # Retain full-rate motion for interval integration and path length.
                    row = parse_odom(d, t)
                    if row:
                        odom.append(row)
                    continue
                if c.topic.startswith('/cmd_vel'):
                    if t - last_command[c.topic] >= cfg['plot_step_s']:
                        tw = d.get('twist', d)
                        vx, vy = number(value(tw, 'linear.x')), number(value(tw, 'linear.y'))
                        wz = number(value(tw, 'angular.z'))
                        if vx is not None and vy is not None and wz is not None:
                            commands.append({'t': t, 'topic': c.topic, 'speed': math.hypot(vx, vy), 'wz': wz})
                            last_command[c.topic] = t
                    continue
                if c.topic == cfg['human_topic']:
                    people = value(d, cfg['human_array'])
                    if not isinstance(people, list):
                        parse_errors[(c.topic, c.msgtype, 'Human array mapping does not match schema')] += 1
                        continue
                    if people and 'nonempty_sample' not in schemas[c.topic]:
                        schemas[c.topic]['nonempty_sample'] = d
                    for person in people:
                        if cfg['human_valid_field']:
                            flag = value(person, cfg['human_valid_field'])
                            if flag not in cfg['human_valid_values']:
                                continue
                        pos = value(person, cfg['human_position'])
                        if not isinstance(pos, dict):
                            parse_errors[(c.topic, c.msgtype, 'Human position mapping does not match schema')] += 1
                            continue
                        x, y = number(pos.get('x')), number(pos.get('y'))
                        if x is None or y is None:
                            continue
                        frame = value(person, cfg['human_frame']) or value(d, 'header.frame_id', '')
                        observed_stamp = stamp(value(person, cfg['human_stamp']))
                        # Do not use collection publication time as individual observation freshness.
                        humans.append({'t': t, 'id': str(value(person, cfg['human_id'], 'unknown')),
                                       'x': x, 'y': y, 'frame': str(frame).lstrip('/'),
                                       'observation_stamp': observed_stamp,
                                       'collection_stamp': stamp(value(d, 'header.stamp'))})
                    continue
                if c.msgtype == 'nav_msgs/msg/Path':
                    if t - last_plan[c.topic] >= 2.0 and len(plans) < 100:
                        pts = [[value(p, 'pose.position.x'), value(p, 'pose.position.y')] for p in d.get('poses', [])]
                        plans.append({'t': t, 'topic': c.topic, 'frame': value(d, 'header.frame_id'), 'points': pts})
                        last_plan[c.topic] = t
                    continue
                if c.topic == '/goal_pose':
                    goals.append({'t': t, 'data': d})
                if c.topic.endswith('/_action/status'):
                    for st in d.get('status_list', []):
                        uid = value(st, 'goal_info.goal_id.uuid', [])
                        gid = bytes(uid).hex() if isinstance(uid, list) else str(uid)
                        statuses.append({'t': t, 'topic': c.topic, 'goal_id': gid,
                                         'status': STATUS.get(st.get('status'), str(st.get('status')))})
                if c.topic.startswith(('/social_navigation/', '/explainability/', '/hri/', '/dialogue_manager/')) or c.topic in HUMAN_TEXT or c.topic == '/intents':
                    record = {'t': t, 'topic': c.topic, 'data': d}
                    raw_events.write(json.dumps(record, ensure_ascii=False) + '\n')
                    state = first(d, cfg['event_state_fields'])
                    events.append({'t': t, 'topic': c.topic, 'label': str(state)[:160] if state is not None else '', 'data': d})
                if c.topic in SPEECH_TEXT | HUMAN_TEXT | SPEECH_STATUS or c.topic in ['/hri/human_intent', '/intents']:
                    role = ('robot_text' if c.topic in SPEECH_TEXT else
                            'caption_unverified_speaker' if c.topic == '/dialogue_manager/closed_captions' else
                            'human_transcript' if c.topic in HUMAN_TEXT else
                            'speech_status' if c.topic in SPEECH_STATUS else 'interpreted_intent')
                    text = text_value(d, cfg['speech_text_fields'])
                    st = first(d, cfg['speech_status_fields'])
                    hri.append({'t': t, 'topic': c.topic, 'role': role, 'text': text,
                                'speech_id': str(first(d, cfg['speech_id_fields']) or ''),
                                'status': '' if st is None else str(st),
                                'detail': str(d.get('detail','')),
                                'intent': str(d.get('intent','')),
                                'source': str(d.get('source','')), 'data': d})
        for (topic, typ, err), count in parse_errors.items():
            errors.append({'topic': topic, 'type': typ, 'count': count, 'error': err})
    for row in inventory:
        row['decoded_count'] = decoded[row['topic']]
    csv_out(out / 'topics_inventory.csv', inventory)
    dump(out / 'schema_samples.json', schemas)
    dump(out / 'decode_errors.json', errors)
    dump(out / 'sampled_plans.json', plans)
    return dict(origin=origin, end=end, odom=odom, humans=humans, events=events, hri=hri,
                goals=goals, statuses=statuses, commands=commands, inventory=inventory, errors=errors)


def window_for(data, manual):
    if manual and manual.get('start_s') not in [None, ''] and manual.get('end_s') not in [None, '']:
        lo, hi = float(manual['start_s']), float(manual['end_s'])
        if not 0 <= lo < hi <= data['end'] + 0.1:
            raise ValueError('Manual analysis window is outside the bag or reversed')
        return lo, hi, 'manual', str(manual.get('outcome', '') or 'unknown')
    ss = [s for s in data['statuses'] if s['topic'] == NAV_STATUS]
    # Status arrays retain terminal goals from earlier runs. Only goals observed
    # active in this recording are candidates for an inferred trial window.
    gids = {s['goal_id'] for s in ss if s['status'] in ['accepted', 'executing', 'canceling']}
    ss = [s for s in ss if s['goal_id'] in gids]
    if len(gids) == 1:
        active = [s for s in ss if s['status'] in ['accepted', 'executing']]
        active.sort(key=lambda s: s['t'])
        terminal = sorted([s for s in ss if s['status'] in ['succeeded', 'aborted']
                           and active and s['t'] > active[0]['t']], key=lambda s: s['t'])
        if active and terminal and terminal[0]['t'] > active[0]['t']:
            return active[0]['t'], terminal[0]['t'], 'inferred_single_public_goal_status', 'nav2_' + terminal[0]['status']
    return 0.0, data['end'], 'recording_only', 'unknown'


def intervals(t, mask, max_gap, minimum=0.0):
    output, start, end = [], None, None
    for i in range(max(0, len(t)-1)):
        good = mask[i] and mask[i+1] and 0 < t[i+1]-t[i] <= max_gap
        if good:
            if start is None:
                start = t[i]
            end = t[i+1]
        elif start is not None:
            if end-start >= minimum:
                output.append((float(start), float(end)))
            start = end = None
    if start is not None and end-start >= minimum:
        output.append((float(start), float(end)))
    return output


def motion_metrics(rows, lo, hi, cfg):
    df = pd.DataFrame([r for r in rows if lo <= r['t'] <= hi])
    metrics = {'path_length_observed_m': None, 'stationary_time_observed_s': None,
               'stop_bouts': None, 'motion_time_coverage': 0., 'pose_segments_rejected': 0,
               'pose_time_coverage': 0.}
    if len(df) < 2:
        return df, metrics, []
    df = df.sort_values('t').drop_duplicates('t')
    t = df.t.to_numpy(); dt = np.diff(t)
    valid = (dt > 0) & (dt <= cfg['max_gap_s'])
    sameframe = df.frame.to_numpy()[1:] == df.frame.to_numpy()[:-1]
    steps = np.hypot(np.diff(df.x), np.diff(df.y))
    pos_ok = valid & sameframe & (steps <= cfg['max_plausible_speed_mps']*dt)
    metrics['path_length_observed_m'] = float(steps[pos_ok & (steps >= cfg['minimum_pose_step_m'])].sum())
    metrics['pose_segments_rejected'] = int((~pos_ok).sum())
    metrics['pose_time_coverage'] = float(dt[pos_ok].sum()/(hi-lo))
    metrics['motion_time_coverage'] = float(dt[valid].sum()/(hi-lo))
    stopped = ((df.speed.to_numpy() <= cfg['stop_speed_mps']) &
               (np.abs(df.wz.to_numpy()) <= cfg['stop_yaw_rps']))
    bouts = intervals(t, stopped, cfg['max_gap_s'], cfg['minimum_stop_s'])
    metrics['stationary_time_observed_s'] = sum(b-a for a,b in bouts)
    metrics['stop_bouts'] = len(bouts)
    return df, metrics, bouts


def human_distances(data, motion, lo, hi, cfg, trusted):
    rows, excluded = [], Counter()
    motion_times = motion.t.to_numpy() if not motion.empty else np.array([])
    for h in data['humans']:
        if not lo <= h['t'] <= hi:
            continue
        distance = None
        frame = h['frame']
        freshness = False
        if h['observation_stamp'] is not None:
            age = data['origin'] + h['t'] - h['observation_stamp']
            freshness = -0.05 <= age <= cfg['max_human_age_s']
        else:
            age = None
        if frame in cfg['base_frames']:
            distance = math.hypot(h['x'], h['y'])
            basis = f'person_to_{frame}_origin'
        elif not motion.empty and frame:
            ins = int(np.searchsorted(motion_times, h['t']))
            candidates = [i for i in [ins-1, ins] if 0 <= i < len(motion_times)]
            idx = min(candidates, key=lambda i: abs(motion_times[i]-h['t']))
            r = motion.iloc[idx]
            if r.frame == frame and abs(r.t-h['t']) <= cfg['sync_tolerance_s']:
                distance = math.hypot(h['x']-r.x, h['y']-r.y)
                basis = f'person_to_odometry_origin_in_{frame}'
        if distance is None:
            excluded['frame_or_synchronization_unavailable'] += 1
            continue
        rows.append({'t': h['t'], 'person_id': h['id'], 'distance_estimate_m': distance,
                     'observation_age_s': age, 'basis': basis,
                     'eligible_for_metrics': bool(trusted and freshness)})
        if not freshness:
            excluded['freshness_unverified_or_stale'] += 1
    df = pd.DataFrame(rows)
    metrics = {'human_min_distance_estimate_m': None, 'human_observed_interval_s': None,
               'intrusion_time_observed_s': None, 'human_time_coverage': 0.}
    if not df.empty:
        valid = df[df.eligible_for_metrics].groupby('t', as_index=False).distance_estimate_m.min()
        if not valid.empty:
            metrics['human_min_distance_estimate_m'] = float(valid.distance_estimate_m.min())
            t = valid.t.to_numpy()
            coverage = intervals(t, np.ones(len(t), bool), cfg['max_gap_s'])
            duration = sum(b-a for a,b in coverage)
            metrics['human_observed_interval_s'] = duration
            metrics['human_time_coverage'] = duration/(hi-lo)
            if cfg['social_distance_m'] is not None:
                intr = intervals(t, valid.distance_estimate_m.to_numpy() < cfg['social_distance_m'], cfg['max_gap_s'])
                metrics['intrusion_time_observed_s'] = sum(b-a for a,b in intr)
    return df, metrics, dict(excluded)


def evidence_examples(data, cfg):
    # Keep every recorded message; collapse only consecutive repeated publications
    # with the same topic, nonempty ID, text and status within one second.
    unique, previous = [], None
    for row in data['hri']:
        key = (row['topic'], row['speech_id'], row['text'], row['status'])
        if row['speech_id'] and previous and key == previous[0] and row['t']-previous[1] < 1:
            continue
        unique.append(row)
        previous = (key, row['t'])
    selected, categories = [], set()
    for row in unique:
        if row['role'] != 'robot_text' or not row['text']:
            continue
        nearby = [e for e in data['events'] if 0 <= row['t']-e['t'] <= cfg['hri_context_s'] and
                  e['topic'].startswith('/social_navigation/') and e['label']]
        context = nearby[-1]['label'] if nearby else 'context_not_identified'
        # Select first example per recorded context, no success-based filtering.
        category = (row['topic'], context)
        if category in categories:
            continue
        categories.add(category)
        next_text = min((s['t'] for s in unique if s['role']=='robot_text'
                         and s['speech_id']==row['speech_id'] and s['t']>row['t']), default=math.inf)
        statuses = [s for s in unique if s['t'] < next_text and s['role']=='speech_status' and row['speech_id'] and
                    s['speech_id'] == row['speech_id'] and s['t'] >= row['t']]
        started = [s for s in statuses if s['status'].lower() in cfg['speech_started_values']]
        finished = [s for s in statuses if s['status'].lower() in cfg['speech_finished_values']]
        selected.append({'t': row['t'], 'topic': row['topic'], 'text': row['text'],
                         'context_label_nearby': context, 'speech_id': row['speech_id'],
                         'delivery_evidence': 'completion_status_matched_id' if finished else
                                             'start_status_matched_id' if started else 'unverified',
                         'text_to_start_s': started[0]['t']-row['t'] if started else None,
                         'matched_statuses': [{k:s[k] for k in ['t','status']} for s in statuses],
                         'window': [{k:v for k,v in r.items() if k!='data'} for r in unique
                                    if abs(r['t']-row['t']) <= cfg['hri_context_s']]})
        if len(selected) >= cfg['max_examples']:
            break
    return unique, selected


def savefig(fig, base):
    fig.tight_layout()
    fig.savefig(str(base)+'.png', dpi=200, bbox_inches='tight')
    fig.savefig(str(base)+'.pdf', bbox_inches='tight')
    plt.close(fig)


def figures(out, df, distances, commands, events, examples, bouts, lo, hi, label, cfg):
    figure_names = []
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        # Break trajectories at recording gaps, pose jumps and reference-frame changes.
        good = ((df.t.diff() <= cfg['max_gap_s']) & (df.frame == df.frame.shift()) &
                (np.hypot(df.x.diff(),df.y.diff()) <= cfg['max_plausible_speed_mps']*df.t.diff()))
        starts = np.flatnonzero(~good.to_numpy())
        for a,b in zip(starts, list(starts[1:])+[len(df)]):
            ax.plot(df.x.iloc[a:b],df.y.iloc[a:b],color=COLORS.get(label,'#137e76'),lw=1.6)
        ax.scatter(df.x.iloc[0],df.y.iloc[0],marker='o',label='Window start',color='#137e76')
        ax.scatter(df.x.iloc[-1],df.y.iloc[-1],marker='X',label='Window end',color='#c24940')
        ax.set(xlabel='x [m]',ylabel='y [m]',title=f'{label}: estimated robot trajectory')
        ax.axis('equal'); ax.grid(alpha=.2);ax.legend(fontsize=8)
        savefig(fig,out/'trajectory');figure_names.append('trajectory')
    fig, axs = plt.subplots(3,1,figsize=(8,7),sharex=True)
    if not df.empty:
        # NaN separators avoid implying continuous motion across long gaps.
        sp=df.speed.copy();wz=df.wz.copy(); bad=df.t.diff()>cfg['max_gap_s'];sp[bad]=np.nan;wz[bad]=np.nan
        axs[0].plot(df.t-lo,sp,label='Measured planar speed',color='#137e76')
        axs[1].plot(df.t-lo,wz,label='Measured yaw rate',color='#526779')
        for a,b in bouts:
            axs[0].axvspan(a-lo,b-lo,color='#d6b763',alpha=.25)
    cmd = pd.DataFrame([c for c in commands if lo<=c['t']<=hi])
    if not cmd.empty:
        for topic,g in cmd.groupby('topic'):
            axs[0].plot(g.t-lo,g.speed,alpha=.5,lw=.7,label=topic)
    axs[0].set_ylabel('Speed [m/s]');axs[1].set_ylabel('Yaw rate [rad/s]')
    if not distances.empty:
        for pid,g in distances.groupby('person_id'):
            # Scatter only: missing people do not imply clearance.
            axs[2].scatter(g.t-lo,g.distance_estimate_m,s=5,label=f'Track {pid}')
        if cfg['social_distance_m'] is not None:
            axs[2].axhline(cfg['social_distance_m'],ls='--',color='#c24940',label='Analysis threshold')
    else:
        axs[2].text(.5,.5,'No usable human position mapping',transform=axs[2].transAxes,ha='center')
    axs[2].set(ylabel='Estimated distance [m]',xlabel='Time from analysis-window start [s]')
    for ax in axs:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6,loc='best')
    axs[0].set_title(f'{label}: recorded motion and human observations')
    savefig(fig,out/'motion_distance');figure_names.append('motion_distance')
    state_rows=[];last={}
    for e in events:
        if lo<=e['t']<=hi and e['label'] and e['topic'].startswith('/social_navigation/'):
            if last.get(e['topic'])!=e['label']:
                state_rows.append(e);last[e['topic']]=e['label']
    lanes=sorted({e['topic'] for e in state_rows})[:7]
    if lanes or examples:
        fig,ax=plt.subplots(figsize=(10,max(3, .45*(len(lanes)+2))))
        for j,topic in enumerate(lanes):
            for e in [r for r in state_rows if r['topic']==topic][:18]:
                ax.scatter(e['t']-lo,j,s=20,color='#137e76')
                ax.annotate(e['label'][:28],(e['t']-lo,j),xytext=(2,5),textcoords='offset points',fontsize=5,rotation=30)
        for i,e in enumerate(examples):
            if lo<=e['t']<=hi:
                ax.scatter(e['t']-lo,len(lanes),marker='s',color='#b46f22')
                ax.annotate(f'M{i+1}',(e['t']-lo,len(lanes)),xytext=(0,6),textcoords='offset points',fontsize=7)
        ax.set_yticks(range(len(lanes)+1),[p.split('/')[-1] for p in lanes]+['Robot text'])
        ax.set(xlabel='Time from analysis-window start [s]',title='Recorded event transitions and message examples')
        ax.grid(axis='x',alpha=.2)
        savefig(fig,out/'event_timeline');figure_names.append('event_timeline')
    for k, example in enumerate(examples[:3], start=1):
        utterances = [r for r in example['window'] if r['text'] and r['role'] != 'speech_status'][:7]
        if not utterances:
            continue
        wrapped = []
        for r in utterances:
            words = textwrap.fill(r['text'], width=88)
            wrapped.append((r, words))
        height = max(3., sum(.55 + .22*(txt.count('\n')+1) for _,txt in wrapped))
        fig, ax = plt.subplots(figsize=(9,height))
        ax.axis('off')
        y = 1.0
        for r, txt in wrapped:
            color = '#137e76' if r['role']=='robot_text' else '#526779'
            ax.text(.01,y, f"{r['t']:.3f} s | {r['role']} | {r['topic']}", transform=ax.transAxes,
                    fontsize=8,color=color,va='top',weight='bold')
            y -= .28/height
            ax.text(.01,y,txt,transform=ax.transAxes,fontsize=9,va='top')
            y -= (.32+.22*(txt.count('\n')+1))/height
        ax.set_title('Recorded HRI excerpt — time from bag start; transcripts are not verified audio', fontsize=9,pad=18)
        name=f'hri_exchange_{k:02d}'
        savefig(fig,out/name);figure_names.append(name)
    return figure_names


def trial_meta(path):
    folder=path.parent if path.is_file() else path
    for candidate in [folder/'trial_metadata.yaml',folder.parent/'trial_metadata.yaml']:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text()) or {}
    name=folder.parent.name if folder.name=='navigation' else folder.name
    m=re.match(r'(technical|participant)_([ACD])_(S\d[a-z]?)_R(\d+)',name)
    return {'trial_id':name,'trial_type':m[1] if m else 'unknown',
            'configuration':{'id':m[2] if m else 'unknown'},'scenario':{'id':m[3] if m else 'unknown'}}


def make_report(out, summary, metrics, examples, errors, figures_list):
    lines=['# Trial analysis', '',f"Trial: {summary['trial_id']}",f"Window: {summary['window_source']}",
           '', 'Values are estimates from recorded messages. Missing data are not zero.',
           'Stationary time is not automatically social waiting; minimum distance is to a frame origin, not body clearance.',
           '', '## Metrics','']
    for k,v in metrics.items():lines.append(f'- {k}: {v}')
    lines += ['', '## Extracted robot messages', '',
              'First example per recorded context, chronological selection. Review before publication.',
              'Nearby events are temporal context, not proof of causality. No translations are generated.','']
    tex=[r'\begin{table*}[t]',r'\centering\small',
         r'\caption{Recorded robot messages. Relative times refer to bag start. Delivery status is software evidence; audibility requires separate confirmation.}',
         r'\begin{tabularx}{\textwidth}{@{}p{.06\textwidth}p{.10\textwidth}Xp{.23\textwidth}@{}}',
         r'\toprule ID & Time [s] & Verbatim text & Delivery evidence\\\midrule']
    for i,e in enumerate(examples):
        lines += [f"### M{i+1} — t={e['t']:.3f} s", '',e['text'],'',f"Delivery: {e['delivery_evidence']}",
                  f"Nearby context: {e['context_label_nearby']}",'','```text']
        for r in e['window']:
            lines.append(f"{r['t']:.3f} | {r['role']} | {r['text'] or r['status']}")
        lines += ['```','']
        tex.append(f"M{i+1} & {e['t']:.2f} & {latex(e['text'])} & {latex(e['delivery_evidence'])} \\\\")
    tex += [r'\bottomrule\end{tabularx}',r'\end{table*}']
    (out/'hri_examples.tex').write_text('\n'.join(tex)+'\n')
    for name in figures_list:lines += [f'![{name}]({name}.png)','']
    lines += ['## Decode/schema issues','',f'{len(errors)} issue categories. See decode_errors.json and schema_samples.json.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def aggregate(out, summaries):
    df=pd.DataFrame(summaries);df.to_csv(out/'trial_summary.csv',index=False)
    valid=df[df.analysis_status=='ok'].copy()
    metrics=['window_duration_s','path_length_observed_m','stationary_time_observed_s','human_min_distance_estimate_m']
    grouped=[]
    # Never pool technical and participant campaigns or manual and recording-only windows.
    for key,g in valid.groupby(['trial_type','configuration','scenario','window_source'],dropna=False):
        row=dict(zip(['trial_type','configuration','scenario','window_source'],key));row['n_trials']=len(g)
        row['outcomes']=json.dumps(dict(Counter(g.outcome)),sort_keys=True)
        manual_outcomes=g[g.window_source=='manual'].outcome
        classified=manual_outcomes[manual_outcomes.isin(['success','timeout','intervention','aborted','failure'])]
        row['manual_classified_n']=len(classified)
        row['manual_success_n']=int((classified=='success').sum())
        row['manual_success_rate']=float((classified=='success').mean()) if len(classified) else None
        for metric in metrics:
            x=pd.to_numeric(g[metric],errors='coerce').dropna()
            row[metric+'_n']=len(x)
            row[metric+'_median']=float(x.median()) if len(x) else None
            row[metric+'_q25']=float(x.quantile(.25)) if len(x) else None
            row[metric+'_q75']=float(x.quantile(.75)) if len(x) else None
        grouped.append(row)
    csv_out(out/'group_summary.csv',grouped)
    for (kind,source),g in valid.groupby(['trial_type','window_source']):
        if g.empty:continue
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,metric in zip(axes,metrics[:3]):
            groups=list(g.groupby(['scenario','configuration']))
            for i,((sc,conf),sub) in enumerate(groups):
                vals=pd.to_numeric(sub[metric],errors='coerce').dropna().to_numpy()
                if len(vals):
                    ax.scatter(i+np.linspace(-.08,.08,len(vals)),vals,color=COLORS.get(conf,'#7c698b'),s=24)
                    ax.plot([i-.18,i+.18],[np.median(vals)]*2,color='black',lw=1.5)
            ax.set_xticks(range(len(groups)),[f'{sc}/{c}' for (sc,c),_ in groups],rotation=45,ha='right')
            ax.set_title(metric.replace('_',' '),fontsize=9);ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{kind} — {source}; dots = trials, bars = medians',fontsize=10)
        savefig(fig,out/f'comparison_{kind}_{source}')
    tex=[r'\begin{table*}[t]',r'\centering\scriptsize',r'\caption{Descriptive bag analysis. Durations refer to the stated analysis window; NA means unavailable.}',
         r'\begin{tabular}{llllrrrr}',r'\toprule Type & Scenario & Mode & Window & $n$ & Duration [s] & Path [m] & Stationary [s]\\\midrule']
    fmt=lambda x:'NA' if x is None or pd.isna(x) else f'{x:.2f}'
    for r in grouped:
        tex.append(' & '.join([latex(r['trial_type']),latex(r['scenario']),latex(r['configuration']),
                    latex({'inferred_single_public_goal_status':'inferred goal', 'recording_only':'recording', 'manual':'manual'}.get(r['window_source'],r['window_source'])),str(r['n_trials'])]+[fmt(r[m+'_median']) for m in metrics[:3]])+r'\\')
    tex += [r'\bottomrule\end{tabular}',r'\end{table*}']
    (out/'navigation_summary.tex').write_text('\n'.join(tex)+'\n')
    report=['# Campaign analysis','',f'Bags discovered: {len(summaries)}',f'Bags analysed: {len(valid)}','',
            'Figures show trial-level descriptive outcomes, not independent participant observations.',
            'No significance tests or automatic trust/comprehension scores are produced.',
            'Review inferred mission windows and all NA values before using the tables.',
            'Raw speech exports may contain identifying information; review excerpts before publication.','',
            '## Trial reports','']
    for r in summaries:
        report.append(f"- [{r['trial_id']}](trials/{r['output_id']}/report.md): {r['analysis_status']}")
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')

def select_latest_trials(bags):
    """Keep the highest repetition per participant, configuration and scenario."""
    groups = defaultdict(list)

    for path in bags:
        meta = trial_meta(path)
        trial_id = str(meta.get('trial_id', ''))
        match = re.search(r'_R(\d+)$', trial_id)

        if not match:
            raise ValueError(
                f'Cannot identify repetition number: {trial_id} ({path})'
            )

        key = (
            str(meta.get('trial_type', 'unknown')),
            str(meta.get('participant_id', '') or ''),
            str(value(meta, 'configuration.id', 'unknown')),
            str(value(meta, 'scenario.id', 'unknown')),
        )
        groups[key].append((int(match.group(1)), path))

    selected = []

    for key, candidates in sorted(groups.items()):
        latest = max(repetition for repetition, _ in candidates)
        winners = [
            path for repetition, path in candidates
            if repetition == latest
        ]

        if len(winners) != 1:
            paths = '\n'.join(str(path) for path in winners)
            raise ValueError(
                f'Ambiguous latest repetition R{latest:02d} for {key}:\n'
                f'{paths}\n'
                'Resolve duplicate trial identifiers before analysis.'
            )

        selected.append(winners[0])
        print(f'[SELECT] R{latest:02d}: {winners[0]}')

        for repetition, path in sorted(candidates):
            if repetition < latest:
                print(f'[SKIP] Earlier repetition R{repetition:02d}: {path}')

    return sorted(selected)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path,help='Campaign directory, one ROS 2 bag directory, or one MCAP file')
    ap.add_argument('--output',type=Path,required=True,help='New output directory; existing directories are rejected')
    ap.add_argument('--config',type=Path)
    ap.add_argument('--msg-root',type=Path,action='append',default=[],help='Workspace src or installed message definitions; repeatable')
    ap.add_argument('--distro',choices=['foxy','humble','iron','jazzy'],default='foxy')
    ap.add_argument('--configs',nargs='+',default=None,help='Optional configuration filter, e.g. A D')
    ap.add_argument('--windows',type=Path,help='CSV with bag_path,start_s,end_s,outcome,participant_id,manual_intervention')
    ap.add_argument('--trust-human-schema',action='store_true',help='Enable human metrics only after reviewing field mapping and freshness semantics')
    args=ap.parse_args()
    cfg=DEFAULTS.copy()
    if args.config:
        supplied=yaml.safe_load(args.config.read_text()) or {}
        unknown=set(supplied)-set(cfg)
        if unknown:ap.error(f'Unknown config keys: {sorted(unknown)}')
        cfg.update(supplied)
    for key in ['speech_started_values', 'speech_finished_values']:
        cfg[key] = [str(x).lower() for x in cfg[key]]
    for key in ['max_gap_s', 'max_human_age_s', 'sync_tolerance_s', 'max_plausible_speed_mps']:
        if cfg[key] <= 0:ap.error(f'{key} must be positive')
    if cfg['social_distance_m'] is not None and cfg['social_distance_m'] <= 0:
        ap.error('social_distance_m must be positive or null')
    args.root=args.root.expanduser().resolve();args.output=args.output.expanduser().resolve()
    if args.output.exists():ap.error('Output directory already exists. Choose a new directory to preserve earlier analyses.')
    bags=discover(args.root)
    if args.configs:bags=[p for p in bags if str(value(trial_meta(p),'configuration.id','unknown')) in args.configs]
    if not bags:ap.error('No ROS 2 bag metadata.yaml or standalone MCAP found. Supply complete bag directories, not isolated DB3 files.')
    
    try:
        bags = select_latest_trials(bags)
    except ValueError as exc:
        ap.error(str(exc))

    windows={}
    if args.windows:
        with args.windows.open(newline='') as f:
            for r in csv.DictReader(f):
                windows[str(Path(r['bag_path']).expanduser().resolve())]=r
    args.output.mkdir(parents=True)
    store,warnings=type_store(args.distro,args.msg_root)
    dump(args.output/'analysis_config.json',{'settings':cfg,'arguments':vars(args),
        'type_warnings':warnings,'created_utc':datetime.now(timezone.utc).isoformat()})
    summaries=[];window_templates=[]
    for p in bags:
        meta=trial_meta(p);tid=str(meta.get('trial_id',p.parent.name))
        slug=re.sub('[^a-zA-Z0-9_-]+','_',tid)+'_'+hashlib.sha1(str(p).encode()).hexdigest()[:8]
        dest=args.output/'trials'/slug;dest.mkdir(parents=True)
        summary={'trial_id':tid,'output_id':slug,'bag_path':str(p),'trial_type':meta.get('trial_type','unknown'),
                 'configuration':str(value(meta,'configuration.id','unknown')),
                 'scenario':str(value(meta,'scenario.id','unknown')),'analysis_status':'failed'}
        print(f'[READ] {p}',flush=True)
        try:
            data=read_trial(p,store,cfg,dest)
            manual=windows.get(str(p),{})
            lo,hi,source,outcome=window_for(data,manual)
            if hi<=lo:raise ValueError('Empty analysis window')
            df,metrics,bouts=motion_metrics(data['odom'],lo,hi,cfg)
            dist,hm,hexcluded=human_distances(data,df,lo,hi,cfg,args.trust_human_schema)
            unique,examples=evidence_examples(data,cfg)
            metrics.update(hm)
            summary.update(metrics)
            summary.update({'analysis_status':'ok','window_start_s':lo,'window_end_s':hi,'window_duration_s':hi-lo,
                            'window_source':source,'outcome':outcome,'participant_id':manual.get('participant_id',''),
                            'manual_intervention':manual.get('manual_intervention',value(meta,'outcome.intervention_required','unknown')),
                            'robot_text_messages':sum(bool(r['text']) and r['role']=='robot_text' for r in unique) if any(i['topic'] in SPEECH_TEXT and i['decoded_count'] for i in data['inventory']) else None,
                            'decode_issue_categories':len(data['errors'])})
            df.to_csv(dest/'motion.csv',index=False);dist.to_csv(dest/'human_distances.csv',index=False)
            csv_out(dest/'stop_intervals.csv',[{'start_s':a,'end_s':b,'duration_s':b-a} for a,b in bouts],['start_s','end_s','duration_s'])
            csv_out(dest/'hri_messages.csv',[{k:v for k,v in r.items() if k!='data'} for r in unique],
                    ['t','topic','role','text','speech_id','status'])
            dump(dest/'hri_examples.json',examples);dump(dest/'action_statuses.json',data['statuses'])
            dump(dest/'human_exclusions.json',hexcluded);dump(dest/'trial_metadata_copy.json',meta)
            dump(dest/'metrics.json',summary)
            figs=figures(dest,df,dist,data['commands'],data['events'],examples,bouts,lo,hi,summary['configuration'],cfg)
            make_report(dest,summary,metrics,examples,data['errors'],figs)
            window_templates.append({'bag_path':str(p),'suggested_start_s':lo,'suggested_end_s':hi,
                                     'start_s':'','end_s':'','outcome':'','participant_id':'','manual_intervention':'',
                                     'suggestion_source':source})
            print(f'[OK] {tid}: {source}; {len(unique)} HRI records; {len(data["errors"])} decode issue categories',flush=True)
        except Exception as exc:
            summary['error']=str(exc)
            (dest/'report.md').write_text(f'# Analysis failed\n\n{exc}\n')
            print(f'[ERROR] {tid}: {exc}',file=sys.stderr)
        summaries.append(summary)
    if any(s['analysis_status']=='ok' for s in summaries):aggregate(args.output,summaries)
    else:csv_out(args.output/'trial_summary.csv',summaries)
    csv_out(args.output/'windows_to_review.csv',window_templates)
    print(f'[DONE] Results: {args.output}')
    return 0 if all(s['analysis_status']=='ok' for s in summaries) else 2

if __name__=='__main__':
    sys.exit(main())
