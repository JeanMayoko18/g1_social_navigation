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
from collections import defaultdict
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
        return {key: primitive(getattr(obj, key), depth + 1)
                for key in obj.get_fields_and_field_types()}
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


def read_bag(bag_dir):
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
    usable = INTEREST & types.keys()
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


def detect_events(rows, t0, t1):
    events = []
    def add(t, typ, topic, detail=''):
        if t0 <= t <= t1:
            events.append({'time_s': round(t - t0, 3), 'event': typ,
                           'topic': topic, 'detail': detail[:260]})

    decisions = ['/social_navigation/decision', '/social_navigation/collaborative_decision']
    for topic in decisions:
        for t, data in rows.get(topic, []):
            text = flat_text(data).upper()
            # Require whole-word command, not substrings like WAITING, REPLANNING.
            matches = [name for name in ('WAIT', 'REPLAN', 'RESUME')
                       if re.search(r'(?<![A-Z0-9_])' + name + r'(?![A-Z0-9_])', text)]
            for name in matches:
                add(t, name, topic, flat_text(data))

    for t, data in rows.get('/explainability/text', []):
        # Text may be a std_msgs/String, or a richer custom message.
        content = get_field(data, 'data', 'text', 'utterance', 'message', 'explanation')
        if content is None:
            content = data
        speech = flat_text(content).strip()
        if speech and speech.lower() not in ('none', 'null', '{}', '[]'):
            add(t, 'EXPLAIN', '/explainability/text', speech)

    for t, data in rows.get('/explainability/speech_status', []):
        st = flat_text(data).upper()
        for name in ('STARTED', 'COMPLETED', 'FAILED'):
            if re.search(r'(?<![A-Z0-9_])' + name + r'(?![A-Z0-9_])', st):
                add(t, 'SPEECH_' + name, '/explainability/speech_status', flat_text(data))
                break

    for topic in ('/navigate_to_pose_internal/_action/status', '/navigate_to_pose/_action/status'):
        prev = set()
        for t, data in rows.get(topic, []):
            current = set(terminal_status(data))
            for status, goal_id in (current - prev):
                name = {4:'GOAL_SUCCEEDED', 5:'GOAL_CANCELED', 6:'GOAL_ABORTED'}[status]
                add(t, name, topic, goal_id)
            prev = current

    # Collapse duplicate command broadcasts across closely spaced decision topics.
    events.sort(key=lambda e: (e['time_s'], e['event']))
    result = []
    for e in events:
        if e['event'] in ('WAIT', 'REPLAN', 'RESUME') and any(
            r['event'] == e['event'] and abs(r['time_s'] - e['time_s']) <= 0.25
            for r in result[-10:]):
            continue
        result.append(e)
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


def choose_window(rows):
    all_ts = [t for topic in rows.values() for t, _ in topic]
    if not all_ts:
        raise ValueError('No relevant messages found in bag.')
    goal_ts = [t for topic in ('/goal_pose', '/navigate_to_pose_internal/_action/goal')
               for t, _ in rows.get(topic, [])]
    t0 = min(goal_ts) if goal_ts else min(all_ts)
    # Prefer the LAST observed success event, to avoid ending at internal cancellation.
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


def make_figure(output, duration, path, stationary, events, segments, note):
    ink = '#183746'; teal = '#246C80'; orange = '#CE893D'; blue = '#5279BA'; green = '#23846B'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig = plt.figure(figsize=(14, 7.5), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.25], hspace=0.54,
                          left=0.08, right=0.97, top=0.84, bottom=0.11)
    fig.text(0.08, 0.955, 'Configuration D | Social Encounter Case Study', color=ink,
             fontsize=20, fontweight='bold', va='top')
    fig.text(0.08, 0.90, 'A. Social encounter timeline   •   B. Robot response',
             color=teal, fontsize=13, fontweight='bold')

    ax = fig.add_subplot(gs[0]); ax.set_facecolor('#F5F8F9')
    ymap = {'WAIT':3, 'EXPLAIN':2, 'REPLAN':1, 'RESUME':0}
    colors = {'WAIT':orange, 'EXPLAIN':blue, 'REPLAN':teal, 'RESUME':green}
    shown = {}
    for e in events:
        et = e['event']
        if et not in ymap:
            continue
        t = e['time_s']; y = ymap[et]
        ax.vlines(t, y-0.30, y+0.30, color=colors[et], linewidth=2, alpha=0.8)
        ax.scatter([t], [y], s=60, color=colors[et], zorder=4)
        # Label only the earliest event of each type to keep plot readable.
        if et not in shown:
            shown[et] = True
            ax.annotate(f'{t:.2f} s', (t,y), xytext=(5,11), textcoords='offset points',
                        fontsize=9, color=ink)
    for start, stop in segments:
        ax.axvspan(start, stop, color=orange, alpha=0.08)
    ax.set_yticks([3,2,1,0], ['WAIT', 'EXPLAIN', 'REPLAN', 'RESUME'])
    ax.set_xlim(0, max(1, duration * 1.02)); ax.set_ylim(-0.65,3.7)
    ax.grid(axis='x', color='#D9E2E5'); ax.set_xlabel('Seconds from start of analysis window')
    ax.spines[['top','right','left']].set_visible(False)
    ax.text(0.98, 1.05, 'Shading = detected stationary periods', transform=ax.transAxes,
            ha='right', color='#5E7079', fontsize=9)

    ax2=fig.add_subplot(gs[1]); ax2.axis('off')
    seq=[('STOP','WAIT command',orange),('EXPLAIN','Text publication',blue),('RESUME','RESUME command',green)]
    for idx,(head,sub,c) in enumerate(seq):
        x=0.03+idx*0.337
        patch=FancyBboxPatch((x,0.29),0.265,0.52,boxstyle='round,pad=0.016,rounding_size=0.04',
                             transform=ax2.transAxes,facecolor='#F5F8F9',edgecolor=c,linewidth=2)
        ax2.add_patch(patch)
        ax2.text(x+0.1325,0.59,head,ha='center',va='center',fontsize=16,fontweight='bold',
                 color=c,transform=ax2.transAxes)
        ax2.text(x+0.1325,0.40,sub,ha='center',va='center',fontsize=10,color=ink,
                 transform=ax2.transAxes)
        if idx<2:
            ax2.annotate('',xy=(x+0.33,0.55),xytext=(x+0.275,0.55),xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='->',linewidth=2,color=teal))

    pval = f'{path:.2f} m' if path is not None else 'N/A'
    sval = f'{stationary:.2f} s' if stationary is not None else 'N/A'
    fig.text(0.08,0.025, f'Total duration: {duration:.2f} s     |     Stationary time: {sval}     |     Path length: {pval}',
             fontsize=13,color=ink,fontweight='bold')
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(output / f'hri_case_study_D_S1a.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bag', type=Path, help='ROS 2 bag directory containing metadata.yaml')
    ap.add_argument('--out', type=Path, help='Output directory (default: sibling poster_outputs)')
    ap.add_argument('--t0', type=float, help='Override analysis start in seconds since UNIX epoch')
    ap.add_argument('--t1', type=float, help='Override analysis end in seconds since UNIX epoch')
    args=ap.parse_args()
    bag=args.bag.expanduser().resolve()
    output=args.out.expanduser().resolve() if args.out else bag.parent / 'poster_outputs'
    rows, types=read_bag(bag)
    t0,t1,note=choose_window(rows)
    if args.t0 is not None:
        t0=args.t0
    if args.t1 is not None:
        t1=args.t1
    if t1 <= t0:
        raise ValueError('Analysis end must be later than analysis start.')
    events=detect_events(rows,t0,t1)
    path,stationary,odom_topic,n,segments=motion_metrics(rows,t0,t1)
    duration=t1-t0
    make_figure(output,duration,path,stationary,events,segments,note)
    with (output / 'events.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['time_s','event','topic','detail'])
        writer.writeheader(); writer.writerows(events)
    summary={'bag_path':str(bag),'analysis_start_unix_s':t0,'analysis_end_unix_s':t1,
             'analysis_window_note':note,'duration_s':round(duration,3),
             'stationary_s':None if stationary is None else round(stationary,3),
             'path_length_m':None if path is None else round(path,3),
             'odom_topic':odom_topic,'odom_samples':n,
             'available_interesting_topics':sorted(set(types) & INTEREST),
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
    print('Review events.csv and summary.json before citing results in your poster.')

if __name__=='__main__':
    main()
