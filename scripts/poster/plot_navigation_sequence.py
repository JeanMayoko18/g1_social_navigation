#!/usr/bin/env python3
"""Plot a verified navigation/social-hold/dialogue/resumption/success sequence.

Inputs: exported JSONL (no ROS required) or ROS 2 bag directory (ROS sourced).
Uses the same goal, matched HRI request/person, and accepted START request.
Does not relabel every START as RESUME or infer physical motion from states.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
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
NAV='/social_navigation/navigation_state'
DEC='/social_navigation/collaborative_decision'
GATE='/social_navigation/gate_command_result'
REQUEST='/hri/dialogue_request'
INTENT='/hri/human_intent'
COLORS=['#176F79','#C68129','#416FA7','#23846B','#1A536B']
INK='#153448'


def decoded(message):
    if isinstance(message, dict) and isinstance(message.get('data'), str):
        try:
            inner=json.loads(message['data'])
            if isinstance(inner,dict):return inner
        except ValueError:pass
    return message if isinstance(message,dict) else {}


def read_records(source):
    records=[]
    if source.is_dir():
        rows,types=read_bag(source)
        all_times=[t for messages in rows.values() for t,_ in messages]
        if not all_times:raise ValueError('No relevant messages decoded.')
        origin=min(all_times)
        for topic,messages in rows.items():
            for t,msg in messages:
                records.append({'t':t-origin,'topic':topic,'msg':decoded(msg),'source_line':None})
        description='Seconds from first decoded relevant bag message'
    else:
        raw=[]
        with source.open(encoding='utf-8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip():continue
                obj=json.loads(line)
                obj['source_line']=line_no
                raw.append(obj)
        if not raw:raise ValueError('Empty JSONL input.')
        has_t=all('t' in r for r in raw)
        if not has_t and not all('timestamp_ns' in r for r in raw):
            raise ValueError('Every row needs t or timestamp_ns; mixed clocks are not supported.')
        origin=0 if has_t else min(r['timestamp_ns'] for r in raw)/1e9
        for r in raw:
            t=float(r['t']) if has_t else r['timestamp_ns']/1e9-origin
            if not math.isfinite(t):raise ValueError('Non-finite timestamp.')
            records.append({'t':t,'topic':r['topic'],'msg':decoded(r['msg']),
                            'source_line':r['source_line']})
        description='JSONL t values' if has_t else 'Seconds from first JSONL timestamp'
    return sorted(records,key=lambda r:r['t']),description


def goal(r):return r['msg'].get('navigation_goal_id','')
def state(r):return r['msg'].get('navigation_state','')
def active(r):return state(r)=='NAVIGATING' and r['msg'].get('navigation_active') is True

def first(rows,predicate):return next((r for r in rows if predicate(r)),None)


def find_sequences(records):
    """Return complete, correlated sequences plus rejected-candidate diagnostics.

    Stop = first SOCIAL_HOLD/WAITING state after NAVIGATING for the same goal,
    with a social WAIT decision and an accepted gate transaction. Resume is
    a later NAVIGATING state following a matched human confirmation and an
    accepted START transaction. Reset/change of goal terminates a candidate.
    """
    nav=[r for r in records if r['topic']==NAV]
    results=[];rejected=[];previous=None;run_start=None
    for stop in nav:
        if active(stop):
            if previous is None or not active(previous) or goal(previous)!=goal(stop):run_start=stop
        if (run_start and previous and active(previous) and goal(stop)==goal(previous)
            and state(stop) in ('SOCIAL_HOLD','WAITING') and not stop['msg'].get('navigation_active')):
            g=goal(stop)
            if not g:previous=stop;continue
            reset=first(nav,lambda r:r['t']>stop['t'] and goal(r)!=g)
            cutoff=reset['t'] if reset else float('inf')
            after=[r for r in records if stop['t']<=r['t']<cutoff]
            resume=first(after,lambda r:r['topic']==NAV and goal(r)==g and active(r))
            def reject(reason):rejected.append({'goal_id':g,'stop_t':stop['t'],'reason':reason})
            waits=[r for r in records if r['topic']==DEC and goal(r)==g
                   and run_start['t']<=r['t']<=stop['t'] and r['msg'].get('action')=='WAIT'
                   and ('SOCIAL' in r['msg'].get('reason','') or 'INTRUSION' in r['msg'].get('reason',''))]
            wait=waits[-1] if waits else None
            if not wait or not resume:
                reject('Missing social WAIT decision or later active navigation');previous=stop;continue
            wait_id=wait['msg'].get('request_id')
            wait_ack=first(records,lambda r:r['topic']==GATE and goal(r)==g
                and wait['t']<=r['t']<=resume['t'] and wait_id
                and r['msg'].get('request_id')==wait_id and r['msg'].get('action')=='WAIT'
                and r['msg'].get('status') in ('ACCEPTED','HOLD_CONFIRMED'))
            if not wait_ack:
                reject('Missing accepted WAIT transaction');previous=stop;continue
            person=wait['msg'].get('person_id','')
            requests=[r for r in after if r['topic']==REQUEST and r['t']<resume['t']
                      and r['msg'].get('request_type')=='PASSAGE_CLARIFICATION'
                      and r['msg'].get('context')=='SOCIAL_CONFLICT'
                      and (not person or r['msg'].get('person_id')==person)]
            matched=None
            for request in requests:
                request_id=request['msg'].get('request_id')
                interaction=request['msg'].get('interaction_id')
                confirm=first(after,lambda r:r['topic']==INTENT and request['t']<=r['t']<resume['t']
                    and request_id and r['msg'].get('request_id')==request_id
                    and (not interaction or r['msg'].get('interaction_id')==interaction)
                    and (not person or r['msg'].get('person_id')==person)
                    and r['msg'].get('intent') in ('CONFIRM','ALLOW_PASSAGE','ALLOW_CONFIRMED'))
                if not confirm:continue
                starts=[r for r in after if r['topic']==DEC and goal(r)==g
                        and confirm['t']<=r['t']<=resume['t'] and r['msg'].get('action')=='START'
                        and r['msg'].get('navigation_authorized') is True
                        and (not person or r['msg'].get('person_id')==person)]
                for start in starts:
                    start_id=start['msg'].get('request_id')
                    accepted=first(after,lambda r:r['topic']==GATE and goal(r)==g and start_id
                        and start['t']<=r['t']<=resume['t'] and r['msg'].get('request_id')==start_id
                        and r['msg'].get('action')=='START' and r['msg'].get('status')=='ACCEPTED')
                    if accepted:matched=(request,confirm,start,accepted);break
                if matched:break
            if not matched:
                reject('No correlated passage confirmation / accepted START');previous=stop;continue
            # Do not jump over another stop or failure to claim this resumption completed the goal.
            boundary=first(after,lambda r:r['topic']==NAV and r['t']>resume['t']
                           and state(r) in ('SOCIAL_HOLD','WAITING','ABORTED','CANCELED'))
            end=boundary['t'] if boundary else cutoff
            success=first(after,lambda r:r['topic']==NAV and goal(r)==g
                          and resume['t']<=r['t']<end and state(r)=='SUCCEEDED')
            if not success:
                reject('Resumption found, but no success before the next hold/failure/reset');previous=stop;continue
            request,confirm,start,accepted=matched
            hold=first(after,lambda r:r['topic']==GATE and goal(r)==g and r['msg'].get('request_id')==wait_id
                       and r['msg'].get('status')=='HOLD_CONFIRMED' and r['t']<=request['t'])
            explanation=first(after,lambda r:r['topic']=='/explainability/text' and goal(r)==g
                              and stop['t']<=r['t']<=request['t'] and r['msg'].get('text'))
            results.append({'goal_id':g,'navigation':run_start,'wait_decision':wait,'wait_accepted':wait_ack,
                'stop':stop,'hold_confirmed':hold,'explanation':explanation,'negotiation':request,
                'confirmation':confirm,'start_decision':start,'start_accepted':accepted,
                'resume':resume,'success':success})
        previous=stop
    return results,rejected


def save(fig,out,name,dpi):
    from matplotlib.text import Text
    for txt in fig.findobj(Text):txt.set_fontweight('bold')
    fig.canvas.draw()
    for ext in ('svg','pdf','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=dpi,facecolor='white')
    plt.close(fig)


def figures(seq,out,dpi=600):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.weight':'bold','axes.labelweight':'bold',
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'svg.fonttype':'none','pdf.fonttype':42})
    base=seq['navigation']['t']
    t={k:seq[k]['t']-base for k in ('navigation','stop','negotiation','confirmation','resume','success')}
    hold=t['resume']-t['stop'];latency=t['resume']-t['confirmation']
    mode='mode C / Nav2' if 'MODE_C' in seq['start_decision']['msg'].get('reason','') else 'navigation controller'
    fig=plt.figure(figsize=(16,5.5),facecolor='white')
    fig.text(.025,.935,'Navigation resumes after human confirmation',fontsize=25)
    fig.text(.025,.855,'Recorded sequence for the same navigation goal',fontsize=17,color='#536974')
    ax=fig.add_axes([.02,.32,.96,.43]);ax.axis('off')
    stages=[('NAVIGATION','navigation','Navigation active'),('SOCIAL STOP','stop','Social hold'),
            ('NEGOTIATION','negotiation','“May I pass?”'),('RESUMPTION','resume','Navigation active'),
            ('SUCCESS','success','SUCCEEDED')]
    for i,(label,key,detail) in enumerate(stages):
        x=.012+i*.198;w=.178
        ax.add_patch(FancyBboxPatch((x,.08),w,.80,boxstyle='round,pad=.006,rounding_size=.035',
                     transform=ax.transAxes,facecolor='#F4F7F8',edgecolor=COLORS[i],linewidth=2))
        ax.text(x+w/2,.70,label,transform=ax.transAxes,ha='center',fontsize=16,color=COLORS[i])
        ax.text(x+w/2,.44,f"{t[key]:.2f} s",transform=ax.transAxes,ha='center',fontsize=24)
        ax.text(x+w/2,.22,detail,transform=ax.transAxes,ha='center',fontsize=13)
        if i<4:ax.annotate('',xy=(x+.198-.006,.47),xytext=(x+w+.004,.47),xycoords='axes fraction',
                          arrowprops={'arrowstyle':'-|>','color':INK,'lw':2})
    fig.text(.03,.245,f'“{seq["confirmation"]["msg"].get("utterance","Confirmation") }” recognized at {t["confirmation"]:.2f} s; navigation active {latency:.3f} s later.',fontsize=18,color=COLORS[3])
    fig.text(.03,.155,f'Social hold: {hold:.2f} s     •     Handoff: {mode}     •     Same goal retained',fontsize=16)
    fig.text(.03,.065,f'Time zero = first NAVIGATING at log t = {base:.3f} s. States confirm software execution, not measured motion.',fontsize=12)
    save(fig,out,'01_navigation_stop_negotiation_resume_success',dpi)

    fig=plt.figure(figsize=(15,8.2),facecolor='white')
    fig.text(.025,.94,'Social encounter: timing and navigation outcome',fontsize=24)
    fig.text(.025,.885,'Navigation status and correlated dialogue events',fontsize=16,color='#536974')
    ax=fig.add_axes([.20,.24,.76,.57]);ax.set_facecolor('#F8FAFB')
    intervals=[(4,t['navigation'],t['stop'],COLORS[0],'Navigation active'),
               (3,t['stop'],t['resume'],COLORS[1],'Social hold'),
               (2,t['negotiation'],t['confirmation'],COLORS[2],'“May I pass?” → “Yes”'),
               (1,t['resume'],t['success'],COLORS[3],'Navigation resumed')]
    for y,a,b,c,label in intervals:
        ax.barh(y,b-a,left=a,height=.48,color=c,zorder=3)
        ax.text((a+b)/2,y+.35,f'{b-a:.2f} s',ha='center',fontsize=16,color=c)
    ax.scatter([t['success']],[0],s=170,marker='D',color=COLORS[4],zorder=4)
    ax.text(t['success']-.7,-.4,f"{t['success']:.2f} s",ha='right',fontsize=15,color=COLORS[4])
    ax.annotate(f'YES recognized\n{t["confirmation"]:.2f} s',xy=(t['confirmation'],2),
        xytext=(t['confirmation']+6,2.4),fontsize=14,color=COLORS[2],
        arrowprops={'arrowstyle':'->','color':COLORS[2]})
    ax.text(t['stop'],3-.46,f"{t['stop']:.2f} s",fontsize=13,ha='left')
    ax.text(t['resume'],1-.46,f"{t['resume']:.2f} s",fontsize=13,ha='left')
    ax.set_yticks([4,3,2,1,0],['NAVIGATION','SOCIAL STOP','NEGOTIATION','RESUMPTION','SUCCESS'],fontsize=15)
    ax.set_ylim(-.7,4.8);ax.set_xlim(0,t['success']+3)
    ax.tick_params(axis='both',length=0,labelsize=14,pad=9)
    ax.grid(axis='x',color='#DCE5E8',zorder=0)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#B7C5CC')
    ax.set_xlabel('Seconds since the initial NAVIGATING state',fontsize=16,labelpad=12)
    fig.text(.03,.125,f'Confirmation → navigation active: {latency:.3f} s. Resumption uses {mode}.',fontsize=16)
    fig.text(.03,.055,'Social hold includes dialogue. SUCCESS is the recorded navigation status. Physical motion is not measured in these figures.',fontsize=11.5)
    save(fig,out,'02_navigation_timeline',dpi)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path,help='JSONL file or ROS 2 bag directory')
    ap.add_argument('--out',type=Path,default=Path('navigation_figures'))
    ap.add_argument('--goal-id',help='Select a specific navigation_goal_id')
    ap.add_argument('--sequence',type=int,default=-1,help='Complete sequence index, default last (-1)')
    ap.add_argument('--dpi',type=int,default=600)
    args=ap.parse_args()
    if not 72<=args.dpi<=1200:ap.error('dpi must be between 72 and 1200')
    records,clock=read_records(args.input.expanduser().resolve())
    complete,rejected=find_sequences(records)
    if args.goal_id:complete=[s for s in complete if s['goal_id']==args.goal_id]
    out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    inventory={'input':str(args.input),'time_reference':clock,'complete_sequences':[
        {'goal_id':s['goal_id'],'navigation_t':s['navigation']['t'],'stop_t':s['stop']['t'],
         'resume_t':s['resume']['t'],'success_t':s['success']['t']} for s in complete],
        'rejected_candidates':rejected}
    (out/'sequence_inventory.json').write_text(json.dumps(inventory,indent=2))
    if not complete:raise SystemExit('No complete correlated sequence found. No success/resumption figure invented; see sequence_inventory.json.')
    try:seq=complete[args.sequence]
    except IndexError:raise SystemExit('Sequence index out of range; see sequence_inventory.json.')
    base=seq['navigation']['t'];events=[]
    for name,value in seq.items():
        if name=='goal_id' or value is None:continue
        events.append({'stage':name,'log_time_s':value['t'],'relative_time_s':value['t']-base,
                       'source_line':value['source_line'],'topic':value['topic'],
                       'message_json':json.dumps(value['msg'],ensure_ascii=False)})
    events.sort(key=lambda x:x['log_time_s'])
    with (out/'sequence_evidence.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
    (out/'selected_sequence.json').write_text(json.dumps(seq,indent=2,ensure_ascii=False))
    figures(seq,out,args.dpi)
    print(json.dumps({'goal_id':seq['goal_id'],'social_hold_s':seq['resume']['t']-seq['stop']['t'],
                      'confirmation_to_navigation_s':seq['resume']['t']-seq['confirmation']['t'],
                      'out':str(out)},indent=2))


if __name__=='__main__':main()
