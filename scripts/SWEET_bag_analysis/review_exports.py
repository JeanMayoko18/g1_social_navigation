#!/usr/bin/env python3
"""Audit exported review files; does not reconstruct absent motion or human data."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_social_bags import window_for, NAV_STATUS

def write(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('review_directory',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    trials=list(csv.DictReader((a.review_directory/'trial_summary.csv').open()))
    windows=[]; speech=[]; quotes=[]; goals=[]
    for r in trials:
        p=a.review_directory/'trials'/r['output_id'];statuses=json.loads((p/'action_statuses.json').read_text())
        # Infer recording duration only for the recording-only exports; single-goal
        # windows need no recording end to infer the same completed goal again.
        end=max(float(r['window_end_s']),max((s['t'] for s in statuses),default=0))
        lo,hi,src,outcome=window_for({'end':end,'statuses':statuses},{})
        public=[s for s in statuses if s['topic']==NAV_STATUS]
        ids={s['goal_id'] for s in public if s['status'] in ['accepted','executing','canceling']}
        windows.append(dict(output_id=r['output_id'],bag_path=r['bag_path'],active_public_goals=len(ids),start_s=lo if src!='recording_only' else '',end_s=hi if src!='recording_only' else '',window_source=src,nav2_outcome=outcome,review_reason='' if src!='recording_only' else ('multiple_active_goals' if len(ids)>1 else 'no_success_or_abort_terminal')))
        for gid in sorted(ids):
            seq=[s for s in public if s['goal_id']==gid]
            goals.append(dict(output_id=r['output_id'],goal_id=gid,transitions='; '.join(f"{s['t']:.3f}:{s['status']}" for s in seq)))
        hri=list(csv.DictReader((p/'hri_messages.csv').open()))
        if r['configuration']=='D':
            counts=Counter(x['status'].upper() for x in hri if x['role']=='speech_status')
            speech.append(dict(output_id=r['output_id'],trial=r['trial_id'],STARTED=counts['STARTED'],COMPLETED=counts['COMPLETED'],FAILED=counts['FAILED']))
        for x in hri:
            if x['role']=='robot_text' and x['text']:
                quotes.append(dict(output_id=r['output_id'],t_s=x['t'],topic=x['topic'],text_verbatim=x['text'],evidence='recorded_text_not_proof_of_audible_delivery'))
    write(a.output/'window_audit.csv',windows);write(a.output/'active_public_goals.csv',goals);write(a.output/'speech_status_counts.csv',speech);write(a.output/'robot_texts_verbatim.csv',quotes)
    fig,ax=plt.subplots(figsize=(9,4.5));ys=list(range(len(speech)));left=[0]*len(speech)
    for key,color in [('STARTED','#429a91'),('COMPLETED','#4673ab'),('FAILED','#c45047')]:
        vals=[r[key] for r in speech];ax.barh(ys,vals,left=left,label=key,color=color);left=[x+y for x,y in zip(left,vals)]
    ax.set_yticks(ys,[r['trial'].replace('participant_','') for r in speech]);ax.invert_yaxis();ax.set_xlabel('Recorded status messages (not independent utterances)');ax.legend(ncol=3,loc='lower right');fig.tight_layout()
    fig.savefig(a.output/'speech_status_audit.png',dpi=200);fig.savefig(a.output/'speech_status_audit.pdf');plt.close(fig)
    print(f'Reviewed {len(windows)} recordings; {sum(x["window_source"]!="recording_only" for x in windows)} single-goal windows inferred.')
if __name__=='__main__':main()
