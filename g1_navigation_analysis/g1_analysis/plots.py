"""Publication exports. Every plot is saved as vector PDF/SVG and 300-dpi PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import textwrap

LABELS = {
    "mission_duration_s": "Observed mission duration (s)",
    "active_command_window_s": "Active-command window (s)",
    "terminal_position_residual_m": "Estimated terminal position residual (m)",
    "terminal_orientation_residual_deg": "Estimated terminal orientation residual (deg)",
    "initial_plan_deviation_rmse_m": "Initial-plan deviation RMSE (m)",
    "odom_amcl_consistency_rmse_m": "Odometry-AMCL position disagreement RMSE (m)",
    "compensation_domain_time_fraction": "Input-command time fraction in calibrated domain",
    "compensation_vy_delta_rmse": "Command compensation: lateral delta RMS (m/s)",
    "compensation_wz_delta_rmse": "Command compensation: yaw-rate delta RMS (rad/s)",
}

from .aggregate import METRICS


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    with matplotlib.rc_context({'pdf.fonttype':42,'svg.fonttype':'none'}):
        for extension in ('png','pdf','svg'):
            fig.savefig(Path(str(path) + '.' + extension),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def campaign_figures(results, summaries, config, output):
    configurations = list(config['configurations'])
    display_names = [textwrap.fill(config['configurations'][c],22) for c in configurations]
    colors = dict(zip(configurations, ['#777777','#0072B2','#E69F00','#009E73']))
    strata = sorted({(r['scenario'],r['speed_mps']) for r in results})
    if not strata:
        return
    for scenario,speed in strata:
        name = f'{scenario}_{speed:.2f}'
        rows = [r for r in summaries if r['scenario']==scenario and r['speed_mps']==speed]
        fig,ax = plt.subplots(figsize=(8,4))
        for i,c in enumerate(configurations):
            row = next((r for r in rows if r['configuration']==c),None)
            if row is None or row['success_rate_known'] is None:
                ax.text(i,.05,'No known\noutcomes',ha='center',fontsize=8)
                continue
            p = row['success_rate_known']
            ax.bar(i,p,color=colors[c],width=.6)
            ax.errorbar(i,p,yerr=[[max(0,p-row['success_ci95_low'])],[max(0,row['success_ci95_high']-p)]],fmt='none',color='black',capsize=4)
            ax.text(i,min(1.1,p+.10),f"{row['n_success']}/{row['n_trials']-row['n_unknown']} known\n{row['n_unknown']} unknown",ha='center',fontsize=8)
        ax.set_xticks(range(len(configurations)),display_names,fontsize=8)
        ax.set_xlim(-.6,len(configurations)-.4)
        ax.set_yticks(np.linspace(0,1,6))
        ax.set(ylim=(0,1.22),ylabel='Success fraction among known outcomes',title=f'{scenario}; maximum speed {speed:.2f} m/s')
        save(fig,output/f'success_{name}')
        for metric in METRICS:
            fig,ax = plt.subplots(figsize=(8,4))
            plotted = False
            for i,c in enumerate(configurations):
                rows = [r for r in results if r['scenario']==scenario and r['speed_mps']==speed and r['configuration']==c and r.get(metric) is not None]
                for j,r in enumerate(rows):
                    # Deterministic visual offsets are not experimental pairing.
                    offset = (j-(len(rows)-1)/2)*min(.05,.4/max(1,len(rows)))
                    ax.scatter(i+offset,r[metric],color=colors[c],marker='o' if r['outcome']=='SUCCEEDED' else 'x',s=35)
                    plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xticks(range(len(configurations)),display_names,fontsize=8)
            ax.set_xlim(-.6,len(configurations)-.4)
            ax.set(ylabel=LABELS.get(metric,metric),title=f'{scenario}; {speed:.2f} m/s | circle: success; cross: other/unknown')
            ax.grid(axis='y',alpha=.2)
            save(fig,output/f'{metric}_{name}')


def trial_figures(trial, streams, output):
    fig,axes = plt.subplots(3,1,figsize=(9,6),sharex=True)
    for topic in ('/cmd_vel','/cmd_vel_nav_scaled','/cmd_vel_rlsar'):
        rows = streams.get(topic,[])
        for ax,key in zip(axes,('vx','vy','wz')):
            if rows:
                ax.plot([r['t']-trial['analysis_start_s'] for r in rows],[r[key] for r in rows],label=topic,linewidth=.8)
            ax.set_ylabel(key+(' (rad/s)' if key=='wz' else ' (m/s)'))
            ax.grid(alpha=.2)
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    axes[-1].set_xlabel('Time since analysis start (s)')
    save(fig,output/'commands')
    rows = streams.get('/amcl_pose',[])
    if rows and len({r['frame'] for r in rows})==1:
        fig,ax = plt.subplots(figsize=(6,6))
        ax.plot([r['x'] for r in rows],[r['y'] for r in rows],label='AMCL estimate')
        plans = [p for p in streams.get('/plan',[]) if p['points'] and p['frame']==rows[0]['frame']]
        if plans:
            xy = np.asarray(plans[0]['points'])
            ax.plot(xy[:,0],xy[:,1],'--',label='First recorded plan in window')
        ax.set(xlabel='x (m)',ylabel='y (m)',title=f"Estimated trajectory; frame: {rows[0]['frame']}")
        ax.set_aspect('equal',adjustable='datalim'); ax.legend(); ax.grid(alpha=.2)
        save(fig,output/'estimated_trajectory')
