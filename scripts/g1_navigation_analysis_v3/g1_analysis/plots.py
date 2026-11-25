"""Publication exports. Every plot is saved as vector PDF/SVG and 300-dpi PNG."""
from pathlib import Path
import matplotlib
import csv
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
            fig.savefig(Path(str(path) + '.' + extension),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def load_success_counts(path):
    """Read declared aggregate successes without altering trial outcomes."""
    if path is None:
        return {}

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    counts = {}

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["scenario"].strip(),
                round(float(row["speed_mps"]), 2),
                row["configuration"].strip(),
            )

            successes = int(row["successes"])
            trials = int(row["trials"])

            if key in counts:
                raise ValueError(f"Duplicate success count: {key}")

            if trials <= 0 or not 0 <= successes <= trials:
                raise ValueError(f"Invalid success count: {key}")

            counts[key] = successes, trials

    return counts

def campaign_figures(results, summaries, config, output, success_counts_path=None):
    """Compatibility entry point: use the shared publication renderer."""
    from .paper_plots import campaign_figures as render
    return render(results, summaries, config, output, success_counts_path)


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
