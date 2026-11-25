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
            fig.savefig(Path(str(path) + '.' + extension),dpi=300,bbox_inches='tight',facecolor='white')
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

def campaign_figures(
    results,
    summaries,
    config,
    output,
    success_counts_path=None,
):
    from .numerics import wilson

    output = Path(output)

    configurations = list(config["configurations"])

    display_names = [
        textwrap.fill(config["configurations"][c], 22)
        for c in configurations
    ]

    colors = dict(zip(
        configurations,
        ["#777777", "#0072B2", "#E69F00", "#009E73"],
    ))

    declared = load_success_counts(success_counts_path)

    strata = sorted({
        (r["scenario"], r["speed_mps"])
        for r in results
    })

    if not strata:
        return

    for scenario, speed in strata:
        name = f"{scenario}_{speed:.2f}"

        summary_rows = [
            r for r in summaries
            if r["scenario"] == scenario
            and r["speed_mps"] == speed
        ]

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, c in enumerate(configurations):
            row = next(
                (
                    r for r in summary_rows
                    if r["configuration"] == c
                ),
                None,
            )

            key = (
                scenario,
                round(float(speed), 2),
                c,
            )

            aggregate = False

            if row is not None and row["success_rate_known"] is not None:
                p = row["success_rate_known"]
                successes = row["n_success"]
                trials = row["n_trials"] - row["n_unknown"]

                ci_low = row["success_ci95_low"]
                ci_high = row["success_ci95_high"]

                label = (
                    f"{successes}/{trials} known\n"
                    f"{row['n_unknown']} unknown"
                )

            elif key in declared:
                successes, trials = declared[key]

                individual_trials = [
                    r for r in results
                    if r["scenario"] == scenario
                    and abs(float(r["speed_mps"]) - speed) < 1e-6
                    and r["configuration"] == c
                ]

                if individual_trials and len(individual_trials) != trials:
                    raise ValueError(
                        f"Trial count mismatch for {key}: "
                        f"{len(individual_trials)} recorded, "
                        f"{trials} declared."
                    )

                p = successes / trials

                ci_low, ci_high = wilson(
                    successes,
                    trials,
                )

                aggregate = True
                label = f"{successes}/{trials} declared*"

            else:
                ax.text(
                    i,
                    0.05,
                    "No known\noutcomes",
                    ha="center",
                    fontsize=8,
                )
                continue

            ax.bar(
                i,
                p,
                color=colors[c],
                width=0.6,
                hatch="//" if aggregate else None,
            )

            ax.errorbar(
                i,
                p,
                yerr=[
                    [max(0, p - ci_low)],
                    [max(0, ci_high - p)],
                ],
                fmt="none",
                color="black",
                capsize=4,
            )

            ax.text(
                i,
                min(1.1, p + 0.10),
                label,
                ha="center",
                fontsize=8,
            )

        ax.set_xticks(
            range(len(configurations)),
            display_names,
            fontsize=8,
        )
        ax.set_xlim(-0.6, len(configurations) - 0.4)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.set(
            ylim=(0, 1.22),
            ylabel="Success fraction",
            title=f"{scenario}; maximum speed {speed:.2f} m/s",
        )

        if declared:
            ax.text(
                0.5,
                -0.20,
                "* Aggregate-reported results; individual "
                "outcomes not assigned.",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )

        save(fig, output / f"success_{name}")

        # Preserve the original continuous-metric plots.
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8, 4))
            plotted = False

            for i, c in enumerate(configurations):
                metric_rows = [
                    r for r in results
                    if r["scenario"] == scenario
                    and r["speed_mps"] == speed
                    and r["configuration"] == c
                    and r.get(metric) is not None
                ]

                for j, r in enumerate(metric_rows):
                    offset = (
                        j - (len(metric_rows) - 1) / 2
                    ) * min(
                        0.05,
                        0.4 / max(1, len(metric_rows)),
                    )

                    ax.scatter(
                        i + offset,
                        r[metric],
                        color=colors[c],
                        marker=(
                            "o"
                            if r["outcome"] == "SUCCEEDED"
                            else "x"
                        ),
                        s=35,
                    )

                    plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xticks(
                range(len(configurations)),
                display_names,
                fontsize=8,
            )
            ax.set_xlim(-0.6, len(configurations) - 0.4)

            ax.set(
                ylabel=LABELS.get(metric, metric),
                title=(
                    f"{scenario}; {speed:.2f} m/s | "
                    "circle: success; cross: other/unknown"
                ),
            )

            ax.grid(axis="y", alpha=0.2)

            save(
                fig,
                output / f"{metric}_{name}",
            )

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
