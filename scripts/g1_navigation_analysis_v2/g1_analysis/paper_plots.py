"""Generate publication-quality navigation campaign figures.

Responsibilities
----------------
- Preserve the existing metric definitions and output filenames.
- Separate all-outcome and successful-only populations.
- Display individual trials, medians, and valid sample counts.
- Export every generated figure as PDF, SVG, and PNG.
- Record unavailable metrics in FIGURE_MANIFEST.md.

This module consumes previously computed trial metrics. It does not
recompute navigation metrics or infer missing mission outcomes.

Integration
-----------
Keep this module inside the existing g1_analysis package.
Its public entry point remains:

    campaign_figures(results, summaries, config, output)
"""

from pathlib import Path
from textwrap import fill
import csv

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .numerics import wilson


# ============================================================================
# CONFIGURATION
# ============================================================================

LABELS = [
    "Combined",
    "Odom-\nonly",
    "Dog",
    "g1_nav",
]

ORDER = [
    "calibrated",
    "Odometry-only",
    "dog",
    "g1_nav",
]

COLOR_BY_CONFIGURATION = {
    "calibrated": "#0072B2",
    "Odometry-only": "#8172B3",
    "dog": "#E69F00",
    "g1_nav": "#009E73",
}

COLORS = [
    COLOR_BY_CONFIGURATION[configuration]
    for configuration in ORDER
]

PANELS = [
    (
        "corridor_with_doors",
        0.12,
        "Fire doors | 0.12 m/s",
        "fire_doors_v12",
    ),
    (
        "corridor_with_doors",
        0.30,
        "Fire doors | 0.30 m/s",
        "fire_doors_v30",
    ),
    (
        "corridor",
        0.12,
        "Office corridor | 0.12 m/s",
        "office_corridor_v12",
    ),
    (
        "corridor",
        0.30,
        "Office corridor | 0.30 m/s",
        "office_corridor_v30",
    ),
]

# Entries contain:
# (CSV metric key, output filename stem, figure title, vertical axis label).
METRICS = [
    (
        "command_window_s",
        "command_windows",
        "Recorded command publication windows",
        "Duration (s)",
    ),
    (
        "odom_distance_5hz_m",
        "odom_distance_5hz_m",
        "Distance from resampled navigation odometry",
        "Distance (m)",
    ),
    (
        "odom_amcl_consistency_rmse_m",
        "active_used_odom_amcl_rmse_m",
        "Active-window odometry–AMCL disagreement",
        "RMSE (m)",
    ),
    (
        "terminal_position_residual_m",
        "action_terminal_position_residual_m",
        "Action-terminal estimated position residual",
        "Residual (m)",
    ),
    (
        "legacy_raw_odom_amcl_rmse_m",
        "raw_odom_amcl_rmse_m",
        "Aligned raw Dog odometry–AMCL disagreement",
        "RMSE (m)",
    ),
    (
        "legacy_raw_odom_amcl_final_m",
        "raw_odom_amcl_final_m",
        "Final raw Dog odometry–AMCL separation",
        "Separation (m)",
    ),
    (
        "legacy_used_odom_amcl_rmse_m",
        "used_odom_amcl_rmse_m",
        "Aligned navigation odometry–AMCL disagreement",
        "RMSE (m)",
    ),
    (
        "legacy_initial_plan_deviation_rmse_m",
        "initial_plan_deviation_rmse_m",
        "Deviation from the first recorded plan",
        "RMSE (m)",
    ),
    (
        "legacy_commanded_stop_count",
        "commanded_stop_count",
        "Sustained commanded stops",
        "Count",
    ),
    (
        "active_command_window_s",
        "active_command_window_s",
        "Active-command windows",
        "Duration (s)",
    ),
    (
        "command_publication_window_s",
        "command_publication_window_s",
        "Command publication windows",
        "Duration (s)",
    ),
    (
        "final_position_residual_m",
        "final_position_residual_m",
        "Final AMCL position residual",
        "Residual (m)",
    ),
    (
        "final_orientation_residual_deg",
        "final_orientation_residual_deg",
        "Final AMCL orientation residual",
        "Residual (deg)",
    ),
    (
        "compensation_vy_delta_rmse",
        "compensation_vy_delta_rmse",
        "Lateral command modification",
        "RMS command difference (m/s)",
    ),
    (
        "compensation_wz_delta_rmse",
        "compensation_wz_delta_rmse",
        "Yaw-rate command modification",
        "RMS command difference (rad/s)",
    ),
]

MARKERS = {
    "SUCCEEDED": "o",
    "ABORTED": "x",
    "CANCELED": "s",
    "UNKNOWN": "D",
    "TIMEOUT": "^",
    "FAILED": "x",
}

OUTCOME_ORDER = [
    "SUCCEEDED",
    "ABORTED",
    "CANCELED",
    "TIMEOUT",
    "FAILED",
    "UNKNOWN",
]

# Dimensions are close to the final width of a two-column paper figure.
PANEL_FIGURE_SIZE = (7.2, 5.8)
COMMAND_FIGURE_SIZE = (7.2, 4.2)

PNG_DPI = 600

PUBLICATION_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # SVG text remains editable; the viewing system needs the font.
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
}


# ============================================================================
# DATA ACCESS
# ============================================================================

def number(row, key):
    """Return a finite numeric metric, or None when unavailable."""
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None

def load_declared_success_counts(path):
    """Load aggregate success counts without assigning trial outcomes."""
    if path is None:
        return {}

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Success counts file not found: {path}"
        )

    counts = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (
                row["scenario"].strip(),
                round(float(row["speed_mps"]), 2),
                row["configuration"].strip(),
            )

            successes = int(row["successes"])
            trials = int(row["trials"])

            if (
                key in counts
                or trials <= 0
                or not 0 <= successes <= trials
            ):
                raise ValueError(
                    f"Invalid or duplicate success counts: {key}"
                )

            counts[key] = (successes, trials)

    return counts

def outcome(row):
    """Read an outcome without treating missing labels as failures."""
    value = row.get("outcome")

    if value is None:
        return "UNKNOWN"

    value = str(value).strip()

    return value if value else "UNKNOWN"


def subset(rows, scenario, speed, config=None):
    """Select one scenario-speed group and optionally one configuration."""
    selected = []

    for row in rows:
        row_speed = number(row, "speed_mps")

        if row.get("scenario") != scenario:
            continue

        if row_speed is None or abs(row_speed - speed) >= 1e-6:
            continue

        if config is not None and row.get("configuration") != config:
            continue

        selected.append(row)

    return selected


def trial_sort_key(row):
    """Sort numeric trial identifiers numerically when possible."""
    value = str(row.get("trial_number", "")).strip()

    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def sorted_outcomes(rows):
    """Return observed outcome labels in a stable display order."""
    present = {outcome(row) for row in rows}

    ordered = [
        label
        for label in OUTCOME_ORDER
        if label in present
    ]

    ordered.extend(sorted(present.difference(OUTCOME_ORDER)))

    return ordered


# ============================================================================
# EXPORT AND COMMON STYLING
# ============================================================================

def save(fig, output_stem):
    """Export vector PDF/SVG and a 600-dpi PNG with identical content."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "bbox_inches": "tight",
        "pad_inches": 0.08,
        "facecolor": "white",
        "transparent": False,
    }

    try:
        with matplotlib.rc_context({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }):
            fig.savefig(
                output_stem.with_suffix(".pdf"),
                **options,
            )
            fig.savefig(
                output_stem.with_suffix(".svg"),
                **options,
            )
            fig.savefig(
                output_stem.with_suffix(".png"),
                dpi=PNG_DPI,
                **options,
            )
    finally:
        plt.close(fig)


def style_axis(ax):
    """Apply consistent, high-contrast axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(
        axis="x",
        length=0,
        pad=7,
        labelsize=9,
    )
    ax.tick_params(
        axis="y",
        length=3,
        width=0.8,
        labelsize=9,
    )

    ax.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def canvas(title, subtitle):
    """Create a four-panel figure with shared vertical scales."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=PANEL_FIGURE_SIZE,
        sharey=True,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.08,
        top=0.81,
        wspace=0.32,
        hspace=0.48,
    )

    fig.suptitle(
        title,
        x=0.5,
        y=0.985,
        fontsize=12,
        fontweight="semibold",
    )

    fig.text(
        0.5,
        0.935,
        fill(subtitle, width=88),
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
        linespacing=1.3,
    )

    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        _, _, panel_title, _ = panel

        ax.set_title(
            f"({chr(97 + index)}) {panel_title}",
            fontsize=10,
            fontweight="semibold",
            pad=12,
        )

        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(LABELS)
        ax.set_xlim(-0.55, len(ORDER) - 0.45)

        style_axis(ax)

    return fig, axes


def sample_count(ax, position, text):
    """Display counts in a reserved upper region of a panel."""
    ax.text(
        position,
        0.96,
        text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        fontweight="medium",
        linespacing=1.2,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
            "pad": 1.5,
        },
        zorder=6,
    )


def metric_axis_label(unit):
    """Wrap long labels while retaining their units."""
    replacements = {
        "RMS command difference (m/s)": (
            "RMS command\n"
            "difference (m/s)"
        ),
        "RMS command difference (rad/s)": (
            "RMS command\n"
            "difference (rad/s)"
        ),
    }

    return replacements.get(unit, unit)


# ============================================================================
# CONTINUOUS METRICS
# ============================================================================

def scatter_metric(
    rows,
    key,
    filename,
    title,
    unit,
    out,
    successful=False,
):
    """Plot trial values and medians without modifying their population."""
    selected = [
        row
        for row in rows
        if not successful or outcome(row) == "SUCCEEDED"
    ]

    values = [
        number(row, key)
        for row in selected
        if number(row, key) is not None
    ]

    if not values:
        return False

    population = (
        "Successful trials only"
        if successful
        else "All outcomes"
    )

    subtitle = (
        f"{population}; points = trials; bars = medians; "
        "n = valid/eligible trials."
    )

    if key in {
        "final_position_residual_m",
        "final_orientation_residual_deg",
    }:
        subtitle += " Reference: goal or initial-plan endpoint proxy."

    fig, axes = canvas(title, subtitle)

    # Preserve a common scale across all four scenario-speed panels.
    maximum = max(values)
    minimum = min(values)

    scale = maximum if maximum > 0 else 0.01
    upper = scale * 1.30
    lower = min(-scale * 0.035, minimum - scale * 0.035)

    is_count = key.endswith("commanded_stop_count")

    if is_count:
        upper = max(1.3, upper)

    for ax, panel in zip(axes.flat, PANELS):
        scenario, speed, _, _ = panel

        for index, configuration in enumerate(ORDER):
            group = sorted(
                subset(selected, scenario, speed, configuration),
                key=trial_sort_key,
            )

            valid = [
                (row, number(row, key))
                for row in group
                if number(row, key) is not None
            ]

            shifts = (
                np.linspace(-0.17, 0.17, len(valid))
                if len(valid) > 1
                else [0.0]
            )

            for shift, (_, value) in zip(shifts, valid):
                ax.scatter(
                    index + shift,
                    value,
                    s=42,
                    color=COLORS[index],
                    edgecolors="white",
                    linewidths=0.6,
                    alpha=0.95,
                    zorder=4,
                )

            if valid:
                median = float(np.median([
                    value
                    for _, value in valid
                ]))

                # Outline improves visibility when points overlap the median.
                ax.hlines(
                    median,
                    index - 0.25,
                    index + 0.25,
                    color="#222222",
                    linewidth=3.2,
                    zorder=2,
                )
                ax.hlines(
                    median,
                    index - 0.25,
                    index + 0.25,
                    color=COLORS[index],
                    linewidth=2.0,
                    zorder=3,
                )

            sample_count(
                ax,
                index,
                f"n={len(valid)}/{len(group)}",
            )

        ax.set_ylabel(metric_axis_label(unit), labelpad=7)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=is_count)
        )

    suffix = "_success_only" if successful else "_all_outcomes"

    save(fig, Path(out) / f"{filename}{suffix}")

    return True


# ============================================================================
# MISSION OUTCOMES
# ============================================================================

def success_plot(rows, out, declared_counts=None):
    """Plot trial outcomes and explicitly identified aggregate counts."""

    declared_counts = declared_counts or {}

    if (
        not any(outcome(row) != "UNKNOWN" for row in rows)
        and not declared_counts
    ):
        return False

    fig, axes = canvas(
        "Navigation success rate",
        (
            "Wilson 95% intervals. Aggregate-reported results "
            "are marked with an asterisk (*); individual "
            "UNKNOWN outcomes are never imputed."
        ),
    )

    aggregate_used = False

    for ax, panel in zip(axes.flat, PANELS):
        scenario, speed, _, _ = panel

        for index, configuration in enumerate(ORDER):
            group = subset(
                rows,
                scenario,
                speed,
                configuration,
            )

            known = [
                row
                for row in group
                if outcome(row) != "UNKNOWN"
            ]

            total = len(known)
            reached = sum(
                outcome(row) == "SUCCEEDED"
                for row in known
            )

            aggregate = False

            # Use declared aggregate results only when individual
            # mission outcomes are entirely unavailable.
            if total == 0:
                key = (
                    scenario,
                    round(speed, 2),
                    configuration,
                )

                if key in declared_counts:
                    reached, total = declared_counts[key]
                    aggregate = True
                    aggregate_used = True

                    if group and len(group) != total:
                        raise ValueError(
                            f"Trial count mismatch for {key}: "
                            f"{len(group)} individual rows "
                            f"versus {total} declared trials."
                        )

            if total:
                percentage = 100.0 * reached / total
                lower, upper = wilson(reached, total)

                ax.bar(
                    index,
                    percentage,
                    color=COLORS[index],
                    width=0.60,
                    edgecolor="#333333",
                    linewidth=0.5,
                    #hatch="//" if aggregate else None,
                    hatch=None,
                    zorder=2,
                )

                ax.errorbar(
                    index,
                    percentage,
                    yerr=[
                        [
                            max(
                                0.0,
                                percentage - 100.0 * lower,
                            )
                        ],
                        [
                            max(
                                0.0,
                                100.0 * upper - percentage,
                            )
                        ],
                    ],
                    fmt="none",
                    color="#222222",
                    elinewidth=1.3,
                    capsize=4,
                    capthick=1.3,
                    zorder=3,
                )
                """
                label = (
                    f"{reached}/{total}"
                    + ("*" if aggregate else "")
                )

                """
                label = f"{reached}/{total}"

            else:
                label = "NA"

            unknown = len(group) - len(known)

            if unknown and not aggregate:
                label += f"\n{unknown} unknown"

            sample_count(ax, index, label)

        ax.set_ylim(0, 132)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Success rate (%)", labelpad=7)

    """
    if aggregate_used:
        fig.text(
            0.5,
            0.015,
            "* Aggregate-reported counts; individual trial "
            "outcomes unavailable.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
    """

    save(fig, Path(out) / "success_rate")

    return True

def availability(rows, out):
    """Plot known and unknown outcome counts without imputing failures."""
    fig, axes = canvas(
        "Navigation outcome availability",
        (
            "Labels = known/total outcomes; "
            "gray bars indicate unknown outcomes, not failures."
        ),
    )

    maximum_group_size = max(
        (
            len(subset(rows, scenario, speed, configuration))
            for scenario, speed, _, _ in PANELS
            for configuration in ORDER
        ),
        default=0,
    )

    upper = max(2.0, maximum_group_size * 1.35 + 0.3)

    for ax, panel in zip(axes.flat, PANELS):
        scenario, speed, _, _ = panel

        for index, configuration in enumerate(ORDER):
            group = subset(rows, scenario, speed, configuration)

            known = sum(
                outcome(row) != "UNKNOWN"
                for row in group
            )

            unknown = len(group) - known

            ax.bar(
                index,
                known,
                color=COLORS[index],
                width=0.60,
                edgecolor="#333333",
                linewidth=0.5,
            )

            if unknown:
                ax.bar(
                    index,
                    unknown,
                    bottom=known,
                    color="#DDDDDD",
                    edgecolor="#777777",
                    linewidth=0.5,
                    hatch="//",
                    width=0.60,
                )

            sample_count(
                ax,
                index,
                f"{known}/{len(group)}",
            )

        ax.set_ylim(0, upper)
        ax.set_ylabel("Trials", labelpad=7)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=True)
        )

    save(fig, Path(out) / "outcome_availability")


# ============================================================================
# INDIVIDUAL COMMAND WINDOWS
# ============================================================================

def command_windows(rows, out):
    """Plot command windows per trial with outcome-specific markers."""
    records = []

    window_definitions = [
        (
            "command_window_s",
            "command_windows",
            "Command publication windows",
        ),
        (
            "active_command_window_s",
            "active_command_windows",
            "Active-command windows",
        ),
    ]

    for scenario, speed, panel_title, slug in PANELS:
        group = subset(rows, scenario, speed)

        for key, prefix, heading in window_definitions:
            filename = f"{prefix}_{slug}"

            values = [
                number(row, key)
                for row in group
                if number(row, key) is not None
            ]

            if not values:
                records.append(
                    f"- {filename}: NOT GENERATED — no eligible values"
                )
                continue

            fig, ax = plt.subplots(figsize=COMMAND_FIGURE_SIZE)

            fig.subplots_adjust(
                left=0.11,
                right=0.985,
                bottom=0.31,
                top=0.80,
            )

            style_axis(ax)

            fig.suptitle(
                heading,
                fontsize=12,
                fontweight="semibold",
                y=0.98,
            )

            ax.set_title(
                f"{panel_title} | All outcomes",
                fontsize=10,
                pad=12,
            )

            position = 0
            centers = []

            for index, configuration in enumerate(ORDER):
                trials = sorted(
                    [
                        row
                        for row in group
                        if row.get("configuration") == configuration
                    ],
                    key=trial_sort_key,
                )

                start = position

                for row in trials:
                    value = number(row, key)
                    status = outcome(row)

                    if value is not None:
                        unknown = status == "UNKNOWN"

                        ax.bar(
                            position,
                            value,
                            color=COLORS[index],
                            width=0.72,
                            alpha=0.45 if unknown else 0.90,
                            edgecolor="#333333",
                            linewidth=0.5,
                            zorder=2,
                        )

                        ax.scatter(
                            position,
                            value,
                            marker=MARKERS.get(status, "D"),
                            color="#222222",
                            s=34,
                            linewidths=1.0,
                            zorder=4,
                        )
                    else:
                        ax.text(
                            position,
                            0.025,
                            "NA",
                            ha="center",
                            va="bottom",
                            transform=ax.get_xaxis_transform(),
                            fontsize=8,
                            color="#555555",
                        )

                    ax.text(
                        position,
                        -0.045,
                        str(row.get("trial_number", "")),
                        ha="center",
                        va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                    )

                    position += 1

                if trials:
                    centers.append((start + position - 1) / 2.0)
                else:
                    centers.append(position)
                    position += 1

                # Leave one empty position between configurations.
                position += 1

            maximum = max(values)
            upper = maximum * 1.20 if maximum > 0 else 1.0

            ax.set_ylim(0, upper)
            ax.set_xlim(-0.75, max(position - 0.5, 0.75))
            ax.set_xticks(centers)
            ax.set_xticklabels(LABELS, fontsize=9)
            ax.tick_params(axis="x", pad=25)

            ax.set_ylabel("Duration (s)", labelpad=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            observed_outcomes = sorted_outcomes(group)

            handles = [
                Line2D(
                    [],
                    [],
                    color="#222222",
                    marker=MARKERS.get(status, "D"),
                    linestyle="",
                    markersize=6,
                    label=status.title(),
                )
                for status in observed_outcomes
            ]

            if handles:
                fig.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.01),
                    ncol=min(3, len(handles)),
                    fontsize=9,
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.5,
                )

            save(fig, Path(out) / filename)
            records.append(f"- {filename}: generated")

    return records


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def command_windows_overview(rows, out):
    """Plot all command publication windows with trial IDs and outcomes.

    Values come directly from command_window_s.
    Unknown individual outcomes remain explicitly marked as unknown.
    No median is drawn in this figure.
    """
    values = [
        number(row, "command_window_s")
        for row in rows
        if number(row, "command_window_s") is not None
    ]

    if not values:
        return False

    fig, axes = canvas(
        "Command publication windows — all outcomes",
        (
            "Numbers identify trials; symbols indicate outcomes. "
            "Short unsuccessful windows do not indicate faster completion."
        ),
    )

    # Reserve space for the shared outcome legend.
    fig.subplots_adjust(
        bottom=0.15,
        hspace=0.55,
    )

    # Restore the historical colors for the original configurations.
    colors = {
        "calibrated": "#0072B2",
        "Odometry-only": "#AA4499",
        "dog": "#E69F00",
        "g1_nav": "#009E73",
    }

    markers = {
        "SUCCEEDED": "o",
        "ABORTED": "X",
        "CANCELED": "s",
        "TIMEOUT": "^",
        "FAILED": "P",
        "UNKNOWN": "D",
    }

    display_names = {
        "SUCCEEDED": "Reached",
        "ABORTED": "Aborted",
        "CANCELED": "Canceled",
        "TIMEOUT": "Timeout",
        "FAILED": "Failed",
        "UNKNOWN": "Unknown",
    }

    maximum = max(values)
    upper = maximum * 1.23 if maximum > 0 else 1.0
    displayed_outcomes = set()

    for ax, panel in zip(axes.flat, PANELS):
        scenario, speed, _, _ = panel

        for index, configuration in enumerate(ORDER):
            trials = sorted(
                subset(rows, scenario, speed, configuration),
                key=trial_sort_key,
            )

            # Keep a horizontal slot for every trial, including missing values.
            offsets = (
                np.linspace(-0.32, 0.32, len(trials))
                if len(trials) > 1
                else [0.0]
            )

            for trial_index, (offset, row) in enumerate(zip(offsets, trials)):
                x = index + offset
                value = number(row, "command_window_s")
                trial_id = str(row.get("trial_number", "")).strip()

                if trial_id.isdigit():
                    trial_id = f"{int(trial_id):02d}"

                if value is None:
                    ax.text(
                        x,
                        0.025,
                        f"{trial_id}\nNA",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#666666",
                    )
                    continue

                status = outcome(row)
                displayed_outcomes.add(status)

                marker = markers.get(status, "D")
                color = colors[configuration]

                # Hollow markers distinguish unknown outcomes.
                ax.scatter(
                    x,
                    value,
                    marker=marker,
                    s=48,
                    facecolors="none" if status == "UNKNOWN" else color,
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=4,
                )

                # Alternate label heights to reduce overlap between trials.
                label_offset = 7 if trial_index % 2 == 0 else 17

                ax.annotate(
                    trial_id,
                    xy=(x, value),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="semibold",
                    color="#222222",
                    zorder=5,
                )

        ax.set_ylim(0, upper)
        ax.set_ylabel("Command window (s)", labelpad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(
            ["Combined", "Odom-\nonly", "Dog", "g1_nav"],
            fontsize=9,
        )

    ordered_statuses = [
        status
        for status in OUTCOME_ORDER
        if status in displayed_outcomes
    ]
    ordered_statuses.extend(
        sorted(displayed_outcomes.difference(OUTCOME_ORDER))
    )

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=markers.get(status, "D"),
            markersize=6,
            markerfacecolor=(
                "none" if status == "UNKNOWN" else "#444444"
            ),
            markeredgecolor="#444444",
            label=display_names.get(status, status.title()),
        )
        for status in ordered_statuses
    ]

    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(4, len(handles)),
            frameon=False,
            fontsize=9,
            columnspacing=1.3,
            handletextpad=0.4,
        )

    save(fig, Path(out) / "command_windows_all_outcomes")
    return True

def campaign_figures(
    results,
    summaries,
    config,
    output,
    success_counts_path=None,
):
    """Generate campaign figures and an export manifest.

    Parameters
    ----------
    results:
        Iterable of trial-level dictionaries.
    summaries:
        Retained for compatibility with the existing analysis pipeline.
    config:
        Retained for compatibility with the existing analysis pipeline.
    output:
        Destination directory for figures and FIGURE_MANIFEST.md.

    Notes
    -----
    Figures are derived from trial-level results. Existing grouped summaries
    are not substituted for individual observations.

    Terminal position and orientation figures retain their original,
    independent validity filtering. If a table requires both quantities
    to be valid simultaneously, verify its population against these figures.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    rows = list(results)

    declared_counts = load_declared_success_counts(
        success_counts_path
    )
    notes = [
        "# Paper figure manifest",
        "",
        "## Export settings",
        "",
        "- PDF: vector graphics with embedded TrueType fonts.",
        "- SVG: vector graphics with editable text.",
        f"- PNG: {PNG_DPI} dpi, white background.",
        "- Configuration colors and labels are consistent across figures.",
        "- Four-panel figures use a common vertical scale.",
        "",
        "## Population definitions",
        "",
        "- All-outcome and successful-only populations are separate.",
        "- Successful-only figures require outcome == SUCCEEDED.",
        "- Missing outcomes remain UNKNOWN.",
        "- Unknown outcomes are excluded from success-rate denominators.",
        "- Continuous figures retain valid metrics from unknown outcomes.",
        "- Sample labels show valid/eligible trials.",
        "- Terminal position and orientation validity is assessed separately.",
        "",
        "## Generated figures",
        "",
    ]

    with matplotlib.rc_context(PUBLICATION_STYLE):
        availability(rows, output)
        notes.append("- outcome_availability: generated")

        if success_plot(
            rows,
            output,
            declared_counts=declared_counts,
        ):
            notes.append("- success_rate: generated")
        else:
            notes.append(
                "- success_rate: NOT GENERATED — "
                "no known outcomes; supply trial annotations"
            )

        for key, name, title, unit in METRICS:
            for successful in (False, True):

                # Use the dedicated outcome-aware layout for this figure.
                if name == "command_windows" and not successful:
                    generated = command_windows_overview(rows, output)
                else:
                    generated = scatter_metric(
                        rows=rows,
                        key=key,
                        filename=name,
                        title=title,
                        unit=unit,
                        out=output,
                        successful=successful,
                    )

                population = (
                    "success_only"
                    if successful
                    else "all_outcomes"
                )

                status = (
                    "generated"
                    if generated
                    else "NOT GENERATED — no eligible values"
                )

                notes.append(
                    f"- {name}_{population}: {status}"
                )

        notes.extend(command_windows(rows, output))
     

    notes.extend([
        "",
        "## Interpretation",
        "",
        (
            "- Command publication windows and active-command "
            "windows are different metrics."
        ),
        (
            "- Publication-window durations are not verified "
            "goal-acceptance-to-result durations."
        ),
        (
            "- Used-odometry/AMCL disagreement retains the "
            "upstream initial-SE2 alignment definition."
        ),
        (
            "- AMCL is a coupled estimate, not independent "
            "trajectory ground truth."
        ),
        (
            "- Historical filenames use publication-window "
            "metrics supplied through legacy_ fields."
        ),
        (
            "- Historical metric definitions were reimplemented "
            "upstream from METHODS.md; this module only plots "
            "the supplied values."
        ),
        (
            "- Historical sustained-stop counts include initial "
            "and terminal zero commands; active-window counts "
            "remain separate."
        ),
        (
            "- Command modification plots quantify command "
            "differences, not independently measured physical "
            "drift reduction."
        ),
        (
            "- Final AMCL residuals use the upstream reference "
            "selection: requested goal or initial-plan endpoint proxy."
        ),
        "",
        "## Configuration labels",
        "",
        "- Combined: calibrated odometry with command compensation.",
        "- Odom-only: calibrated odometry without command compensation.",
        "- Dog: Dog odometry configuration.",
        "- g1_nav: reference configuration.",
        "",
    ])

    manifest_path = output / "FIGURE_MANIFEST.md"
    manifest_path.write_text(
        "\n".join(notes),
        encoding="utf-8",
    )