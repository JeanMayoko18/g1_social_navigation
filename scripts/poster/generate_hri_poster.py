#!/usr/bin/env python3
"""Generate publication/poster figures from the supplied aggregated HRI data.

Dependencies: numpy, matplotlib (>=3.6).
Example: python generate_hri_poster.py --output figures --width-cm 30 --dpi 600

DIFF and N are transcribed unchanged from the supplied script. The figures
are descriptive: aggregate differences cannot establish significance, CIs,
participant-level effects, or physical safety. N is labelled as complete
participant pairs following the source script, not independently verified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter

SCENARIOS = ["S1a", "S1b", "S2", "S3"]
CRITERIA = ["Safety", "Comfort", "Predictability", "Social responsiveness",
            "Navigation efficiency", "Trust", "Consideration", "Restaurant acceptability"]
DIFF = np.array([
    [1.07, .29, .36, .29], [1.43, .79, .43, .79],
    [1.23, .85, .50, -.07], [.92, .71, .57, .50],
    [.00, .69, .07, .07], [1.00, 1.07, .71, .71],
    [.54, .42, .62, .46], [.92, 1.29, .93, .69],
])
N = np.array([
    [14, 14, 14, 14], [14, 14, 14, 14], [13, 13, 14, 14],
    [13, 14, 14, 14], [14, 13, 14, 14], [14, 14, 14, 14],
    [13, 12, 13, 13], [13, 14, 14, 13],
])
INK = "#153448"
TEAL = "#176F79"
RED = "#B34D5B"
GOLD = "#D69C40"
GRID = "#D9E2E5"
MUTED = "#536974"
COLORS = [TEAL, "#3266AD", "#B96819", "#86538F"]
MARKERS = ["o", "s", "D", "^"]


def signed(value: float) -> str:
    return "0.00" if abs(value) < 0.0000001 else f"{value:+.2f}".replace("-", "−")


def criterion_label(text: str) -> str:
    return {"Social responsiveness": "Social\nresponsiveness",
            "Navigation efficiency": "Navigation\nefficiency",
            "Restaurant acceptability": "Restaurant\nacceptability"}.get(text, text)


def contrast_text(rgba) -> str:
    """Choose white/dark text by relative luminance, not by data threshold."""
    rgb = np.array(rgba[:3])
    rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(rgb @ np.array([.2126, .7152, .0722]))
    return "white" if luminance < .32 else INK


def validate_data() -> None:
    if DIFF.shape != (len(CRITERIA), len(SCENARIOS)) or N.shape != DIFF.shape:
        raise ValueError("DIFF/N dimensions must match criteria and scenarios.")
    if not np.isfinite(DIFF).all() or not np.isfinite(N).all():
        raise ValueError("All values must be finite.")
    if not ((N > 0) & (N == np.floor(N))).all():
        raise ValueError("N must contain positive integer counts.")


class Exporter:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.scale = args.width_cm / 30.0
        self.font_scale = self.scale * args.font_scale
        self.records = []

    def font(self, pt):
        return pt * self.font_scale

    def canvas(self, ratio=.69):
        return plt.figure(figsize=(self.args.width_cm / 2.54,
                                   self.args.width_cm * ratio / 2.54),
                          facecolor="white")

    def heading(self, fig, title, subtitle):
        fig.text(.035, .967, title, ha="left", va="top",
                 fontsize=self.font(25), weight="bold", color=INK)
        fig.text(.035, .902, subtitle, ha="left", va="top",
                 fontsize=self.font(15.5), color=MUTED)

    def footer(self, fig, text):
        fig.text(.035, .035, text, ha="left", va="bottom",
                 fontsize=self.font(13.5), color=MUTED)

    def save(self, fig, stem):
        """Keep the exact canvas size; no tight bounding box resizing."""
        fig.canvas.draw()
        # Detect text outside the canvas before delivery. This also catches
        # excessive --font-scale values rather than silently clipping labels.
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        from matplotlib.text import Text
        hidden_ticks = set()
        for ax in fig.axes:
            for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                lo, hi = sorted(limits)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not lo <= tick.get_loc() <= hi:
                        hidden_ticks.update([id(tick.label1), id(tick.label2)])
        for artist in fig.findobj(Text):
            if not artist.get_visible() or not artist.get_text() or id(artist) in hidden_ticks:
                continue
            # Some unused tick labels exist as artists outside fixed limits.
            if artist.axes is not None and artist.get_clip_on():
                continue
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < canvas.x0 - 2 or box.x1 > canvas.x1 + 2
                                            or box.y0 < canvas.y0 - 2 or box.y1 > canvas.y1 + 2):
                raise ValueError(f"Text exceeds canvas in {stem}: {artist.get_text()!r}. "
                                 "Reduce --font-scale or adjust the layout.")
        opts = dict(facecolor="white", transparent=self.args.transparent)
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            fig.savefig(self.out / f"{stem}.svg", **opts)
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            fig.savefig(self.out / f"{stem}.text.svg", **opts)
        fig.savefig(self.out / f"{stem}.pdf", **opts)
        fig.savefig(self.out / f"{stem}.png", dpi=self.args.dpi, **opts)
        self.records.append({"name": stem, "width_cm": self.args.width_cm,
                             "height_cm": float(fig.get_figheight() * 2.54),
                             "png_dpi": self.args.dpi})
        plt.close(fig)


def direction_figure(e):
    positive, zero, negative = [int(v) for v in
        ((DIFF > 0).sum(), (DIFF == 0).sum(), (DIFF < 0).sum())]
    fig = e.canvas(.55)
    e.heading(fig, "Direction of HRI rating differences",
              "Criterion–scenario comparisons using the reported D − A values")
    ax = fig.add_axes([.025, .16, .46, .63])
    ax.pie([positive, zero, negative], colors=[TEAL, GOLD, RED],
           startangle=90, counterclock=False,
           wedgeprops={"width": .24, "edgecolor": "white", "linewidth": 1.8 * e.scale})
    ax.text(0, .08, f"{positive}/{DIFF.size}", ha="center", va="center",
            fontsize=e.font(39), weight="bold", color=INK)
    ax.text(0, -.23, "positive", ha="center", va="center", fontsize=e.font(18), color=MUTED)
    for y, color, count, label in zip([.68, .49, .30], [TEAL, GOLD, RED],
                                     [positive, zero, negative],
                                     ["D > A", "D = A", "D < A"]):
        fig.text(.53, y, str(count), color=color, fontsize=e.font(33), weight="bold", va="center")
        fig.text(.65, y + .02, label, color=INK, fontsize=e.font(21), weight="bold", va="center")
        fig.text(.65, y - .045, "comparisons" if count != 1 else "comparison",
                 color=MUTED, fontsize=e.font(16), va="center")
    e.footer(fig, "32 cells = 8 criteria × 4 scenarios. These are not participant counts.")
    e.save(fig, "01_donut_direction")


def heatmap_figure(e):
    fig = e.canvas(.80)
    e.heading(fig, "HRI rating differences by scenario", "Reported mean differences: D − A")
    ax = fig.add_axes([.29, .23, .675, .60])
    limit = max(1.5, float(np.ceil(np.abs(DIFF).max() * 2) / 2))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list("difference", [RED, "#F7F8F7", TEAL])
    im = ax.imshow(DIFF, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(8), [criterion_label(v) for v in CRITERIA], fontsize=e.font(17))
    ax.set_xticks(range(4), SCENARIOS, fontsize=e.font(19), weight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0, pad=10 * e.scale)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5 * e.scale)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(DIFF.shape):
        ax.text(j, i, signed(DIFF[i, j]), ha="center", va="center",
                fontsize=e.font(20), weight="bold", color=contrast_text(cmap(norm(DIFF[i, j]))))
    cax = fig.add_axes([.39, .135, .48, .023])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=e.font(14), length=0)
    cb.set_label("D − A (rating points)", fontsize=e.font(14), labelpad=3)
    e.footer(fig, "Positive: higher rating for D. Descriptive differences; no significance test.")
    e.save(fig, "02_heatmap_poster")


def ranked_figure(e):
    means = DIFF.mean(axis=1)  # Equal weight per scenario; NOT pooled participants.
    order = np.argsort(-means, kind="stable")
    values = means[order]
    fig = e.canvas(.73)
    e.heading(fig, "HRI rating differences by criterion", "Equal-weight mean of the four scenario differences")
    ax = fig.add_axes([.295, .20, .64, .61])
    y = np.arange(len(CRITERIA))
    ax.barh(y, values, height=.58, color=[TEAL if v >= 0 else RED for v in values], zorder=3)
    ax.set_yticks(y, [criterion_label(CRITERIA[i]) for i in order], fontsize=e.font(18))
    ax.invert_yaxis()
    ax.set_xlim(min(0, float(values.min()) - .20) if values.min() < 0 else 0,
                max(1.15, float(values.max()) + .22))
    ax.xaxis.set_major_locator(MultipleLocator(.25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="x", labelsize=e.font(14), length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=11 * e.scale)
    ax.grid(axis="x", color=GRID, linewidth=.8 * e.scale, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(0, color=INK, linewidth=.9 * e.scale)
    for yy, val in zip(y, values):
        ax.text(val + (.025 if val >= 0 else -.025), yy, signed(val), va="center",
                ha="left" if val >= 0 else "right", fontsize=e.font(20), weight="bold", color=INK)
    ax.set_xlabel("Mean D − A (rating points)", fontsize=e.font(16), labelpad=12)
    e.footer(fig, "Descriptive averages from rounded inputs; no confidence intervals inferred.")
    e.save(fig, "03_criteria_ranked")


def scenario_figure(e):
    fig = e.canvas(.80)
    e.heading(fig, "HRI differences across scenarios", "Each point represents one reported criterion–scenario difference")
    ax = fig.add_axes([.29, .255, .67, .57])
    y = np.arange(8)
    for yy in y[::2]:
        ax.axhspan(yy - .48, yy + .48, color="#F1F5F6", zorder=0)
    for j, (name, color, marker) in enumerate(zip(SCENARIOS, COLORS, MARKERS)):
        ax.scatter(DIFF[:, j], y + (j - 1.5) * .18, s=105 * e.scale**2,
                   marker=marker, color=color, edgecolor="white", linewidth=.65 * e.scale,
                   label=name, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.1 * e.scale, zorder=2)
    ax.set_yticks(y, [criterion_label(v) for v in CRITERIA], fontsize=e.font(17))
    ax.set_ylim(7.6, -.6)
    ax.set_xlim(min(-.25, float(DIFF.min()) - .15), max(1.6, float(DIFF.max()) + .15))
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.grid(axis="x", color=GRID, linewidth=.8 * e.scale)
    ax.tick_params(axis="y", length=0, pad=10 * e.scale)
    ax.tick_params(axis="x", length=0, labelsize=e.font(15), pad=8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("D − A (rating points)", fontsize=e.font(16), labelpad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.625, .105),
               ncol=4, frameon=False, fontsize=e.font(17), handletextpad=.35, columnspacing=1.1)
    e.footer(fig, "Colour and marker shape identify scenarios. No uncertainty estimates are available.")
    e.save(fig, "04_scenario_dotplot")


def counts_figure(e):
    fig = e.canvas(.76)
    e.heading(fig, "Available participant pairs", "Complete pairs per criterion and scenario, as labelled in the source")
    ax = fig.add_axes([.29, .14, .675, .66])
    cmap = LinearSegmentedColormap.from_list("counts", ["#F2F6F7", "#C4DDDF"])
    ax.imshow(N, cmap=cmap, vmin=N.min(), vmax=max(N.min()+1, N.max()), aspect="auto")
    ax.set_yticks(range(8), [criterion_label(v) for v in CRITERIA], fontsize=e.font(17))
    ax.set_xticks(range(4), SCENARIOS, fontsize=e.font(19), weight="bold")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5 * e.scale)
    ax.tick_params(which="both", length=0, pad=10 * e.scale)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(N.shape):
        ax.text(j, i, str(N[i, j]), ha="center", va="center", fontsize=e.font(22), weight="bold", color=INK)
    e.footer(fig, "Do not sum these cells to obtain the number of unique participants.")
    e.save(fig, "05_paired_counts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--width-cm", type=float, default=30, help="Exact physical width of each exported figure (default: 30 cm).")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution only; SVG/PDF remain vector.")
    parser.add_argument("--font-scale", type=float, default=1, help="Optional text multiplier; inspect output if changed.")
    parser.add_argument("--transparent", action="store_true", help="Transparent figure/axes backgrounds.")
    args = parser.parse_args(argv)
    if not 8 <= args.width_cm <= 100 or not 72 <= args.dpi <= 1200 or not .6 <= args.font_scale <= 1.5:
        parser.error("Use width 8–100 cm, DPI 72–1200 and font scale 0.6–1.5.")
    validate_data()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.bbox": None, "axes.unicode_minus": True})
    e = Exporter(args)
    for draw in [direction_figure, heatmap_figure, ranked_figure, scenario_figure, counts_figure]:
        draw(e)
    with (e.out / "source_values.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["criterion", "scenario", "reported_mean_difference_D_minus_A", "reported_complete_pairs_N"])
        for i, criterion in enumerate(CRITERIA):
            for j, scenario in enumerate(SCENARIOS):
                writer.writerow([criterion, scenario, f"{DIFF[i,j]:.2f}", N[i,j]])
    summary = {"figures": e.records, "positive_cells": int((DIFF > 0).sum()),
               "reported_zero_cells": int((DIFF == 0).sum()), "negative_cells": int((DIFF < 0).sum()),
               "unweighted_means": dict(zip(CRITERIA, DIFF.mean(axis=1).tolist())),
               "note": "Signs and means use the supplied rounded differences. No inferential statistics."}
    (e.out / "figure_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(e.records)} figures as SVG, editable-text SVG, PDF and {args.dpi}-dpi PNG in {e.out}")


if __name__ == "__main__":
    main()

"""
python generate_hri_poster.py --output figures --width-cm 30 --dpi 600
"""