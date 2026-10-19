#!/usr/bin/env python3
"""Render the inference-service figures from results/*.csv.

    python make_figures.py

Static light-mode PNGs.  The four bounded configs use categorical slots 1-4 in
fixed order; the supplied baseline is a neutral gray reference.  The Node
palette validator could not be run because Node.js is not installed; the palette
is the documented pre-validated reference instance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GRAY = "#898781"

CONFIG_ORDER = ["baseline", "bounded", "batched", "wide_fast", "priority"]
CONFIG_COLORS = {
    "baseline": GRAY,
    "bounded": ORANGE,
    "batched": AQUA,
    "wide_fast": YELLOW,
    "priority": BLUE,
    "priority_batched": MAGENTA,
}
CONFIG_LABELS = {
    "baseline": "Baseline",
    "bounded": "Bounded (4w)",
    "batched": "Batched (4w)",
    "wide_fast": "Wide (16w)",
    "priority": "Priority (4w)",
    "priority_batched": "Combined (8w)",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.055,
        y=0.985,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.055,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=SECONDARY,
    )


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=3, width=0.8)
    ax.set_ylabel(ylabel, color=SECONDARY)


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS / "scenario_summary.csv")
    return summary


def figure_overload(summary: pd.DataFrame) -> None:
    frame = summary[summary["scenario"] == "overload"].set_index("config")
    combined_path = RESULTS / "combined_policy.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        combined_row = combined[combined["scenario"] == "overload"].iloc[0]
        frame.loc["priority_batched"] = combined_row
        order = CONFIG_ORDER + ["priority_batched"]
    else:
        order = CONFIG_ORDER
    frame = frame.reindex(order)
    positions = np.arange(len(order))
    colors = [CONFIG_COLORS[c] for c in order]

    fig, (ax_lat, ax_err) = plt.subplots(
        2,
        1,
        figsize=(7.6, 5.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    p95 = frame["latency_p95_all_ms"].to_numpy(dtype=float)
    err = frame["error_rate"].to_numpy(dtype=float)
    bars = ax_lat.bar(positions, p95, width=0.55, color=colors)
    for bar, value in zip(bars, p95):
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2,
            value + 25,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY,
        )
    ax_lat.set_ylim(0, max(p95) * 1.13)
    ax_lat.set_ylabel("p95 latency, all requests (ms)", color=SECONDARY)
    style_axes(ax_lat, "p95 latency, all requests (ms)")

    bars = ax_err.bar(positions, err, width=0.55, color=colors)
    for bar, value in zip(bars, err):
        ax_err.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY,
        )
    ax_err.set_ylim(0, 1.12)
    ax_err.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_err.set_ylabel("Error rate", color=SECONDARY)
    ax_err.set_xticks(positions)
    ax_err.set_xticklabels([CONFIG_LABELS[c] for c in order])
    style_axes(ax_err, "Error rate")
    ax_err.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    figure_header(
        fig,
        "Overload: admission control trades errors for latency",
        "200 requests, 64 concurrent; 3 repeats; 30 ms simulated model time per request.",
    )
    fig.subplots_adjust(left=0.115, right=0.975, top=0.83, bottom=0.12, hspace=0.18)
    path = FIGURES / "fig1_overload.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_mixed(mixed: pd.DataFrame) -> None:
    frame = mixed.set_index("config").reindex(CONFIG_ORDER)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    ax.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    ax.text(
        0.55,
        0.58,
        "equal treatment",
        fontsize=8,
        color=MUTED,
        rotation=39,
        ha="center",
        va="center",
    )

    offsets = {
        "baseline": (-9, 10),
        "bounded": (-9, 9),
        "priority": (-9, -12),
    }
    aligns = {
        "baseline": ("right", "bottom"),
        "bounded": ("right", "bottom"),
        "priority": ("right", "top"),
    }
    for config in CONFIG_ORDER:
        x = float(frame.loc[config, "batch_error_rate"])
        y = float(frame.loc[config, "interactive_error_rate"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=CONFIG_COLORS[config],
            edgecolors=SURFACE,
            linewidths=2.0,
            zorder=3,
            label=CONFIG_LABELS[config],
        )
        if config in offsets:
            dx, dy = offsets[config]
            ha, va = aligns[config]
            ax.annotate(
                CONFIG_LABELS[config],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color=SECONDARY,
                ha=ha,
                va=va,
            )

    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.03, 1.06)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Batch user error rate", color=SECONDARY)
    ax.set_ylabel("Interactive user error rate", color=SECONDARY)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, length=3, width=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.885),
        ncol=5,
        frameon=False,
        labelcolor=SECONDARY,
        handletextpad=0.2,
        columnspacing=1.0,
        markerscale=0.9,
    )

    figure_header(
        fig,
        "Who bears the overload?",
        "Mixed workload: 70% interactive / 30% batch. Points below the line protect interactive users.",
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.78, bottom=0.13)
    path = FIGURES / "fig2_who_bears_overload.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    mixed = pd.read_csv(RESULTS / "mixed_by_class.csv")
    figure_overload(summary)
    figure_mixed(mixed)


if __name__ == "__main__":
    main()
