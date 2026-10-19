#!/usr/bin/env python3
"""Render the evaluation-runner figures from results/*.csv.

    python make_figures.py

Static light-mode PNGs.  Four categorical slots are used in fixed order from a
documented, pre-validated palette; the Node validator could not be run because
Node.js is not installed on this machine.
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

# Documented categorical slots 1-4 (light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
ADAPTER_COLORS = {
    "fast": BLUE,
    "transient": ORANGE,
    "slow": AQUA,
    "truncating": YELLOW,
}
ADAPTER_LABELS = {
    "fast": "Fast (baseline)",
    "transient": "Transient",
    "slow": "Slow / timeout",
    "truncating": "Truncating",
}
ADAPTER_ORDER = ["fast", "transient", "slow", "truncating"]

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


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=3, width=0.8)
    ax.set_ylabel(ylabel, color=SECONDARY)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


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


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULTS / "adapter_summary.csv")
    budget = pd.read_csv(RESULTS / "budget_outcomes.csv")
    return summary, budget


def figure_exclusion_bias(summary: pd.DataFrame) -> None:
    frame = summary[summary["budget"] == summary["budget"].max()].set_index("adapter")
    frame = frame.loc[ADAPTER_ORDER]
    positions = np.arange(len(ADAPTER_ORDER))
    width = 0.34

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.6, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )

    ax_top.bar(
        positions - width / 2 - 0.01,
        frame["accuracy_all"],
        width=width,
        color=BLUE,
        label="Failure-inclusive (all runs)",
    )
    ax_top.bar(
        positions + width / 2 + 0.01,
        frame["accuracy_completed"],
        width=width,
        color=ORANGE,
        label="Completed-only (timeouts/errors dropped)",
    )

    for i, adapter in enumerate(ADAPTER_ORDER):
        value = frame.loc[adapter, "accuracy_all"]
        ax_top.text(
            positions[i] - width / 2 - 0.01,
            value + 0.012,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY,
        )
        if adapter in {"slow", "truncating"}:
            value2 = frame.loc[adapter, "accuracy_completed"]
            ax_top.text(
                positions[i] + width / 2 + 0.01,
                value2 + 0.012,
                f"{value2:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=SECONDARY,
            )

    ax_top.set_ylim(0, 1.12)
    ax_top.set_ylabel("Accuracy", color=SECONDARY)
    ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_top.grid(axis="y", color=GRID, linewidth=0.8)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(colors=MUTED, length=3, width=0.8)
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.895),
        ncol=2,
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.4,
        columnspacing=1.8,
    )

    ax_bottom.bar(
        positions,
        frame["retained_fraction"],
        width=0.55,
        color=SECONDARY,
    )
    for i, adapter in enumerate(ADAPTER_ORDER):
        value = frame.loc[adapter, "retained_fraction"]
        ax_bottom.text(
            positions[i],
            value + 0.012,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SECONDARY,
        )
    ax_bottom.set_ylim(0, 1.12)
    ax_bottom.set_ylabel("Runs retained", color=SECONDARY)
    ax_bottom.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_bottom.set_xticks(positions)
    ax_bottom.set_xticklabels([ADAPTER_LABELS[a] for a in ADAPTER_ORDER])
    ax_bottom.set_xlabel("Adapter")
    ax_bottom.grid(axis="y", color=GRID, linewidth=0.8)
    ax_bottom.set_axisbelow(True)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    ax_bottom.tick_params(colors=MUTED, length=3, width=0.8)

    figure_header(
        fig,
        "Excluding failures hides the user experience",
        "40 cases x 5 repeats. Bottom: share of runs left if every failed run is excluded.",
    )
    fig.subplots_adjust(left=0.1, right=0.975, top=0.79, bottom=0.11, hspace=0.18)
    path = FIGURES / "fig1_exclusion_bias.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_retry_tradeoff(summary: pd.DataFrame) -> None:
    fig, (ax_acc, ax_cost) = plt.subplots(
        2,
        1,
        figsize=(7.6, 5.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    for adapter in ADAPTER_ORDER:
        frame = summary[summary["adapter"] == adapter].sort_values("budget")
        color = ADAPTER_COLORS[adapter]
        ax_acc.plot(
            frame["budget"],
            frame["accuracy_all"],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            label=ADAPTER_LABELS[adapter],
        )
        ax_cost.plot(
            frame["budget"],
            frame["cost_per_correct"],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )

    ax_acc.annotate(
        "fast 100%",
        xy=(3, 1.0),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=SECONDARY,
    )
    ax_acc.annotate(
        "transient 99%",
        xy=(3, 0.99),
        xytext=(8, 2),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=SECONDARY,
    )
    ax_acc.annotate(
        "slow / truncating 65%",
        xy=(3, 0.65),
        xytext=(8, -14),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=SECONDARY,
    )
    ax_cost.annotate(
        "fast 1.00",
        xy=(3, 1.0),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=SECONDARY,
    )
    ax_cost.annotate(
        "slow 4.23",
        xy=(3, 4.231),
        xytext=(8, -4),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=SECONDARY,
    )
    ax_cost.annotate(
        "transient 1.50 / truncating 1.54",
        xy=(3, 1.52),
        xytext=(8, -18),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color=SECONDARY,
    )

    ax_acc.set_ylim(0.55, 1.08)
    ax_acc.set_ylabel("Failure-inclusive accuracy", color=SECONDARY)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_cost.set_ylabel("Cost per correct answer", color=SECONDARY)
    ax_cost.set_xlabel("Retry budget (extra attempts after the first)")
    ax_cost.set_xticks([0, 1, 2, 3])

    for ax in (ax_acc, ax_cost):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=MUTED, length=3, width=0.8)

    ax_acc.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=4,
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.3,
        columnspacing=1.4,
    )

    figure_header(
        fig,
        "Retries help transient failures, not timeouts",
        "50 ms timeout; cost = 1 per attempt + 0.5 per timeout. 200 runs per adapter.",
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.81, bottom=0.09, hspace=0.16)
    path = FIGURES / "fig2_retry_tradeoff.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_failure_modes(summary: pd.DataFrame) -> None:
    frame = summary[summary["budget"] == summary["budget"].max()].set_index("adapter")
    frame = frame.loc[ADAPTER_ORDER]
    total = frame["runs"]
    outcomes = {
        "Correct": frame["correct_runs"] / total,
        "Timeout": frame["timeout_runs"] / total,
        "Error": frame["error_runs"] / total,
        "Wrong output": frame["wrong_runs"] / total,
    }
    colors = {"Correct": BLUE, "Timeout": ORANGE, "Error": AQUA, "Wrong output": YELLOW}
    label_colors = {
        "Correct": "#ffffff",
        "Timeout": "#ffffff",
        "Error": "#ffffff",
        "Wrong output": INK,
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    positions = np.arange(len(ADAPTER_ORDER))
    left = np.zeros(len(ADAPTER_ORDER))
    for name, values in outcomes.items():
        ax.barh(
            positions,
            values,
            left=left,
            height=0.55,
            color=colors[name],
            edgecolor=SURFACE,
            linewidth=2.0,
            label=name,
        )
        for i, value in enumerate(values):
            if value >= 0.05:
                ax.text(
                    left[i] + value / 2,
                    positions[i],
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=label_colors[name],
                )
        left += values.to_numpy()

    ax.set_yticks(positions)
    ax.set_yticklabels([ADAPTER_LABELS[a] for a in ADAPTER_ORDER])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(ADAPTER_ORDER) - 0.4)
    ax.invert_yaxis()  # fast at the top
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of final outcomes")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
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
        ncol=4,
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.3,
        columnspacing=1.4,
    )
    figure_header(
        fig,
        "Failure modes are not interchangeable",
        "Final outcome composition at retry budget 3; 200 runs per adapter.",
    )
    fig.subplots_adjust(left=0.16, right=0.975, top=0.8, bottom=0.15)
    path = FIGURES / "fig3_failure_modes.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary, _ = load()
    figure_exclusion_bias(summary)
    figure_retry_tradeoff(summary)
    figure_failure_modes(summary)


if __name__ == "__main__":
    main()
