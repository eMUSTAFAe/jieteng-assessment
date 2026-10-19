#!/usr/bin/env python3
"""Render the three report figures from the experiment's result tables.

Run after cost_models_experiment.py:

    python make_figures.py

The figures are static light-mode PNGs for a written report.  The two series use
the first two slots of a documented, pre-validated categorical palette
(blue #2a78d6, orange #eb6834); the skill's Node validator could not be run in
this environment because Node.js is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
LR_COLOR = "#2a78d6"   # categorical slot 1, blue
DT_COLOR = "#eb6834"   # categorical slot 2, orange

MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "DecisionTree": "Decision tree",
}
MODEL_COLORS = {
    "LogisticRegression": LR_COLOR,
    "DecisionTree": DT_COLOR,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
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


def load_json() -> dict:
    with (RESULTS / "main_results.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def figure_dev_cost_curves(summary: dict) -> None:
    curves = pd.read_csv(RESULTS / "dev_cost_curves.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    for model in ("LogisticRegression", "DecisionTree"):
        subset = curves[curves["model"] == model]
        ax.plot(
            subset["threshold"],
            subset["cost"],
            color=MODEL_COLORS[model],
            linewidth=2.0,
            label=MODEL_LABELS[model],
        )

    lr = summary["main_split"]["LogisticRegression"]
    lr_fin_t = lr["final_threshold"]
    lr_fin_c = lr["final_dev_cost"]

    ax.axvline(
        lr_fin_t,
        color=MUTED,
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    ax.plot(
        [lr_fin_t],
        [lr_fin_c],
        marker="o",
        markersize=6,
        color=LR_COLOR,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        zorder=3,
    )
    ax.text(
        lr_fin_t + 0.012,
        0.985,
        f"recommended: threshold {lr_fin_t:.3f}\ncost {lr_fin_c:.3f}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color=SECONDARY,
        linespacing=1.4,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.44, 1.22)
    ax.set_xlabel("Decision threshold: P(bad credit risk)")
    style_axes(ax, "Dev expected cost per applicant")
    ax.legend(
        loc="lower right",
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.6,
        borderaxespad=0.8,
    )

    figure_header(
        fig,
        "Dev expected cost by decision threshold",
        "German Credit; FN:FP = 4:1; lower is better. Test data is not shown.",
    )
    fig.subplots_adjust(left=0.095, right=0.975, top=0.83, bottom=0.14)
    path = FIGURES / "fig1_dev_cost_curves.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_cost_ratio() -> None:
    table = pd.read_csv(RESULTS / "cost_ratio_sensitivity.csv")
    ratios = sorted(table["cost_ratio_fn_to_fp"].unique())
    positions = np.arange(len(ratios))
    labels = [f"{int(r)}:1" for r in ratios]

    fig, (ax_cost, ax_threshold) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    for model in ("LogisticRegression", "DecisionTree"):
        subset = (
            table[table["model"] == model]
            .set_index("cost_ratio_fn_to_fp")
            .loc[ratios]
        )
        color = MODEL_COLORS[model]
        ax_cost.plot(
            positions,
            subset["test_cost"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            label=MODEL_LABELS[model],
        )
        ax_threshold.plot(
            positions,
            subset["threshold"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )

    supplied_x = positions[ratios.index(4.0)]
    for ax in (ax_cost, ax_threshold):
        ax.axvline(
            supplied_x,
            color=MUTED,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
    ax_cost.text(
        supplied_x + 0.08,
        0.985,
        "supplied 1:4 table",
        transform=ax_cost.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )

    # Direct end labels in ink, with the colored line carrying identity.
    lr_cost = table[table["model"] == "LogisticRegression"].set_index(
        "cost_ratio_fn_to_fp"
    )
    dt_cost = table[table["model"] == "DecisionTree"].set_index(
        "cost_ratio_fn_to_fp"
    )
    ax_cost.annotate(
        f"{lr_cost.loc[ratios[-1], 'test_cost']:.2f}",
        xy=(positions[-1], lr_cost.loc[ratios[-1], "test_cost"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )
    ax_cost.annotate(
        f"{dt_cost.loc[ratios[-1], 'test_cost']:.2f}",
        xy=(positions[-1], dt_cost.loc[ratios[-1], "test_cost"]),
        xytext=(8, -6),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )

    ax_cost.set_ylabel("Test expected cost per applicant", color=SECONDARY)
    ax_threshold.set_ylabel("Dev-selected threshold", color=SECONDARY)
    ax_threshold.set_xlabel("Assumed FN:FP cost ratio")
    ax_threshold.set_xticks(positions)
    ax_threshold.set_xticklabels(labels)
    style_axes(ax_cost, "Test expected cost per applicant")
    style_axes(ax_threshold, "Dev-selected threshold")
    ax_cost.legend(
        loc="upper left",
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.6,
        borderaxespad=0.8,
    )

    figure_header(
        fig,
        "Sensitivity to the assumed FN:FP cost ratio",
        "Thresholds re-selected on dev for each ratio; test used only for reporting.",
    )
    fig.subplots_adjust(left=0.13, right=0.975, top=0.84, bottom=0.09, hspace=0.18)
    path = FIGURES / "fig2_cost_ratio_sensitivity.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_fp_harm() -> None:
    table = pd.read_csv(RESULTS / "fp_harm_sensitivity.csv")
    added = sorted(table["added_fp_cost_s"].unique())
    positions = np.arange(len(added))
    labels = ["0\n(supplied)", "+0.5", "+1", "+2", "+4", "+8"]

    fig, (ax_fp, ax_cost) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    for model in ("LogisticRegression", "DecisionTree"):
        subset = (
            table[table["model"] == model].set_index("added_fp_cost_s").loc[added]
        )
        color = MODEL_COLORS[model]
        ax_fp.plot(
            positions,
            subset["test_fp_per_1000"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            label=MODEL_LABELS[model],
        )
        ax_cost.plot(
            positions,
            subset["test_business_cost"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )

    lr = table[table["model"] == "LogisticRegression"].set_index("added_fp_cost_s")
    dt = table[table["model"] == "DecisionTree"].set_index("added_fp_cost_s")
    ax_fp.annotate(
        f"{lr.loc[added[-1], 'test_fp_per_1000']:.0f}",
        xy=(positions[-1], lr.loc[added[-1], "test_fp_per_1000"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )
    ax_fp.annotate(
        "0",
        xy=(positions[-1], dt.loc[added[-1], "test_fp_per_1000"]),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )
    ax_cost.annotate(
        f"{lr.loc[added[-1], 'test_business_cost']:.2f}",
        xy=(positions[-1], lr.loc[added[-1], "test_business_cost"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )
    ax_cost.annotate(
        f"{dt.loc[added[-1], 'test_business_cost']:.2f}",
        xy=(positions[-1], dt.loc[added[-1], "test_business_cost"]),
        xytext=(8, -6),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY,
        va="center",
    )

    ax_fp.set_ylabel("Test false positives per 1,000", color=SECONDARY)
    ax_cost.set_ylabel("Test business cost per applicant", color=SECONDARY)
    ax_cost.set_xlabel("Unmeasured FP harm added to C_FP (s)")
    ax_fp.set_xticks(positions)
    ax_fp.set_xticklabels([])
    ax_cost.set_xticks(positions)
    ax_cost.set_xticklabels(labels)
    style_axes(ax_fp, "Test false positives per 1,000")
    style_axes(ax_cost, "Test business cost per applicant")
    ax_fp.legend(
        loc="upper right",
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.6,
        borderaxespad=0.8,
    )

    figure_header(
        fig,
        "Sensitivity to an unmeasured false-positive harm",
        "Effective C_FP = 1 + s while C_FN = 4; thresholds re-selected on dev.",
    )
    fig.subplots_adjust(left=0.14, right=0.975, top=0.84, bottom=0.1, hspace=0.18)
    path = FIGURES / "fig3_fp_harm_sensitivity.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_json()
    figure_dev_cost_curves(summary)
    figure_cost_ratio()
    figure_fp_harm()


if __name__ == "__main__":
    main()
