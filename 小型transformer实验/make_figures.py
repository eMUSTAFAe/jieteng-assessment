#!/usr/bin/env python3
"""Render the tiny-transformer figures from results/*.csv.

    python make_figures.py

Static light-mode PNGs.  Two series use the first two categorical slots of the
documented, pre-validated palette (blue, orange).  The Node validator could not
be run because Node.js is not installed.
"""

from __future__ import annotations

import json
import math
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

MODEL_ORDER = ["causal", "unmasked"]
MODEL_LABELS = {"causal": "Causal mask", "unmasked": "No mask (unmasked)"}
MODEL_COLORS = {"causal": BLUE, "unmasked": ORANGE}

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


def load_metadata() -> dict:
    with (RESULTS / "metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def figure_training_curves(metadata: dict) -> None:
    curves = pd.read_csv(RESULTS / "training_curves.csv")
    bayes_loss = float(metadata["bayes_loss"])

    fig, (ax_train, ax_val) = plt.subplots(
        2,
        1,
        figsize=(7.6, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    for model in MODEL_ORDER:
        subset = curves[curves["model"] == model]
        for ax, metric in ((ax_train, "train_loss"), (ax_val, "val_loss")):
            grouped = subset.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            mean = grouped["mean"].to_numpy()
            std = grouped["std"].fillna(0.0).to_numpy()
            steps = grouped["step"].to_numpy()
            color = MODEL_COLORS[model]
            ax.plot(
                steps,
                mean,
                color=color,
                linewidth=2.0,
                label=MODEL_LABELS[model],
            )
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    for ax in (ax_train, ax_val):
        ax.axhline(
            bayes_loss,
            color=MUTED,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )
        ax.set_ylim(0.0, 1.95)
        ax.set_ylabel("Loss (nats)", color=SECONDARY)
        style_axes(ax, "Loss (nats)")
    ax_train.text(
        0.99,
        bayes_loss + 0.035,
        f"Bayes loss {bayes_loss:.2f}",
        transform=ax_train.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax_train.set_ylabel("Train loss (nats)", color=SECONDARY)
    ax_val.set_ylabel("Validation loss (nats)", color=SECONDARY)
    ax_val.set_xlabel("Training step")

    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.89),
        ncol=2,
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.5,
        columnspacing=1.8,
    )

    figure_header(
        fig,
        "Training and validation loss",
        "Mean of 3 seeds; shaded ±1 sd; lower is not automatically better.",
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.79, bottom=0.10, hspace=0.18)
    path = FIGURES / "fig1_training_curves.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_heldout(metadata: dict) -> None:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    bayes_loss = float(metadata["bayes_loss"])
    bayes_accuracy = float(metadata["bayes_accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.4))
    panels = [
        ("test_loss", "Held-out loss (nats)", bayes_loss, True),
        ("test_accuracy", "Top-1 accuracy", bayes_accuracy, False),
        ("next_token_attention", "Attention to next token", None, False),
    ]

    for ax, (metric, ylabel, reference, is_loss) in zip(axes, panels):
        means = []
        for position, model in enumerate(MODEL_ORDER):
            values = metrics.loc[metrics["model"] == model, metric].to_numpy()
            mean = float(np.mean(values))
            means.append(mean)
            ax.bar(
                position,
                mean,
                width=0.52,
                color=MODEL_COLORS[model],
                edgecolor=SURFACE,
                linewidth=2.0,
            )
            offsets = np.linspace(-0.09, 0.09, len(values))
            ax.scatter(
                position + offsets,
                values,
                s=26,
                color=INK,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
            ax.text(
                position,
                mean + (0.025 if not is_loss else 0.035),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=SECONDARY,
            )
        if reference is not None:
            ax.axhline(
                reference,
                color=MUTED,
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                zorder=1,
            )
            label = (
                f"Bayes loss {reference:.2f}"
                if is_loss
                else f"Bayes acc. {reference:.2%}"
            )
            ax.text(
                0.5,
                reference - (0.03 if is_loss else 0.02),
                label,
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                color=MUTED,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_ylim(0, 1.75 if is_loss else (1.05 if metric != "next_token_attention" else 0.62))
        style_axes(ax, ylabel)
        if metric == "test_accuracy":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Held-out loss (nats)", color=SECONDARY)
    axes[1].set_ylabel("Top-1 accuracy", color=SECONDARY)

    # Shared legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.89),
        ncol=2,
        frameon=False,
        labelcolor=SECONDARY,
        handlelength=1.3,
        columnspacing=1.8,
    )

    figure_header(
        fig,
        "Held-out result and the leakage diagnostic",
        "Causal reaches the Bayes loss; the unmasked model copies the target from the next input position.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.14, wspace=0.38)
    path = FIGURES / "fig2_heldout_and_leakage.png"
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    figure_training_curves(metadata)
    figure_heldout(metadata)


if __name__ == "__main__":
    main()
