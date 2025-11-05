#!/usr/bin/env python3
"""Generate publication-ready calibration figures from aggregated summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_output.calibration.style import (
    FIG_WIDTH_1COL,
    FIG_WIDTH_2COL,
    apply_publication_style,
)


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "calibration"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot styling helpers
# ---------------------------------------------------------------------------

MODE_STYLE_SEQUENCE = [
    {"color": "#0072B2", "linestyle": "-"},
    {"color": "#D55E00", "linestyle": "--"},
    {"color": "#009E73", "linestyle": "-."},
    {"color": "#CC79A7", "linestyle": ":"},
]

MODE_STYLE_OVERRIDES: Dict[str, Dict[str, object]] = {
    "global": {"color": "#0072B2", "linestyle": "-"},
    "hybrid": {"color": "#D55E00", "linestyle": "--"},
    "hybrid tau5": {"color": "#D55E00", "linestyle": "--"},
    "mondrian": {"color": "#009E73", "linestyle": "-."},
    "mondrian tau5": {"color": "#009E73", "linestyle": "-."},
}


def get_mode_style(mode: str, index: int) -> Dict[str, object]:
    key = mode.lower().replace("_", " ")
    if key in MODE_STYLE_OVERRIDES:
        return MODE_STYLE_OVERRIDES[key]
    base = MODE_STYLE_SEQUENCE[index % len(MODE_STYLE_SEQUENCE)]
    return base


def format_mode_label(mode: str) -> str:
    mode_lower = mode.lower().replace("_", " ")
    if mode_lower.startswith("global"):
        return "Global"
    if mode_lower.startswith("hybrid"):
        return "Hybrid"
    if mode_lower.startswith("mondrian"):
        return "Mondrian"
    return mode.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_alpha_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"dataset", "model", "mode", "alpha", "metric", "mean", "ci_lower", "ci_upper"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Alpha summary missing columns: {missing}")
    return df


def load_repeat_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"dataset", "model", "mode", "metric", "mean", "ci_lower", "ci_upper"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Repeat summary missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_coverage_curves(df: pd.DataFrame,
                         dataset: str,
                         model: str,
                         modes: Iterable[str],
                         column_width: float = FIG_WIDTH_1COL) -> Path:
    subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["metric"] == "coverage")]
    if subset.empty:
        raise ValueError(f"No coverage data for dataset={dataset}, model={model}")

    fig_width = column_width
    fig_height = fig_width * 0.75
    apply_publication_style(fig_width, fig_height / fig_width)
    fig, ax = plt.subplots()

    x_ref = np.linspace(0, 1, 200)
    ax.plot(x_ref, x_ref, linestyle="--", color="0.6", linewidth=1.0, label="Ideal")

    for idx, mode in enumerate(modes):
        mode_df = subset[subset["mode"] == mode]
        if mode_df.empty:
            continue
        style = get_mode_style(mode, idx)
        x = 1.0 - mode_df["alpha"].values
        order = np.argsort(x)
        x_sorted = x[order]
        mean = mode_df["mean"].values[order]
        lower = mode_df["ci_lower"].values[order]
        upper = mode_df["ci_upper"].values[order]
        label = format_mode_label(mode)
        ax.plot(
            x_sorted,
            mean,
            label=label,
            linewidth=1.6,
            **style,
        )
        fill_color = style.get("color", None)
        ax.fill_between(x_sorted, lower, upper, alpha=0.18, color=fill_color)

    ax.set_xlabel("Target coverage (1 - α)")
    ax.set_ylabel("Observed coverage")
    ax.set_xlim(0.5, 1.01)
    ax.set_ylim(0.5, 1.01)
    ax.legend(frameon=False, title="Conformal mode")
    ax.set_title(f"{model} · {dataset}")

    fig.tight_layout()
    out_path = FIGURE_DIR / f"coverage_{dataset}_{model}.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


def plot_interval_sizes(df: pd.DataFrame,
                        dataset: str,
                        model: str,
                        modes: Iterable[str],
                        column_width: float = FIG_WIDTH_1COL) -> Path:
    subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["metric"] == "interval_size")]
    if subset.empty:
        raise ValueError(f"No interval size data for dataset={dataset}, model={model}")

    fig_width = column_width
    fig_height = fig_width * 0.75
    apply_publication_style(fig_width, fig_height / fig_width)
    fig, ax = plt.subplots()

    for idx, mode in enumerate(modes):
        mode_df = subset[subset["mode"] == mode]
        if mode_df.empty:
            continue
        style = get_mode_style(mode, idx)
        x = 1.0 - mode_df["alpha"].values
        order = np.argsort(x)
        x_sorted = x[order]
        mean = mode_df["mean"].values[order]
        lower = mode_df["ci_lower"].values[order]
        upper = mode_df["ci_upper"].values[order]
        label = format_mode_label(mode)
        ax.plot(
            x_sorted,
            mean,
            label=label,
            linewidth=1.6,
            **style,
        )
        fill_color = style.get("color", None)
        ax.fill_between(x_sorted, lower, upper, alpha=0.18, color=fill_color)

    ax.set_xlabel("Target coverage (1 - α)")
    ax.set_ylabel("Average prediction-set size")
    ax.set_xlim(0.5, 1.01)
    ax.legend(frameon=False, title="Conformal mode")
    ax.set_title(f"{model} · {dataset}")

    fig.tight_layout()
    out_path = FIGURE_DIR / f"psize_{dataset}_{model}.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


def plot_repeat_metrics(df: pd.DataFrame,
                        dataset: str,
                        model: str,
                        modes: Iterable[str],
                        metric: str,
                        column_width: float = FIG_WIDTH_1COL) -> Path:
    subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["metric"] == metric)]
    if subset.empty:
        raise ValueError(f"No repeat metric '{metric}' for dataset={dataset}, model={model}")

    fig_width = column_width
    fig_height = fig_width * 0.75
    apply_publication_style(fig_width, fig_height / fig_width)
    fig, ax = plt.subplots()

    x_positions = np.arange(len(list(modes)))
    width = 0.65
    means = []
    lower = []
    upper = []
    labels = []
    for idx, mode in enumerate(modes):
        mode_df = subset[subset["mode"] == mode]
        if mode_df.empty:
            continue
        means.append(mode_df["mean"].values[0])
        lower.append(mode_df["ci_lower"].values[0])
        upper.append(mode_df["ci_upper"].values[0])
        labels.append(format_mode_label(mode))

    colors = [get_mode_style(mode, idx).get("color", "#0072B2") for idx, mode in enumerate(modes) if not subset[subset["mode"] == mode].empty]
    ax.bar(np.arange(len(means)), means, width=width, color=colors, alpha=0.8)
    err_lower = np.array(means) - np.array(lower)
    err_upper = np.array(upper) - np.array(means)
    ax.errorbar(np.arange(len(means)), means, yerr=[err_lower, err_upper], fmt="none", ecolor="0.3", capsize=3)

    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{model} · {dataset}")

    fig.tight_layout()
    out_path = FIGURE_DIR / f"{metric}_{dataset}_{model}.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration plots using aggregated summaries.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., GoEmotions)")
    parser.add_argument("--model", required=True, help="Model short name (e.g., Emollama-7b)")
    parser.add_argument("--column-width", choices=["1col", "2col"], default="1col")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Explicit list of conformal modes to plot (defaults to all available)")
    parser.add_argument("--repeat-metrics", nargs="*", default=["accuracy", "f1_micro", "f1_macro", "pcc", "ace"],
                        help="Repeat-level metrics to render")
    parser.add_argument("--alpha-summary", type=Path,
                        default=OUTPUT_DIR / "calibration_alpha_summary.csv")
    parser.add_argument("--repeat-summary", type=Path,
                        default=OUTPUT_DIR / "calibration_repeat_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    column_width = FIG_WIDTH_1COL if args.column_width == "1col" else FIG_WIDTH_2COL

    alpha_summary = load_alpha_summary(args.alpha_summary)
    repeat_summary = load_repeat_summary(args.repeat_summary)

    if args.modes:
        modes = args.modes
    else:
        modes_alpha = alpha_summary[
            (alpha_summary["dataset"] == args.dataset)
            & (alpha_summary["model"] == args.model)
        ]["mode"].dropna().unique()
        modes_repeat = repeat_summary[
            (repeat_summary["dataset"] == args.dataset)
            & (repeat_summary["model"] == args.model)
        ]["mode"].dropna().unique()
        modes = sorted({*modes_alpha, *modes_repeat})
        if not modes:
            raise ValueError("No modes found for the specified dataset/model. Use --modes to override.")

    coverage_path = plot_coverage_curves(alpha_summary, args.dataset, args.model, modes, column_width)
    size_path = plot_interval_sizes(alpha_summary, args.dataset, args.model, modes, column_width)

    repeat_paths = []
    for metric in args.repeat_metrics:
        try:
            repeat_paths.append(plot_repeat_metrics(repeat_summary, args.dataset, args.model, modes, metric, column_width))
        except ValueError as exc:
            print(exc)

    print("Saved plots:")
    for path in [coverage_path, size_path, *repeat_paths]:
        print(f" - {path}")


if __name__ == "__main__":
    main()
