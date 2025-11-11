#!/usr/bin/env python3
"""Create merged coverage figures per dataset across all models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from analysis_output.calibration.style import (
    FIG_WIDTH_2COL,
    COVERAGE_Y_RANGE,
    apply_publication_style,
)

from .calibration_plots import get_mode_style, format_mode_label

_MODEL_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "H", "*"]


def _get_model_marker_map(models: Sequence[str]) -> dict[str, str]:
    markers: dict[str, str] = {}
    for idx, model in enumerate(models):
        markers[model] = _MODEL_MARKERS[idx % len(_MODEL_MARKERS)]
    return markers


def _prepare_metric_dataframe(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)].copy()
    if subset.empty:
        return subset
    subset["confidence"] = 1.0 - subset["alpha"].astype(float)
    subset = subset[(subset["confidence"] >= 0.5) & (subset["confidence"] <= 0.9)]
    return subset


def _plot_integrated_series(
    ax: plt.Axes,
    subset: pd.DataFrame,
    models: Sequence[str],
    modes: Sequence[str],
    marker_map: dict[str, str],
    mode_styles: dict[str, dict[str, object]],
    y_label: str,
    title: str | None,
    include_reference: bool,
    show_xlabel: bool,
) -> None:
    for model in models:
        for mode in modes:
            mode_df = subset[(subset["model"] == model) & (subset["mode"] == mode)]
            if mode_df.empty:
                continue
            style = mode_styles[mode]
            color = style.get("color", "#1f77b4")
            points = mode_df.sort_values("confidence")
            x_vals = points["confidence"].values
            y_vals = points["mean"].values
            y_err_low = y_vals - points["ci_lower"].values
            y_err_high = points["ci_upper"].values - y_vals
            y_err = np.vstack([y_err_low, y_err_high])
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                fmt="none",
                ecolor="#1a1a1a",
                elinewidth=0.75,
                capsize=2.2,
                alpha=0.75,
                zorder=1,
            )
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                x_vals,
                y_vals,
                marker=marker_map[model],
                s=38,
                facecolor=color,
                edgecolor="0.1",
                linewidth=0.45,
                alpha=0.9,
                zorder=3,
            )

    if include_reference and not subset.empty:
        conf_min = subset["confidence"].min()
        conf_max = subset["confidence"].max()
        ax.plot([conf_min, conf_max], [conf_min, conf_max], color="0.6", linestyle="--", linewidth=0.85)

    confidence_levels = sorted(subset["confidence"].unique()) if not subset.empty else []
    if confidence_levels:
        ax.set_xticks(confidence_levels)
        ax.set_xticklabels([f"{level:.1f}" for level in confidence_levels])
        xmin = max(0.0, min(confidence_levels) - 0.02)
        xmax = min(1.0, max(confidence_levels) + 0.02)
        ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    if show_xlabel:
        ax.set_xlabel("Confidence (1 - α)", fontsize=8)
        ax.tick_params(labelbottom=True)
    else:
        ax.set_xlabel("", fontsize=8)
        ax.tick_params(labelbottom=False)

    ax.set_ylabel(y_label, fontsize=8)
    if title:
        ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.grid(color="0.85", linewidth=0.5, alpha=1.0)
    ax.set_axisbelow(True)


def _create_mode_handles(mode_styles: dict[str, dict[str, object]], include_reference: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    for mode, style in mode_styles.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "#1f77b4"),
                linestyle="-",
                linewidth=1.0,
                alpha=0.85,
                label=format_mode_label(mode),
            )
        )
    if include_reference:
        handles.append(Line2D([0], [0], color="0.6", linestyle="--", linewidth=1.0, label="Ideal"))
    return handles


def _create_model_handles(marker_map: dict[str, str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for model, marker in marker_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                color="0.2",
                markersize=5,
                label=model,
            )
        )
    return handles


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "calibration"
MERGED_DIR = OUTPUT_DIR / "figures_merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)


def _load_alpha_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "model", "mode", "alpha", "metric", "mean", "ci_lower", "ci_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Alpha summary missing columns: {missing}")
    df["mode"] = df["mode"].fillna("default")
    return df


def _sorted_modes(modes: Iterable[str]) -> Sequence[str]:
    order = {"global": 0, "hybrid": 1, "mondrian": 2, "default": 3}
    return sorted(modes, key=lambda m: order.get(m.lower(), 9))


def _plot_model_ax(ax, df: pd.DataFrame, model: str, dataset: str) -> None:
    subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["metric"] == "coverage")]
    if subset.empty:
        ax.set_visible(False)
        return

    modes = _sorted_modes(subset["mode"].unique())
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
        ax.plot(x_sorted, mean, linewidth=1.6, label=label, **style)
        color = style.get("color")
        ax.fill_between(x_sorted, lower, upper, alpha=0.18, color=color)

    x_ref = np.linspace(0, 1, 200)
    ax.plot(x_ref, x_ref, linestyle="--", color="0.6", linewidth=1.0)

    ax.set_title(model, fontsize=9)
    ax.set_xlim(0.5, 1.01)
    ax.set_ylim(*COVERAGE_Y_RANGE)
    ax.grid(True, alpha=0.25)


def _plot_model_size(ax, df: pd.DataFrame, model: str, dataset: str, ylim: tuple[float, float]) -> None:
    subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["metric"] == "interval_size")]
    if subset.empty:
        ax.set_visible(False)
        return

    modes = _sorted_modes(subset["mode"].unique())
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
        ax.plot(x_sorted, mean, linewidth=1.6, label=label, **style)
        color = style.get("color")
        ax.fill_between(x_sorted, lower, upper, alpha=0.18, color=color)

    ax.set_title(model, fontsize=8)
    ax.set_xlim(0.5, 1.01)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def _legend_handles(axes: Sequence[plt.Axes]) -> tuple[list, list]:
    """Collect legend handles from visible axes."""
    for ax in axes:
        if not ax.get_visible():
            continue
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles = list(handles)
            labels = list(labels)
            return handles, labels
    return [], []


def _build_coverage_figure(dataset: str,
                           df: pd.DataFrame,
                           models: Sequence[str],
                           columns: int,
                           total_width: float,
                           base_ratio: float = 0.75) -> Path:
    ncols = max(1, columns)
    cells_needed = len(models) + 1  # reserve one for legend
    nrows = math.ceil(cells_needed / ncols)

    cell_width = total_width / ncols
    cell_height = cell_width * base_ratio
    total_height = cell_height * nrows

    apply_publication_style(total_width, total_height / total_width)
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_width, total_height), sharex=True, sharey=True)
    axes_flat = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        _plot_model_ax(ax, df, model, dataset)
        if not ax.get_visible():
            continue
        ax.tick_params(axis="both", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Empirical coverage", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Target coverage (1 - α)", fontsize=8)

    # Legend axis
    legend_index = len(models)
    if legend_index < len(axes_flat):
        legend_ax = axes_flat[legend_index]
        handles, labels = _legend_handles(axes_flat[:len(models)])
        legend_ax.axis("off")
        if handles:
            handles = list(handles)
            labels = list(labels)
            handles.append(Line2D([0], [0], color="0.6", linestyle="--", linewidth=1.0))
            labels.append("Ideal")
            legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=6)
            
        else:
            legend_ax.legend([Line2D([0], [0], color="0.6", linestyle="--", linewidth=1.0)],
                             ["Ideal"], loc="center", frameon=False, fontsize=6)
            
        legend_index += 1

    for ax in axes_flat[legend_index:]:
        ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.12)

    fig.text(
        0.04,
        0.5,
        dataset,
        transform=fig.transFigure,
        ha="left",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=9,
    )

    out_dir = MERGED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coverage_merged_{dataset}.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out_path


def _build_size_figure(dataset: str,
                       df: pd.DataFrame,
                       models: Sequence[str],
                       columns: int,
                       total_width: float,
                       base_ratio: float = 0.75) -> Path:
    ncols = max(1, columns)
    cells_needed = len(models) + 1
    nrows = math.ceil(cells_needed / ncols)

    cell_width = total_width / ncols
    cell_height = cell_width * base_ratio
    total_height = cell_height * nrows

    dataset_df = df[(df["dataset"] == dataset) & (df["metric"] == "interval_size")]
    values = dataset_df["mean"].to_numpy()
    if values.size:
        span = values.max() - values.min()
        pad = 0.05 * (span if span > 0 else max(values.max(), 1.0))
        y_min = max(0.0, values.min() - pad)
        y_max = values.max() + pad
    else:
        y_min, y_max = 0.0, 1.0
    ylim = (y_min, y_max)

    apply_publication_style(total_width, total_height / total_width)
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_width, total_height), sharex=True, sharey=True)
    axes_flat = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        _plot_model_size(ax, df, model, dataset, ylim)
        if not ax.get_visible():
            continue
        ax.tick_params(axis="both", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Prediction set size", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Target coverage (1 - α)", fontsize=8)

    legend_index = len(models)
    if legend_index < len(axes_flat):
        legend_ax = axes_flat[legend_index]
        handles, labels = _legend_handles(axes_flat[:len(models)])
        legend_ax.axis("off")
        if handles:
            legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=6)
        legend_index += 1

    for ax in axes_flat[legend_index:]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.12)

    fig.text(
        0.04,
        0.5,
        dataset,
        transform=fig.transFigure,
        ha="left",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=9,
    )

    out_dir = MERGED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"size_merged_{dataset}.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out_path


def _build_integrated_metric_figure(dataset: str,
                                    df: pd.DataFrame,
                                    metric: str,
                                    y_label: str,
                                    filename_stub: str,
                                    include_reference: bool = False,
                                    total_width: float = FIG_WIDTH_2COL,
                                    base_ratio: float = 0.6) -> Path | None:
    subset = _prepare_metric_dataframe(df, dataset, metric)
    if subset.empty:
        print(f"Skipping {dataset}: no records for {metric}.")
        return None

    models = sorted(subset["model"].unique())
    modes = _sorted_modes(subset["mode"].unique())
    marker_map = _get_model_marker_map(models)
    mode_styles = {mode: get_mode_style(mode, idx) for idx, mode in enumerate(modes)}

    apply_publication_style(total_width, base_ratio)
    fig, ax = plt.subplots(figsize=(total_width, total_width * base_ratio))

    for model in models:
        for mode in modes:
            mode_df = subset[(subset["model"] == model) & (subset["mode"] == mode)]
            if mode_df.empty:
                continue
            style = mode_styles[mode]
            color = style.get("color", "#1f77b4")
            linestyle = style.get("linestyle", "-")
            points = mode_df.sort_values("confidence")
            x_vals = points["confidence"].values
            y_vals = points["mean"].values
            y_err_low = y_vals - points["ci_lower"].values
            y_err_high = points["ci_upper"].values - y_vals
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=[y_err_low, y_err_high],
                fmt="none",
                ecolor="#1a1a1a",
                elinewidth=0.75,
                capsize=2.2,
                alpha=0.75,
                zorder=1,
            )
            ax.plot(
                x_vals,
                y_vals,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
            ax.scatter(
                x_vals,
                y_vals,
                marker=marker_map[model],
                s=44,
                facecolor=color,
                edgecolor="0.1",
                linewidth=0.45,
                alpha=0.9,
                zorder=3,
            )

    if include_reference:
        conf_min, conf_max = subset["confidence"].min(), subset["confidence"].max()
    ax.plot([conf_min, conf_max], [conf_min, conf_max], color="0.6", linestyle="--", linewidth=0.85)

    confidence_levels = sorted(subset["confidence"].unique())
    ax.set_xticks(confidence_levels)
    ax.set_xticklabels([f"{level:.1f}" for level in confidence_levels])
    xmin = max(0.0, min(confidence_levels) - 0.02)
    xmax = min(1.0, max(confidence_levels) + 0.02)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    ax.set_xlabel("Confidence (1 - α)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(f"{y_label} vs Confidence · {dataset}")

    mode_handles = [
        Line2D(
            [0],
            [0],
            color=mode_styles[mode].get("color", "#1f77b4"),
            linestyle=mode_styles[mode].get("linestyle", "-"),
            linewidth=1.0,
            alpha=0.85,
            label=format_mode_label(mode),
        )
        for mode in modes
    ]
    if include_reference:
        mode_handles.append(Line2D([0], [0], color="0.6", linestyle="--", linewidth=1.0, label="Ideal"))

    mode_legend = ax.legend(
        handles=mode_handles,
        loc="upper left",
        frameon=False,
        fontsize=6,
    )
    ax.add_artist(mode_legend)

    model_handles = [
        Line2D([0], [0], marker=marker_map[model], linestyle="", color="0.2", markersize=5, label=model)
        for model in models
    ]
    ax.legend(
        handles=model_handles,
        title="Model",
        loc="lower right",
        frameon=False,
        fontsize=6,
        title_fontsize=6,
    )

    out_dir = MERGED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_stub}_{dataset}.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out_path


def _build_integrated_joint_figure(dataset: str,
                                   df: pd.DataFrame,
                                   total_width: float = FIG_WIDTH_2COL,
                                   base_ratio: float = 0.55) -> Path | None:
    coverage_subset = _prepare_metric_dataframe(df, dataset, "coverage")
    size_subset = _prepare_metric_dataframe(df, dataset, "interval_size")

    if coverage_subset.empty or size_subset.empty:
        print(f"Skipping {dataset}: missing data for joint integrated plot.")
        return None

    models = sorted({*coverage_subset["model"].unique(), *size_subset["model"].unique()})
    modes = _sorted_modes({*coverage_subset["mode"].unique(), *size_subset["mode"].unique()})
    marker_map = _get_model_marker_map(models)
    mode_styles = {mode: get_mode_style(mode, idx) for idx, mode in enumerate(modes)}

    apply_publication_style(total_width, base_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(total_width, total_width * base_ratio), sharex=True)
    coverage_ax, size_ax = axes

    _plot_integrated_series(
        ax=coverage_ax,
        subset=coverage_subset,
        models=models,
        modes=modes,
        marker_map=marker_map,
        mode_styles=mode_styles,
        y_label="Empirical coverage",
        title=None,
        include_reference=True,
        show_xlabel=True,
    )

    _plot_integrated_series(
        ax=size_ax,
        subset=size_subset,
        models=models,
        modes=modes,
        marker_map=marker_map,
        mode_styles=mode_styles,
        y_label="Prediction set size",
        title=None,
        include_reference=False,
        show_xlabel=True,
    )

    coverage_ax.set_box_aspect(1)
    size_ax.set_box_aspect(1)

    mode_handles = _create_mode_handles(mode_styles, include_reference=True)
    model_handles = _create_model_handles(marker_map)

    mode_legend = coverage_ax.legend(
        mode_handles,
        [handle.get_label() for handle in mode_handles],
        loc="upper left",
        bbox_to_anchor=(0.03, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        fontsize=6,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    mode_legend.get_frame().set_edgecolor("0.7")
    coverage_ax.add_artist(mode_legend)

    model_legend = coverage_ax.legend(
        model_handles,
        [handle.get_label() for handle in model_handles],
        loc="lower right",
        bbox_to_anchor=(0.97, 0.02),
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        fontsize=6,
        title_fontsize=6,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    model_legend.get_frame().set_edgecolor("0.7")
    coverage_ax.add_artist(model_legend)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.11, wspace=0.3)

    fig.text(
        0.04,
        0.5,
        dataset,
        transform=fig.transFigure,
        ha="left",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        fontsize=9,
    )

    out_dir = MERGED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"integrated_joint_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def _generate_integrated_joint(dataset: str,
                               df: pd.DataFrame,
                               width: float) -> Path | None:
    return _build_integrated_joint_figure(dataset, df, width)


def _generate_dataset_coverage(dataset: str,
                               df: pd.DataFrame,
                               columns: int,
                               width: float) -> Path | None:
    dataset_df = df[(df["dataset"] == dataset) & (df["metric"] == "coverage")]
    if dataset_df.empty:
        print(f"Skipping {dataset}: no coverage records found.")
        return None
    models = sorted(dataset_df["model"].unique())
    if not models:
        print(f"Skipping {dataset}: no models found.")
        return None
    return _build_coverage_figure(dataset, df, models, columns, width)


def _generate_dataset_size(dataset: str,
                           df: pd.DataFrame,
                           columns: int,
                           width: float) -> Path | None:
    dataset_df = df[(df["dataset"] == dataset) & (df["metric"] == "interval_size")]
    if dataset_df.empty:
        print(f"Skipping {dataset}: no interval size records found.")
        return None
    models = sorted(dataset_df["model"].unique())
    if not models:
        print(f"Skipping {dataset}: no models found for interval size.")
        return None
    return _build_size_figure(dataset, df, models, columns, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate merged coverage figures for datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to include")
    parser.add_argument("--columns", type=int, default=3, help="Number of subplot columns")
    parser.add_argument("--width", type=float, default=FIG_WIDTH_2COL, help="Total figure width in inches")
    parser.add_argument("--alpha-summary", type=Path,
                        default=OUTPUT_DIR / "calibration_alpha_summary.csv",
                        help="Path to calibration alpha summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_summary = _load_alpha_summary(args.alpha_summary)

    for dataset in args.datasets:
        coverage_path = _generate_dataset_coverage(dataset, alpha_summary, args.columns, args.width)
        if coverage_path:
            print(f"Saved coverage figure for {dataset}: {coverage_path}")
        size_path = _generate_dataset_size(dataset, alpha_summary, args.columns, args.width)
        if size_path:
            print(f"Saved size figure for {dataset}: {size_path}")
        joint_path = _generate_integrated_joint(dataset, alpha_summary, args.width)
        if joint_path:
            print(f"Saved joint integrated figure for {dataset}: {joint_path}")


if __name__ == "__main__":
    main()
