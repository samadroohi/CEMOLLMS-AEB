
import os
from typing import Optional, Tuple, List, Dict
import numpy as np
from src.config import Config
import matplotlib.pyplot as plt
# Import shared style utilities
import sys
import importlib.util
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
style_path = os.path.join(os.path.dirname(__file__), '../../analysis_output/calibration/style.py')
spec = importlib.util.spec_from_file_location('style', style_path)
style = importlib.util.module_from_spec(spec)
sys.modules['style'] = style
spec.loader.exec_module(style)


def _muted_color(color: str, alpha: float, blend: float = 0.25) -> Tuple[float, float, float, float]:
    """Blend a palette color toward white to keep scatter subdued."""
    rgb = np.array(mcolors.to_rgb(color))
    muted = np.clip(rgb * (1.0 - blend) + blend, 0.0, 1.0)
    return float(muted[0]), float(muted[1]), float(muted[2]), alpha


def _smooth_series(x: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a simple moving-average smoothing of (x, y) pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size < window:
        return x, y
    kernel = np.ones(window, dtype=float) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    x_smooth = np.convolve(x, kernel, mode="valid")
    return x_smooth, y_smooth


def _summarize_residuals(
    preds: np.ndarray,
    residuals: np.ndarray,
    num_bins: int,
    min_count: int,
) -> Dict[str, np.ndarray]:
    preds = np.asarray(preds, dtype=float).ravel()
    residuals = np.asarray(residuals, dtype=float).ravel()
    if preds.size == 0:
        return {"bin_centers": np.array([]), "mean": np.array([]), "std": np.array([])}

    order = np.argsort(preds)
    preds = preds[order]
    residuals = residuals[order]

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(preds, quantiles))
    if edges.size < 2:
        edges = np.array([preds.min(), preds.max()], dtype=float)
        edges[1] = edges[0] + 1e-6

    centers: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    for i in range(edges.size - 1):
        left, right = edges[i], edges[i + 1]
        if i == edges.size - 2:
            mask = (preds >= left) & (preds <= right)
        else:
            mask = (preds >= left) & (preds < right)
        if mask.sum() < min_count:
            continue
        centers.append(float(np.mean(preds[mask])))
        means.append(float(np.mean(residuals[mask])))
        stds.append(float(np.std(residuals[mask], ddof=1)))

    return {
        "bin_centers": np.asarray(centers, dtype=float),
        "mean": np.asarray(means, dtype=float),
        "std": np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0),
    }


def _compute_single_label_bins(results: dict) -> dict:
    probs_arr = np.asarray(results["probs"], dtype=float)
    y_true = np.array([label[1] for label in results["true_values"]], dtype=int)
    if probs_arr.ndim != 2:
        raise ValueError("Expected probability matrix with shape [N, C] for single-label reliability diagram")
    pred_labels = np.argmax(probs_arr, axis=1)
    confidences = probs_arr[np.arange(len(pred_labels)), pred_labels]
    correctness = (pred_labels == y_true).astype(float)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.full(n_bins, np.nan, dtype=float)
    bin_confidences = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)
    total = len(confidences)
    ece_value = 0.0
    mcale = 0.0
    min_plot_count = 10

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = int(mask.sum())
        bin_counts[i] = count
        if count == 0:
            continue
        bin_acc = float(correctness[mask].mean())
        bin_conf = float(confidences[mask].mean())
        diff = abs(bin_acc - bin_conf)
        ece_value += (count / max(1, total)) * diff
        mcale = max(mcale, diff)
        if count >= min_plot_count:
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": float(ece_value),
        "mcale": float(mcale),
        "accuracy": float(np.mean(correctness)),
        "mean_confidence": float(np.mean(confidences)),
        "confidences": confidences,
        "pred_labels": pred_labels,
        "y_true": y_true,
    }


def _compute_multilabel_bins(results: dict, dataset_type: str) -> dict:
    true_values = results["true_values"]
    probs = results["probs"]
    class_labels = list(Config.VALID_D_TYPES[dataset_type].values())
    n_bins = 10
    confidences = []
    f1_scores = []
    subset_coverages = []
    for instance_probs, labels in zip(probs, true_values):
        true_set = set(labels)
        predicted_labels = []
        avg_confidence = 0.0
        for step_probs in instance_probs:
            if len(step_probs) == 0:
                continue
            class_idx = int(np.argmax(step_probs))
            avg_confidence += float(step_probs[class_idx])
            predicted_labels.append(class_labels[class_idx])
        if len(instance_probs) > 0:
            avg_confidence /= len(instance_probs)
        predicted_set = set(predicted_labels)
        if not true_set and not predicted_set:
            f1 = 1.0
        elif not true_set or not predicted_set:
            f1 = 0.0
        else:
            precision = len(true_set & predicted_set) / len(predicted_set) if predicted_set else 0.0
            recall = len(true_set & predicted_set) / len(true_set) if true_set else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        subset_coverages.append(int(true_set.issubset(predicted_set)))
        confidences.append(avg_confidence)
        f1_scores.append(f1)

    confidences = np.asarray(confidences, dtype=float)
    f1_scores = np.asarray(f1_scores, dtype=float)
    subset_coverages = np.asarray(subset_coverages, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.full(n_bins, np.nan, dtype=float)
    bin_confidences = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)
    total = len(confidences)
    ece_value = 0.0
    mcale = 0.0
    min_plot_count = 10

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = int(mask.sum())
        bin_counts[i] = count
        if count == 0:
            continue
        bin_acc = float(f1_scores[mask].mean())
        bin_conf = float(confidences[mask].mean())
        diff = abs(bin_acc - bin_conf)
        ece_value += (count / max(1, total)) * diff
        mcale = max(mcale, diff)
        if count >= min_plot_count:
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": float(ece_value),
        "mcale": float(mcale),
        "mean_confidence": float(np.mean(confidences)) if confidences.size else float("nan"),
        "f1": float(np.mean(f1_scores)) if f1_scores.size else float("nan"),
        "subset_coverage": float(np.mean(subset_coverages)) if subset_coverages.size else float("nan"),
    }


def _compute_regression_bins(results: dict, n_bins: int = 10) -> dict:
    preds = np.asarray(results.get("predictions", []), dtype=float)
    targets = np.asarray(results.get("targets", []), dtype=float)
    if preds.size == 0 or targets.size == 0:
        empty = np.full(n_bins, np.nan, dtype=float)
        return {
            "bin_centers": empty,
            "bin_accuracies": empty,
            "bin_confidences": empty,
            "bin_counts": np.zeros(n_bins, dtype=int),
            "ece": float("nan"),
            "_plot_x_limits": (0.0, 1.0),
            "_plot_y_limits": (0.0, 1.0),
            "_plot_axis_equal": True,
        }

    bin_centers = np.asarray(results.get("bin_centers", []), dtype=float)
    bin_pred_means = np.asarray(results.get("bin_pred_means", []), dtype=float)
    bin_true_means = np.asarray(results.get("bin_true_means", []), dtype=float)
    bin_counts = np.asarray(results.get("bin_counts", []), dtype=float)

    if bin_centers.size == 0 or bin_true_means.size == 0:
        if preds.min() == preds.max():
            span = max(1.0, abs(preds.min())) or 1.0
            edges = np.linspace(preds.min() - span * 0.5, preds.max() + span * 0.5, n_bins + 1)
        else:
            edges = np.linspace(preds.min(), preds.max(), n_bins + 1)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        bin_pred_means = np.full(n_bins, np.nan, dtype=float)
        bin_true_means = np.full(n_bins, np.nan, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (preds >= edges[i]) & (preds <= edges[i + 1])
            else:
                mask = (preds >= edges[i]) & (preds < edges[i + 1])
            count = int(mask.sum())
            bin_counts[i] = count
            if count == 0:
                continue
            bin_pred_means[i] = float(preds[mask].mean())
            bin_true_means[i] = float(targets[mask].mean())

    bin_accuracies = bin_true_means.copy()
    total = bin_counts.sum()
    diffs = np.abs(np.nan_to_num(bin_pred_means - bin_true_means, nan=0.0))
    ece = float(np.dot(bin_counts, diffs) / total) if total > 0 else float("nan")

    x_limits = results.get("plot_x_limits")
    y_limits = results.get("plot_y_limits")
    if x_limits is None:
        x_limits = (float(np.nanmin(bin_pred_means)), float(np.nanmax(bin_pred_means)))
    if y_limits is None:
        y_limits = (float(np.nanmin(bin_true_means)), float(np.nanmax(bin_true_means)))

    return {
        "bin_centers": bin_pred_means,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_pred_means,
        "bin_counts": bin_counts,
        "ece": float(results.get("ece", ece)),
        "_plot_x_limits": tuple(x_limits),
        "_plot_y_limits": tuple(y_limits),
        "_plot_axis_equal": bool(results.get("plot_axis_equal", True)),
    }


def _plot_reliability_comparison(
    entries,
    dataset_type: str,
    title: str,
    output_path: Optional[str],
    stats_fn,
    x_label: str,
    y_label: str,
    fig_width: float = getattr(style, "FIG_WIDTH_1COL", 3.4),
    ax=None,
    add_legend: bool = True,
):
    if not entries:
        return None, ax, ([], [])
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = style.styled_subplots(width=fig_width)
    else:
        fig = ax.figure
    perfect_color = "#d7a0a0"

    colors = style.COLOR_PALETTE
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    marker_size = getattr(style, "MARKER_SIZE", 42)
    marker_edge_width = getattr(style, "MARKER_EDGE_WIDTH", 0.55)
    marker_face_color = getattr(style, "MARKER_FACE_COLOR", None)
    marker_edge_color = getattr(style, "MARKER_EDGE_COLOR", "#1a1a1a")
    marker_alpha = getattr(style, "MARKER_ALPHA", 0.9)
    legend_marker_size = float(np.sqrt(marker_size)) if marker_size and marker_size > 0 else 6.0
    color_map: Dict[str, str] = {}
    scheme_markers: Dict[str, Tuple[str, str]] = {}
    x_limits_accum = [None, None]
    y_limits_accum = [None, None]
    axis_equal_flags: List[bool] = []

    def parse_label(label: str) -> Tuple[str, str]:
        if "·" in label:
            parts = [part.strip() for part in label.split("·", 1)]
            if len(parts) == 2:
                return parts[0], parts[1]
        lower = label.lower()
        scheme = "Identity" if "identity" in lower else ("Platt" if "platt" in lower else "Baseline")
        return label.strip(), scheme

    for entry in entries:
        stats = stats_fn(entry["results"])
        mask = ~np.isnan(stats["bin_accuracies"])
        if not np.any(mask):
            continue

        entry_x_limits = stats.get("_plot_x_limits")
        entry_y_limits = stats.get("_plot_y_limits")
        if entry_x_limits:
            x_limits_accum[0] = entry_x_limits[0] if x_limits_accum[0] is None else min(x_limits_accum[0], entry_x_limits[0])
            x_limits_accum[1] = entry_x_limits[1] if x_limits_accum[1] is None else max(x_limits_accum[1], entry_x_limits[1])
        if entry_y_limits:
            y_limits_accum[0] = entry_y_limits[0] if y_limits_accum[0] is None else min(y_limits_accum[0], entry_y_limits[0])
            y_limits_accum[1] = entry_y_limits[1] if y_limits_accum[1] is None else max(y_limits_accum[1], entry_y_limits[1])
        axis_equal_flags.append(bool(stats.get("_plot_axis_equal", True)))

        model_name, scheme_name = parse_label(entry.get("label", "Model"))
        if model_name not in color_map:
            color_map[model_name] = colors[len(color_map) % len(colors)]
        color = color_map[model_name]

        scheme_key = scheme_name.strip().lower()
        if scheme_key not in scheme_markers:
            marker = marker_cycle[len(scheme_markers) % len(marker_cycle)]
            scheme_markers[scheme_key] = (marker, scheme_name)
        marker, scheme_label = scheme_markers[scheme_key]
        ax.plot(
            stats["bin_centers"][mask],
            stats["bin_accuracies"][mask],
            color=color,
            linewidth=1.4,
            linestyle="-",
        )
        face_color = color if marker_face_color in (None, "", "auto") else marker_face_color
        edge_color = marker_edge_color if marker_edge_color not in (None, "", "auto") else color
        ax.scatter(
            stats["bin_centers"][mask],
            stats["bin_accuracies"][mask],
            s=marker_size,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=marker_edge_width,
            marker=marker,
            alpha=marker_alpha,
            zorder=3,
        )

    x_limits = (0.0, 1.0) if x_limits_accum[0] is None else tuple(x_limits_accum)
    y_limits = (0.0, 1.0) if y_limits_accum[0] is None else tuple(y_limits_accum)
    diag_min = min(x_limits[0], y_limits[0])
    diag_max = max(x_limits[1], y_limits[1])
    reference_handle = Line2D([0], [0], color=perfect_color, linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color=perfect_color, linewidth=1.2)

    if not ax_provided:
        ax.set_title(title or f"Reliability · {dataset_type}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    axis_equal = all(axis_equal_flags) if axis_equal_flags else True
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('auto')
    ax.grid(color="0.92", linewidth=0.6)

    combined_handles = [reference_handle]
    combined_labels = [reference_handle.get_label()]

    for model, color in color_map.items():
        combined_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=1.6))
        combined_labels.append(model)

    for scheme_key, (marker, scheme_label) in scheme_markers.items():
        combined_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markerfacecolor="#d9d9d9",
                markeredgecolor="0.1",
                markeredgewidth=0.6,
                markersize=legend_marker_size * 1.1,
            )
        )
        combined_labels.append(scheme_label)

    legend_handles = combined_handles if add_legend else []
    legend_labels = combined_labels if add_legend else []

    if not ax_provided and add_legend:
        ax.legend(legend_handles, legend_labels, frameon=False, loc="best")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()

    return (None if ax_provided else fig), ax, (legend_handles, legend_labels)


def classification_reliability_comparison(
    entries,
    dataset_type: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    fig_width = style.FIG_WIDTH_2COL if len(entries) > 1 else style.FIG_WIDTH_1COL
    _plot_reliability_comparison(
        entries,
        dataset_type,
        title or f"Reliability · {dataset_type}",
        output_path,
        stats_fn=_compute_single_label_bins,
        x_label="Confidence",
        y_label="Observed Frequency",
        fig_width=fig_width,
    )


def multiclass_classification_reliability_comparison(
    entries,
    dataset_type: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    def stats_fn(results):
        return _compute_multilabel_bins(results, dataset_type)

    fig_width = style.FIG_WIDTH_2COL if len(entries) > 1 else style.FIG_WIDTH_1COL
    _plot_reliability_comparison(
        entries,
        dataset_type,
        title or f"Reliability · {dataset_type}",
        output_path,
        stats_fn=stats_fn,
        x_label="Confidence",
        y_label="F1 Score",
        fig_width=fig_width,
    )


def plot_task_comparison_panel(
    task_name: str,
    dataset_infos: List[Dict[str, object]],
    output_path: str,
) -> None:
    if not dataset_infos:
        return
    if dataset_infos and dataset_infos[0].get("task_type") == "regression_tasks":
        plot_residual_panel(task_name, dataset_infos, output_path)
        return
    n = len(dataset_infos)
    base_width = getattr(style, "CLASS_COL_WIDTH", style.FIG_WIDTH_1COL)
    width = max(base_width * n, style.FIG_WIDTH_2COL if n > 1 else base_width)
    height = width * style.GOLDEN_RATIO
    fig, axes = style.styled_subplots(width=width, height=height, ncols=n, squeeze=False)
    axes = axes.flatten()
    for idx, (ax, info) in enumerate(zip(axes, dataset_infos)):
        entries = info.get("entries", [])
        dataset = info.get("dataset", "")
        multilabel = bool(info.get("multilabel", False))
        task_type = info.get("task_type", "")
        if task_type == "multiclass_classification" or multilabel:
            stats_fn = lambda res, ds=dataset: _compute_multilabel_bins(res, ds)
            x_label = "Confidence"
            y_label = "F1 Score"
        elif task_type == "regression_tasks":
            stats_fn = _compute_regression_bins
            x_label = "Predicted"
            y_label = "Observed"
        else:
            stats_fn = _compute_single_label_bins
            x_label = "Confidence"
            y_label = "Accuracy"
        _, _, (handles, labels) = _plot_reliability_comparison(
            entries,
            dataset,
            title=dataset,
            output_path=None,
            stats_fn=stats_fn,
            x_label=x_label,
            y_label=y_label,
            fig_width=width / max(1, n),
            ax=ax,
            add_legend=(idx == 0),
        )
        if dataset:
            ax.text(
                0.5,
                1.05,
                dataset,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=style.FONT_SIZE_TITLE,
                fontweight="semibold",
            )
        if handles and labels and idx == 0:
            legend_font = getattr(style, "CLASS_LEGEND_FONT", getattr(style, "FONT_SIZE_LEGEND", 5.0))
            ax.legend(
                handles,
                labels,
                frameon=True,
                loc="upper left",
                fontsize=legend_font,
                handlelength=1.2,
                handletextpad=0.4,
                borderpad=0.25,
            )
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()
        if idx == 0:
            ylabel_offset = getattr(style, "CLASS_YLABEL_OFFSET", -0.08)
            ax.yaxis.set_label_coords(ylabel_offset, 0.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # Hide unused axes if dataset count < n (shouldn't happen but safe)
    for ax in axes[len(dataset_infos):]:
        ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(
        wspace=getattr(style, "CLASS_PANEL_WSPACE", 0.08),
        hspace=getattr(style, "CLASS_PANEL_HSPACE", 0.18),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_panel(
    task_name: str,
    dataset_infos: List[Dict[str, object]],
    output_path: str,
) -> None:
    if not dataset_infos:
        return

    scheme_order = ["identity", "isotonic"]
    available_schemes = {
        (entry.get("scheme") or "identity")
        for info in dataset_infos
        for entry in info.get("entries", [])
    }
    schemes = [s for s in scheme_order if s in available_schemes]
    if not schemes:
        schemes = list(available_schemes) or ["identity"]

    n_cols = len(dataset_infos)
    n_rows = len(schemes)
    base_width = getattr(style, "RESIDUAL_COL_WIDTH", style.FIG_WIDTH_1COL)
    width = max(base_width * n_cols, style.FIG_WIDTH_2COL)
    row_height = getattr(style, "RESIDUAL_ROW_HEIGHT", style.FIG_WIDTH_1COL * style.GOLDEN_RATIO)
    height = row_height * n_rows
    fig, axes = style.styled_subplots(
        width=width,
        height=height,
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
    )
    colors = style.COLOR_PALETTE
    scheme_titles = {"identity": "Identity", "isotonic": "Isotonic"}
    show_trends = bool(getattr(style, "RESIDUAL_SHOW_TRENDS", False))
    params = {
        "agg_max_points": getattr(style, "RESIDUAL_TREND_MAX_POINTS", 250),
        "smooth_window": getattr(style, "RESIDUAL_TREND_SMOOTHING", 19),
        "num_bins": getattr(style, "RESIDUAL_NUM_BINS", 30),
        "min_bin_count": getattr(style, "RESIDUAL_MIN_BIN_COUNT", 15),
        "band_alpha": getattr(style, "RESIDUAL_BAND_ALPHA", 0.18),
        "mean_alpha": getattr(style, "RESIDUAL_MEAN_LINE_ALPHA", 0.45),
    }

    for col, info in enumerate(dataset_infos):
        dataset = info.get("dataset", "")
        entries = info.get("entries", []) or []
        by_scheme: Dict[str, List[Dict[str, object]]] = {}
        for entry in entries:
            scheme = entry.get("scheme") or "identity"
            by_scheme.setdefault(scheme, []).append(entry)
        model_colors: Dict[str, str] = {}

        for row, scheme in enumerate(schemes):
            ax = axes[row, col]
            subset = by_scheme.get(scheme, [])
            handles = _plot_residual_axis(
                ax=ax,
                dataset=dataset or task_name,
                entries=subset,
                scheme_key=scheme,
                scheme_title=scheme_titles.get(scheme, scheme.title()),
                color_map=model_colors,
                colors=colors,
                show_trends=show_trends,
                show_xlabel=(row == len(schemes) - 1),
                show_ylabel=(col == 0),
                show_title=(row == 0),
                **params,
            )
            if handles and col == 0:
                legend_font = getattr(style, "FONT_SIZE_LEGEND", 5.0)
                ax.legend(
                    handles,
                    [h.get_label() for h in handles],
                    frameon=True,
                    loc="upper right",
                    fontsize=legend_font,
                    handlelength=1.2,
                    handletextpad=0.4,
                    borderpad=0.25,
                )
            if col == 0:
                row_label_x = getattr(style, "RESIDUAL_ROW_LABEL_X", -0.2)
                ax.text(
                    row_label_x,
                    0.5,
                    scheme_titles.get(scheme, scheme.title()),
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    rotation=90,
                    fontsize=style.FONT_SIZE_TITLE + 1,
                    fontweight="semibold",
                )

    fig.tight_layout()
    fig.subplots_adjust(
        wspace=getattr(style, "RESIDUAL_WSPACE", 0.18),
        hspace=getattr(style, "RESIDUAL_HSPACE", 0.35),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_axis(
    ax: plt.Axes,
    dataset: str,
    entries: List[Dict[str, object]],
    scheme_key: str,
    scheme_title: str,
    color_map: Dict[str, str],
    colors: List[str],
    show_trends: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    agg_max_points: int,
    smooth_window: int,
    num_bins: int,
    min_bin_count: int,
    band_alpha: float,
    mean_alpha: float,
) -> List[Line2D]:
    ax.axhline(0.0, linestyle=(0, (2, 2)), color="#b2182b", linewidth=1.0, alpha=0.85)
    ax.set_facecolor("#ffffff")
    if not entries:
        ax.set_visible(False)
        return []

    res_abs_max = 0.0
    model_handles: Dict[str, Line2D] = {}
    aggregate: Dict[str, List[np.ndarray]] = {"preds": [], "res": []} if show_trends else None

    for entry in entries:
        payload = entry.get("results", {})
        preds = np.asarray(payload.get("predictions", []), dtype=float)
        residuals = np.asarray(payload.get("residuals", []), dtype=float)
        if preds.size == 0 or residuals.size == 0:
            continue
        model_name = entry.get("model") or entry.get("label", "Model")
        entry_label = entry.get("label") or model_name
        if model_name not in color_map:
            color_map[model_name] = colors[len(color_map) % len(colors)]
        color = color_map[model_name]

        summary = _summarize_residuals(preds, residuals, num_bins, min_bin_count)
        if summary["bin_centers"].size == 0:
            continue
        upper = summary["mean"] + summary["std"]
        lower = summary["mean"] - summary["std"]
        ax.fill_between(
            summary["bin_centers"],
            lower,
            upper,
            color=_muted_color(color, band_alpha),
            edgecolor="none",
        )
        ax.plot(
            summary["bin_centers"],
            summary["mean"],
            color=color,
            linewidth=0.9,
            alpha=mean_alpha,
        )
        res_abs_max = max(
            res_abs_max,
            float(
                np.max(
                    np.abs(
                        np.concatenate(
                            [np.asarray(residuals, dtype=float), lower, upper]
                        )
                    )
                )
            ),
        )
        if model_name not in model_handles:
            model_handles[model_name] = Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.0,
                label=entry_label,
            )
        if show_trends and aggregate is not None:
            aggregate["preds"].append(preds)
            aggregate["res"].append(residuals)

    scheme_handles: List[Line2D] = []
    if show_trends and aggregate is not None and aggregate["preds"]:
        preds_all = np.concatenate(aggregate["preds"])
        res_all = np.concatenate(aggregate["res"])
        summary = _summarize_residuals(
            preds_all,
            res_all,
            max(num_bins, int(num_bins * 1.5)),
            min_bin_count,
        )
        if summary["bin_centers"].size > 0:
            centers = summary["bin_centers"]
            mean_vals = summary["mean"]
            if agg_max_points and centers.size > agg_max_points:
                idx = np.linspace(0, centers.size - 1, int(agg_max_points)).astype(int)
                centers = centers[idx]
                mean_vals = mean_vals[idx]
            if smooth_window and smooth_window > 1:
                centers, mean_vals = _smooth_series(centers, mean_vals, smooth_window)
            linestyle = "-" if scheme_key == "identity" else "--"
            marker = "o" if scheme_key == "identity" else "s"
            highlight_color = "#1a1a1a" if scheme_key == "identity" else "#666666"
            line_kwargs = {
                "color": highlight_color,
                "linestyle": linestyle,
                "linewidth": 2.3,
                "alpha": 0.95,
            }
            if marker:
                line_kwargs.update(
                    {
                        "marker": marker,
                        "markersize": 4.5,
                        "markevery": max(1, centers.size // 8),
                        "markerfacecolor": "white",
                        "markeredgewidth": 0.8,
                    }
                )
            ax.plot(centers, mean_vals, **line_kwargs)
            scheme_handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    color=highlight_color,
                    linewidth=2.3,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    label=f"{scheme_title} trend",
                )
            )
            res_abs_max = max(res_abs_max, float(np.max(np.abs(mean_vals))))

    valid_range = Config.VALID_D_TYPES.get(dataset) if dataset else None
    if isinstance(valid_range, dict) and "min" in valid_range and "max" in valid_range:
        x_min = float(valid_range["min"])
        x_max = float(valid_range["max"])
        ax.set_xlim(x_min, x_max)
        range_text = f"{dataset} [{x_min}, {x_max}]"
    else:
        range_text = dataset
    if res_abs_max > 0:
        ax.set_ylim(-res_abs_max * 1.05, res_abs_max * 1.05)
    ax.set_box_aspect(1)
    if show_xlabel:
        ax.set_xlabel("Predicted")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Residual (True - Predicted)")
        ylabel_offset = getattr(style, "RESIDUAL_YLABEL_OFFSET", -0.13)
        ax.yaxis.set_label_coords(ylabel_offset, 0.5)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if show_title:
        ax.set_title(range_text)
    else:
        ax.set_title("")
    ax.grid(color="0.92", linewidth=0.6)

    return list(model_handles.values()) + scheme_handles


def regression_calibration_diagram(results: dict,
                                  ds_type: str,
                                  alpha: float,
                                 output_dir: str = None,
                                 title: str = None,
                                 figsize: tuple = (10, 6)):
    """
    Create a calibration diagram for regression predictions showing prediction intervals
    and whether true values fall within them.
    
    Args:
        results (dict): Dictionary containing conformal prediction results
        ds_type (str): Dataset type
        alpha (float): Significance level (alpha value)
        output_dir (str, optional): Directory to save the plot
        title (str, optional): Custom title for the plot
        figsize (tuple, optional): Figure size (width, height)
    """
    try:
        # Extract and convert data from results to float arrays
        y_true = np.array([float(x) for x in results["true_values"]], dtype=np.float64)
        y_pred = np.array([float(x) for x in results["predictions"]], dtype=np.float64)
        
        # Handle prediction sets that come as separate lower and upper bounds
        if isinstance(results["prediction_sets"], list) and len(results["prediction_sets"]) == 2:
            lower_bounds = np.array([float(x) for x in results["prediction_sets"][0]], dtype=np.float64)
            upper_bounds = np.array([float(x) for x in results["prediction_sets"][1]], dtype=np.float64)
        else:
            raise ValueError("Prediction sets must be a list containing [lower_bounds, upper_bounds]")
        
        # Sort all arrays by predicted values for better visualization
        sort_idx = np.argsort(y_pred)
        y_pred = y_pred[sort_idx]
        y_true = y_true[sort_idx]
        lower_bounds = lower_bounds[sort_idx]
        upper_bounds = upper_bounds[sort_idx]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot prediction intervals
        x_range = np.arange(len(y_pred))
        plt.fill_between(x_range, lower_bounds, upper_bounds, 
                        alpha=0.2, color='blue', label='Prediction interval')
        
        # Plot regression line
        z = np.polyfit(x_range, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(x_range, p(x_range), 'orange', label='Regression line')
        
        # Plot points with colors based on whether true value is in interval
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        
        # Plot points inside interval
        plt.scatter(x_range[in_interval], y_true[in_interval], 
                   color='green', marker='o', label='Within interval', alpha=0.6)
        
        # Plot points outside interval
        plt.scatter(x_range[~in_interval], y_true[~in_interval], 
                   color='red', marker='x', label='Outside interval', alpha=0.6)
        
        # Calculate coverage
        coverage = np.mean(in_interval)
        interval_sizes = upper_bounds - lower_bounds
        avg_interval_size = np.mean(interval_sizes)
        
        # Add title and labels
        if title is None:
            title = f'Regression Calibration ({ds_type}, α={alpha})\n' \
                    f'Coverage: {coverage:.3f}, Avg Interval Size: {avg_interval_size:.3f}'
        plt.title(title)
        plt.xlabel('Instance Index (sorted by predicted value)')
        plt.ylabel('Target Value')
        
        # Add legend
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'calibration_plot_{ds_type}_alpha_{alpha}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        data ={
            'coverage': coverage,
            'avg_interval_size': avg_interval_size,
            'in_interval_count': np.sum(in_interval),
            'total_points': len(y_true)
        }
        return data
    
    except Exception as e:
        print(f"Error in regression_calibration_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"predictions type: {type(results['predictions'][0])}")
        print(f"prediction_sets type: {type(results['prediction_sets'][0][0])}")
        raise

def classification_relibaility_diagram(results: dict,
                                        dataset_type: str,
                                        output_dir: str = None,
                                        title: str = None,
                                        figsize: tuple = (10, 6),
                                        output_path: Optional[str] = None):
    """
    Legacy compatibility wrapper that now delegates to classification_reliability_comparison.
    """
    try:
        stats = _compute_single_label_bins(results)
        save_path = output_path
        if save_path is None and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"reliability_plot_{dataset_type}.png")

        classification_reliability_comparison(
            [{"label": title or "Model", "results": results}],
            dataset_type,
            title=title,
            output_path=save_path,
        )

        if results.get("predictions"):
            pred_indices_list = [pred[1] for pred in results["predictions"]]
            if len(pred_indices_list) != len(stats["y_true"]):
                pred_indices_array = stats["pred_labels"]
            else:
                pred_indices_array = np.array(pred_indices_list, dtype=int)
        else:
            pred_indices_array = stats["pred_labels"]
        y_true_one_hot = (pred_indices_array == stats["y_true"]).astype(float)
        brier_score = float(np.mean((stats["confidences"] - y_true_one_hot) ** 2))

        return {
            "accuracy": float(stats["accuracy"]),
            "ece": float(stats["ece"]),
            "mcale": float(stats["mcale"]),
            "brier_score": brier_score,
            "mean_prediction": float(stats["mean_confidence"]),
        }
    except Exception as e:
        print(f"Error in classification_relibaility_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise

def multiclass_classification_relibaility_diagram(results: dict,
                                      dataset_type: str,
                                      output_dir: str = None,
                                      title: str = None,
                                      figsize: tuple = (10, 6),
                                      output_path: Optional[str] = None):
    """
    Legacy wrapper that now reuses the comparison plotting utilities while returning summary metrics.
    """
    try:
        stats = _compute_multilabel_bins(results, dataset_type)
        save_path = output_path
        if save_path is None and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"reliability_plot_{dataset_type}.png")

        multiclass_classification_reliability_comparison(
            [{"label": title or "Model", "results": results}],
            dataset_type,
            title=title,
            output_path=save_path,
        )

        return {
            "f1": float(stats["f1"]),
            "ece": float(stats["ece"]),
            "mcale": float(stats["mcale"]),
            "mean_confidence": float(stats["mean_confidence"]),
            "subset_coverage": float(stats["subset_coverage"]),
        }
    except Exception as e:
        raise Exception(f"Error in multiclass_classification_relibaility_diagram: {str(e)}")


def cp_diagrams(results,dataset_type, output_dir):
    plot_confidence_vs_coverage(results, dataset_type, output_dir)
    cp_results = plot_coverage_vs_prediction_set_size(results, dataset_type, output_dir)
    return cp_results


    


def plot_confidence_vs_coverage(results, dataset_type, output_dir=None):
    """
    Plots confidence vs empirical coverage and annotates the plot with the ACE (Average Coverage Error).
    
    Args:
        results (dict): Dictionary containing conformal prediction results.
        dataset_type (str): Dataset type.
        alphas (list): A list of confidence levels to evaluate.
        output_dir (str, optional): Directory to save the plot.
    """
    coverage = []
    ace = 0
    mcove = 0
    alphas = Config.CP_ALPHA
    for alpha in alphas:
        cov = results[str(alpha)]["coverage"]
        coverage.append(cov)
        ace += abs(cov - (1-alpha))
        mcove = max(mcove, abs(cov - (1-alpha)))
    ace /= len(alphas)
    confidence_values = [1 - alpha for alpha in alphas]
    
    plt.figure(figsize=(10, 6))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', alpha=0.5)
    
    # Plot confidence vs coverage
    plt.plot(confidence_values, coverage, 'b-', label='Model calibration')
    plt.scatter(confidence_values, coverage, c='blue')
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.title(f'Confidence vs Empirical Coverage ({dataset_type})')
    plt.xlabel('Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Add annotation for ACE
    plt.text(0.05, 0.95, f'ACE: {ace:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
    plt.text(0.05, 0.90, f'MCovE: {mcove:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'confidence_vs_coverage_{dataset_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
 

def plot_coverage_vs_prediction_set_size(results, dataset_type, output_dir=None):
    """
    Plots prediction set size vs confidence level (1 - alpha) with circles for each point,
    connects the points with a line, and annotates the coverage for each point.
    
    Args:
        results (dict): Dictionary containing conformal prediction results.
        dataset_type (str): Dataset type.
        output_dir (str, optional): Directory to save the plot.
    """
    alphas = Config.CP_ALPHA
    plt.figure(figsize=(8, 6))
    confidences = [1 - alpha for alpha in alphas]
    set_sizes = []
    coverages = []
    cp_metrics = {}
    mcove = 0
    for alpha in alphas:
        result = results[str(alpha)]
        sizes = []
        coverage = result["coverage"]
        mcove = max(mcove, abs(coverage - (1-alpha)))
        for pred_set in result["prediction_sets"]:
            sizes.append(len(pred_set))
        avg_size = np.mean(sizes)
        set_sizes.append(avg_size)
        coverages.append(coverage)
        plt.scatter(1 - alpha, avg_size, c='b', marker='o')
        plt.annotate(f"{coverage:.2f}", (1 - alpha, avg_size), xytext=(5, 5), textcoords="offset points")
    ace = np.mean(np.abs(np.array(coverages) - np.array(confidences)))
    plt.plot(confidences, set_sizes, 'b-')

    plt.xlabel('Confidence Level (1 - alpha)')
    plt.ylabel('Prediction Set Size')
    plt.title(f'Prediction Set Size vs Confidence Level ({dataset_type})')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'size_vs_confidence_{dataset_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    cp_metrics = {
        'alpha': alphas,
        'coverage': coverages,
        'psize': set_sizes,
        'ace': ace,
        'mcove': mcove
    }
    return cp_metrics
        


    
    return metrics
def compute_regression_metrics(y_true, y_pred, model=None, X=None):
    """
    Compute standard regression metrics including uncertainty estimates.
    
    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        model: Trained model (optional) - used to estimate prediction uncertainty
        X: Feature data corresponding to y_true (optional) - needed for uncertainty estimation
        
    Returns:
        dict: Dictionary containing regression metrics
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate basic metrics
    metrics = {
        'mse': np.mean(residuals**2),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals))
    }

    return metrics
def calibration_anlaysis(results, ds_type, output_dir=None):
    # Initialize all metrics dictionaries
    calibration_metrics = {}
    cp_metrics = {}
    top_p_metrics = {}

    if ds_type in Config.TASK_TYPES["ordinal_classification"]:
        calibration_metrics = classification_relibaility_diagram(results[str(Config.CP_ALPHA[0])], ds_type, output_dir=output_dir)
        cp_metrics = cp_diagrams(results, ds_type, output_dir)
        top_p_metrics = top_p_analysis(results[str(Config.CP_ALPHA[0])], ds_type, output_dir)
    elif ds_type in Config.TASK_TYPES["regression_tasks"]:
        cp_metrics = {'alpha':[],  'coverages': [], 'average_interval_sizes': []}
        for alpha in Config.CP_ALPHA:
            cp_data = regression_calibration_diagram(results[str(alpha)], ds_type, alpha, output_dir=output_dir)
            cp_metrics['alpha'].append(alpha)
            cp_metrics['coverages'].append(cp_data['coverage'])
            cp_metrics['average_interval_sizes'].append(cp_data['avg_interval_size'])
            
        calibration_metrics = compute_regression_metrics(results[str(Config.CP_ALPHA[0])]["true_values"], results[str(Config.CP_ALPHA[0])]["predictions"])
        # For regression, top_p_metrics remains empty
    elif ds_type in Config.TASK_TYPES["multiclass_classification"]:
        calibration_metrics = multiclass_classification_relibaility_diagram(results[str(Config.CP_ALPHA[0])], ds_type, output_dir)
        cp_metrics = cp_diagrams(results, ds_type, output_dir)
        top_p_metrics = multiclass_top_p_analysis(results[str(Config.CP_ALPHA[0])], ds_type, output_dir)
    return calibration_metrics, cp_metrics, top_p_metrics

def top_p_analysis(results: dict, ds_type: str, output_dir: str = None):
    """
    Analyze prediction sets based on accumulating probabilities until reaching target confidence levels.
    For ordinal classification, includes all classes between min and max in prediction set.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        output_dir (str, optional): Directory to save plots
        
    Returns:
        dict: Dictionary containing top-p analysis metrics
    """
    try:
        # Extract true values and probabilities
        y_true = np.array([label[1] for label in results["true_values"]])
        probs = np.array(results["probs"])
        
        # Initialize metrics
        confidences = []
        coverages = []
        set_sizes = []
        mcove = 0
        
        # For each confidence level (1 - alpha)
        for alpha in Config.CP_ALPHA:
            target_confidence = 1 - alpha
            prediction_sets = []
            coverage = 0
            
            # For each example
            for i in range(len(y_true)):
                # Sort probabilities in descending order
                sorted_probs = np.sort(probs[i])[::-1]
                sorted_indices = np.argsort(probs[i])[::-1]
                
                # Accumulate probabilities until reaching target confidence
                cumulative_prob = 0
                prediction_set = []
                
                for j in range(len(sorted_probs)):
                    cumulative_prob += sorted_probs[j]
                    prediction_set.append(sorted_indices[j])
                    if cumulative_prob >= target_confidence:
                        break
                
                # For ordinal classification, include all classes between min and max
                if len(prediction_set) > 1:
                    min_pred_set = min(prediction_set)
                    max_pred_set = max(prediction_set)
                    prediction_set = list(range(min_pred_set, max_pred_set + 1))
                
                prediction_sets.append(prediction_set)
                
                # Check if true label is in prediction set
                if y_true[i] in prediction_set:
                    coverage += 1
            
            # Calculate metrics
            coverage_rate = coverage / len(y_true)
            avg_set_size = np.mean([len(ps) for ps in prediction_sets])
            
            confidences.append(target_confidence)
            coverages.append(coverage_rate)
            set_sizes.append(avg_set_size)
            
            # Update maximum coverage error
            mcove = max(mcove, abs(coverage_rate - target_confidence))
        
        # Calculate average coverage error
        ace = np.mean(np.abs(np.array(coverages) - np.array(confidences)))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot prediction set size vs confidence
        plt.plot(confidences, set_sizes, 'b-', label='Prediction set size')
        plt.scatter(confidences, set_sizes, c='blue')
        
        # Add coverage annotations
        for i, (conf, size) in enumerate(zip(confidences, set_sizes)):
            plt.annotate(f"{coverages[i]:.2f}", (conf, size), xytext=(5, 5), textcoords="offset points")
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.title(f'Top-p Analysis ({ds_type})')
        plt.xlabel('Confidence Level (1 - alpha)')
        plt.ylabel('Average Prediction Set Size')
        plt.legend()
        
        # Add metrics annotations
        plt.text(0.05, 0.95, f'ACE: {ace:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        plt.text(0.05, 0.90, f'MCovE: {mcove:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'top_p_analysis_{ds_type}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        # Return metrics
        metrics = {
            'alpha': [1 - conf for conf in confidences],
            'coverage': coverages,
            'psize': set_sizes,
            'ace': ace,
            'mcove': mcove
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error in top_p_analysis: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise

def multiclass_top_p_analysis(results: dict, ds_type: str, output_dir: str = None):
    """
    Analyze prediction sets for multiclass classification using top-p approach.
    Uses max logit for each class, applies softmax, then follows top-p procedure.
    Coverage is calculated as the fraction of examples where all true labels are in the prediction set.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        output_dir (str, optional): Directory to save plots
        
    Returns:
        dict: Dictionary containing top-p analysis metrics
    """
    try:
        # Extract true values and probabilities
        y_true = results["true_values"]
        probs = results["probs"]
        
        # Get emotion labels mapping
        emotion_labels = list(Config.VALID_D_TYPES[ds_type].values())
        
        # Initialize metrics
        confidences = []
        coverages = []
        set_sizes = []
        mcove = 0
        
        # For each confidence level (1 - alpha)
        for alpha in Config.CP_ALPHA:
            target_confidence = 1 - alpha
            prediction_sets = []
            coverage = 0
            total_examples = 0
            
            # For each example
            for i in range(len(y_true)):
                # Convert true labels to a set of strings
                true_labels = set(str(label) if not isinstance(label, str) else label for label in y_true[i])
                max_logits = []
                # Get max logit for each class
                if len(probs[i])>0:
                    # Get max logit for each position across all class probability vectors
                    max_logits = [max(prob_vec[j] for prob_vec in probs[i]) for j in range(len(probs[i][0]))]
                else:
                    max_logits.append(float('-inf'))  # Use negative infinity for empty lists

                # Convert to numpy array and check if it's empty
                max_logits = np.array(max_logits)
                if len(max_logits) == 0 or np.all(max_logits == float('-inf')):
                    continue  # Skip this example if no valid logits
                
                # Apply softmax to get probabilities
                try:
                    #exp_logits = np.exp(max_logits - np.max(max_logits))  # For numerical stability
                    probs_softmax = max_logits / max_logits.sum()
                except (ValueError, RuntimeWarning):
                    # If there's an issue with the logits, use uniform probabilities
                    probs_softmax = np.ones_like(max_logits) / len(max_logits)
                
                # Sort probabilities in descending order
                sorted_probs = np.sort(probs_softmax)[::-1]
                sorted_indices = np.argsort(probs_softmax)[::-1]
                
                # Accumulate probabilities until reaching target confidence
                cumulative_prob = 0
                prediction_set = []
                
                for j in range(len(sorted_probs)):
                    cumulative_prob += sorted_probs[j]
                    # Map index to emotion label
                    pred_label = str(emotion_labels[sorted_indices[j]])
                    prediction_set.append(pred_label)
                    if cumulative_prob >= target_confidence:
                        break
                
                prediction_sets.append(prediction_set)
                
                # Convert prediction set to set of strings for comparison
                pred_set = set(prediction_set)
                
                # Debug prints for first few examples
                if i < 5:  # Print first 5 examples
                    print(f"\nExample {i}:")
                    print(f"True labels: {true_labels}")
                    print(f"Prediction set: {pred_set}")
                    print(f"Cumulative probability: {cumulative_prob}")
                    print(f"Is subset: {true_labels.issubset(pred_set)}")
                
                # Check if true labels are a subset of prediction set
                if true_labels.issubset(pred_set):
                    coverage += 1
                total_examples += 1
            
            # Calculate metrics
            if total_examples > 0:  # Only calculate metrics if we have valid examples
                coverage_rate = coverage / total_examples
                avg_set_size = np.mean([len(ps) for ps in prediction_sets])
                
                confidences.append(target_confidence)
                coverages.append(coverage_rate)
                set_sizes.append(avg_set_size)
                
                # Update maximum coverage error
                mcove = max(mcove, abs(coverage_rate - target_confidence))
                
                # Debug print for this confidence level
                print(f"\nConfidence level {target_confidence}:")
                print(f"Coverage rate: {coverage_rate}")
                print(f"Average set size: {avg_set_size}")
                print(f"Total coverage: {coverage} out of {total_examples}")
        
        # Calculate average coverage error
        if len(coverages) > 0:  # Only calculate if we have valid coverages
            ace = np.mean(np.abs(np.array(coverages) - np.array(confidences)))
        else:
            ace = 0
            mcove = 0
            confidences = [1 - alpha for alpha in Config.CP_ALPHA]
            coverages = [0] * len(Config.CP_ALPHA)
            set_sizes = [0] * len(Config.CP_ALPHA)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot prediction set size vs confidence
        plt.plot(confidences, set_sizes, 'b-', label='Prediction set size')
        plt.scatter(confidences, set_sizes, c='blue')
        
        # Add coverage annotations
        for i, (conf, size) in enumerate(zip(confidences, set_sizes)):
            plt.annotate(f"{coverages[i]:.2f}", (conf, size), xytext=(5, 5), textcoords="offset points")
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.title(f'Top-p Analysis ({ds_type})')
        plt.xlabel('Confidence Level (1 - alpha)')
        plt.ylabel('Average Prediction Set Size')
        plt.legend()
        
        # Add metrics annotations
        plt.text(0.05, 0.95, f'ACE: {ace:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        plt.text(0.05, 0.90, f'MCovE: {mcove:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'top_p_analysis_{ds_type}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        # Return metrics
        metrics = {
            'alpha': [1 - conf for conf in confidences],
            'coverage': coverages,
            'psize': set_sizes,
            'ace': ace,
            'mcove': mcove
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error in multiclass_top_p_analysis: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise
