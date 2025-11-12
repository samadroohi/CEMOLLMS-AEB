
import os
from typing import Optional, Tuple, List, Dict
import numpy as np
from src.config import Config
import matplotlib.pyplot as plt
# Import shared style utilities
import sys
import importlib.util
from matplotlib.lines import Line2D
style_path = os.path.join(os.path.dirname(__file__), '../../analysis_output/calibration/style.py')
spec = importlib.util.spec_from_file_location('style', style_path)
style = importlib.util.module_from_spec(spec)
sys.modules['style'] = style
spec.loader.exec_module(style)


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
):
    if not entries:
        return None, ax, ([], [])
    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = style.styled_subplots(width=fig_width)
    else:
        fig = ax.figure
    perfect_color = "#d7a0a0"
    reference_handle = Line2D([0], [0], color=perfect_color, linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color=perfect_color, linewidth=1.2)

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

    if not ax_provided:
        ax.set_title(title or f"Reliability · {dataset_type}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
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

    legend_handles = combined_handles
    legend_labels = combined_labels

    if not ax_provided:
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
    n = len(dataset_infos)
    width = max(style.FIG_WIDTH_1COL * n, style.FIG_WIDTH_2COL)
    height = width * style.GOLDEN_RATIO
    fig, axes = style.styled_subplots(width=width, height=height, ncols=n, squeeze=False)
    axes = axes.flatten()
    for ax, info in zip(axes, dataset_infos):
        entries = info.get("entries", [])
        dataset = info.get("dataset", "")
        multilabel = bool(info.get("multilabel", False))
        stats_fn = (lambda res, ds=dataset: _compute_multilabel_bins(res, ds)) if multilabel else _compute_single_label_bins
        x_label = "Confidence"
        y_label = "F1 Score" if multilabel else "Accuracy"
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
        if handles and labels:
            legend_font = getattr(style, "FONT_SIZE_LEGEND", 5.0)
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

    # Hide unused axes if dataset count < n (shouldn't happen but safe)
    for ax in axes[len(dataset_infos):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    elif ds_type in Config.TASK_TYPES["regression"]:
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
