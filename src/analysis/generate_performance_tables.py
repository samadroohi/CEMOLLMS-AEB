#!/usr/bin/env python3
"""Generate summary tables for predictive performance and calibration metrics.

This script aggregates per-repeat statistics (F1, PCC, ECE, Brier) across
multiple conformal prediction runs and exports both CSV and Markdown tables.

Currently, multiclass multilabel datasets (e.g., GoEmotions, E-c) are fully
supported. Hooks for ordinal classification and regression are in place so the
script can be extended when those results are added later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from .analysis_utils import get_performance_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFORMAL_DIR = REPO_ROOT / "results" / "conformal_results"
DEFAULT_TEMP_DIR = "temp_0.9"
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "calibration"
DEFAULT_CSV = OUTPUT_DIR / "performance_summary.csv"
DEFAULT_MD = OUTPUT_DIR / "performance_summary.md"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def infer_task_type(dataset: str) -> Optional[str]:
    """Return the high-level task type configured for a dataset name."""

    priority = (
        "multiclass_classification",
        "ordinal_classification",
        "classification",
        "regression",
    )

    for task in priority:
        if dataset in Config.TASK_TYPES.get(task, []):
            return task

    # Fall back to the first matching entry if the dataset sits in an auxiliary
    # collection (e.g. weighted_regression shares members with regression).
    for task, datasets in Config.TASK_TYPES.items():
        if dataset in datasets:
            return task

    return None


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean())


def aggregate_metric(values: Iterable[Optional[float]], confidence: float = 0.95) -> Optional[Dict[str, float]]:
    filtered = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if filtered.size == 0:
        return None

    n = filtered.size
    mean = float(filtered.mean())
    std = float(filtered.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        sem = std / math.sqrt(n)
        t_score = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci = float(sem * t_score)
    else:
        ci = 0.0

    return {
        "mean": mean,
        "std": std,
        "ci": ci,
        "n": int(n),
    }


def format_stat_pm(summary: Optional[Dict[str, float]]) -> str:
    """Return human-readable 'mean +/- std' string for table cells."""

    if not summary:
        return "n/a"

    mean = summary["mean"]
    std = summary["std"]
    return f"{mean:.2f} +/- {std:.3f}"


def _ensure_list(obj: Sequence[str] | str | None) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    return [str(item) for item in obj]


def compute_multilabel_calibration_metrics(
    true_values: Sequence[Sequence[str]] | Sequence[str],
    probs: Sequence[Sequence[Sequence[float]]],
    class_labels: Sequence[str],
    n_bins: int = 10,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (ECE, Brier) for multilabel multiclass predictions."""

    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    confidences: List[float] = []
    jaccard_scores: List[float] = []
    brier_terms: List[float] = []

    for labels, prob_seq in zip(true_values, probs):
        prob_seq = list(prob_seq) if prob_seq is not None else []
        if not prob_seq:
            continue

        arr = np.asarray(prob_seq, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != len(class_labels):
            continue

        # Brier score (mean squared error over classes).
        agg_probs = arr.mean(axis=0)
        true_vec = np.zeros(len(class_labels), dtype=float)
        for label in _ensure_list(labels):
            idx = class_to_idx.get(label)
            if idx is not None:
                true_vec[idx] = 1.0
        brier_terms.append(float(np.mean((agg_probs - true_vec) ** 2)))

        # Confidence and empirical accuracy (Jaccard) for ECE style binning.
        predicted_labels: List[str] = []
        conf_sum = 0.0
        valid_steps = 0
        for step_probs in arr:
            if step_probs.size == 0:
                continue
            best_idx = int(np.argmax(step_probs))
            predicted_labels.append(class_labels[best_idx])
            conf_sum += float(step_probs[best_idx])
            valid_steps += 1

        if valid_steps == 0:
            continue

        avg_conf = conf_sum / valid_steps
        confidences.append(avg_conf)

        true_set = set(label for label in _ensure_list(labels) if label in class_to_idx)
        pred_set = set(label for label in predicted_labels if label in class_to_idx)

        if not true_set and not pred_set:
            jaccard = 1.0
        elif not true_set or not pred_set:
            jaccard = 0.0
        else:
            jaccard = len(true_set & pred_set) / len(true_set | pred_set)
        jaccard_scores.append(jaccard)

    ece = _ece_from_bins(confidences, jaccard_scores, n_bins)
    brier = _safe_mean(brier_terms)
    return ece, brier


def _ece_from_bins(confidences: Sequence[float], accuracies: Sequence[float], n_bins: int) -> Optional[float]:
    if not confidences:
        return None

    conf_arr = np.asarray(confidences, dtype=float)
    acc_arr = np.asarray(accuracies, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    total = 0
    weighted_error = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf_arr >= bin_edges[i]) & (conf_arr <= bin_edges[i + 1])
        else:
            mask = (conf_arr >= bin_edges[i]) & (conf_arr < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(conf_arr[mask].mean())
        bin_acc = float(acc_arr[mask].mean())
        weighted_error += count * abs(bin_acc - bin_conf)
        total += count

    if total == 0:
        return None
    return float(weighted_error / total)


def compute_brier_and_ece(dataset: str, record: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Dispatch calibration metric computation based on dataset type."""

    task_type = infer_task_type(dataset)
    true_values = record.get("true_values", [])
    probs = record.get("probs", [])

    if task_type == "multiclass_classification":
        class_labels = list(Config.VALID_D_TYPES[dataset].values())
        ece, brier = compute_multilabel_calibration_metrics(true_values, probs, class_labels)
        return brier, ece

    # Placeholder branches for future tasks – returning None keeps the cell as n/a.
    return None, None


def gather_repeat_metrics(dataset: str, record: Dict) -> Dict[str, Optional[float]]:
    """Compute per-repeat metrics from a single conformal record."""

    payload = {
        "ds_type": dataset,
        "true_values": record.get("true_values", []),
        "predictions": record.get("predictions", []),
        "probs": record.get("probs", []),
        "prediction_sets": record.get("prediction_sets", []),
    }

    try:
        performance = get_performance_metrics(payload, dataset, Config.TASK_TYPES)
    except Exception:  # noqa: BLE001
        performance = {}

    task_type = infer_task_type(dataset)
    metrics: Dict[str, Optional[float]] = {
        "f1_micro": None,
        "f1_macro": None,
        "pcc": None,
    }

    if task_type == "multiclass_classification":
        metrics["f1_micro"] = performance.get("f1_micro")
        metrics["f1_macro"] = performance.get("f1_macro")
    elif task_type == "ordinal_classification":
        metrics["f1_micro"] = performance.get("micro_f1")
        metrics["f1_macro"] = performance.get("macro_f1")
        metrics["pcc"] = performance.get("average_pearson")
    elif task_type == "regression":
        metrics["pcc"] = performance.get("pearson_correlation")

    brier, ece = compute_brier_and_ece(dataset, record)
    metrics["brier"] = brier
    metrics["ece"] = ece

    return metrics


def collect_metrics(
    conformal_dir: Path,
    temp_dir: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for dataset_dir in sorted(conformal_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        temp_path = dataset_dir / temp_dir
        if not temp_path.exists():
            continue

        processed_models: Set[Tuple[str, str]] = set()

        for json_path in sorted(temp_path.glob("*.json")):
            model_name = json_path.stem.split("__", 1)[0]
            key = (dataset, model_name)
            if key in processed_models:
                # Skip additional conformal modes for the same model.
                continue
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)

            repeat_records: Dict[int, Dict] = {}
            for record in payload.get("results", []):
                repeat_idx = int(record.get("repeat_index", 0))
                existing = repeat_records.get(repeat_idx)
                if existing is None:
                    repeat_records[repeat_idx] = record
                else:
                    try:
                        current_alpha = float(existing.get("alpha", 1.0))
                        new_alpha = float(record.get("alpha", 1.0))
                    except (TypeError, ValueError):
                        current_alpha = 1.0
                        new_alpha = 1.0
                    if new_alpha < current_alpha:
                        repeat_records[repeat_idx] = record

            if not repeat_records:
                continue

            metric_lists: Dict[str, List[Optional[float]]] = {
                "f1_micro": [],
                "f1_macro": [],
                "pcc": [],
                "ece": [],
                "brier": [],
            }

            for repeat_idx in sorted(repeat_records):
                metrics = gather_repeat_metrics(dataset, repeat_records[repeat_idx])
                for key in metric_lists:
                    metric_lists[key].append(metrics.get(key))

            row: Dict[str, object] = {
                "dataset": dataset,
                "model": model_name,
            }

            for metric_name, values in metric_lists.items():
                summary = aggregate_metric(values)
                row[metric_name] = format_stat_pm(summary)

            rows.append(row)
            processed_models.add(key)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["dataset", "model"], keep="first")
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = [
        "| " + " | ".join(str(row[col]) for col in headers) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header_row, separator, *body_rows])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance summary tables")
    parser.add_argument("--conformal-dir", type=Path, default=DEFAULT_CONFORMAL_DIR)
    parser.add_argument("--temperature", default=DEFAULT_TEMP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    parser.add_argument("--skip-markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_metrics(args.conformal_dir, args.temperature)

    if df.empty:
        print("No conformal metrics were found; nothing to export.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved performance summary table to {args.output}")

    if not args.skip_markdown:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(dataframe_to_markdown(df), encoding="utf-8")
        print(f"Saved Markdown summary to {args.markdown_output}")


if __name__ == "__main__":
    main()
