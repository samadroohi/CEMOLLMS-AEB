from __future__ import annotations
import os
import sys
import importlib.util
vis_path = os.path.join(os.path.dirname(__file__), 'visualization_utils.py')
spec = importlib.util.spec_from_file_location('visualization_utils', vis_path)
vis = importlib.util.module_from_spec(spec)
sys.modules['visualization_utils'] = vis
spec.loader.exec_module(vis)
"""Utility to fit Platt scaling baselines for saved probability dumps.

This module reads per-example probability vectors stored during inference,
fits lightweight post-hoc calibrators using the calibration split, and reports
baseline metrics on the held-out test portion. It currently supports ordinal
classification tasks (single-label) and multilabel multiclass tasks (GoEmotions,
E-c) without mutating the existing conformal prediction pipeline.
"""



import argparse
import csv
import json
import re
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from src.config import Config
from .src.utils import cleaning_results, convert_to_serializable
from .src.analysis.generate_performance_tables import compute_multilabel_calibration_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "calibration" / "baselines"
EPS = 1e-12

def _slug(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


DEFAULT_METRICS_CSV = OUTPUT_DIR / "baseline_metrics.csv"


def _infer_task_type(dataset: str) -> str:
    for task, ds_list in Config.TASK_TYPES.items():
        if dataset in ds_list:
            return task
    return "unknown"


def _aggregate_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {"num_repeats": len(metric_dicts)}
    if not metric_dicts:
        return agg
    keys = set()
    for entry in metric_dicts:
        keys.update(k for k, v in entry.items() if isinstance(v, numbers.Number) and np.isfinite(v))
    for key in keys:
        values = [
            float(entry[key])
            for entry in metric_dicts
            if isinstance(entry.get(key), numbers.Number) and np.isfinite(entry.get(key))
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        if arr.size > 1:
            std = float(arr.std(ddof=1))
            ci = float(stats.t.ppf(0.975, arr.size - 1) * std / np.sqrt(arr.size)) if std > 0 else 0.0
        else:
            std = 0.0
            ci = 0.0
        agg[f"{key}_mean"] = mean
        agg[f"{key}_std"] = std
        agg[f"{key}_ci95"] = ci
    return agg


@dataclass
class CalibrationExample:
    """Container for a single probability vector and its target indices."""

    probs: np.ndarray
    target_indices: List[int]


def _ensure_list(obj: Sequence[str] | str | None) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    return [str(item) for item in obj]


def _replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {key: _replace_nan_with_none(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan_with_none(item) for item in obj]
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _is_multilabel_dataset(dataset: str) -> bool:
    return dataset in Config.TASK_TYPES.get("multiclass_classification", [])


def _get_class_catalog(dataset: str) -> Tuple[List[str], Dict[str, int]]:
    mapping = Config.VALID_D_TYPES.get(dataset)
    if not mapping:
        raise ValueError(f"No class mapping defined for dataset '{dataset}'")

    if dataset in Config.TASK_TYPES.get("ordinal_classification", []):
        ordered = sorted(mapping.keys(), key=lambda x: float(x))
        return ordered, {label: idx for idx, label in enumerate(ordered)}

    # Multiclass / multilabel datasets store human-readable labels in values.
    ordered_keys = sorted(mapping.keys(), key=lambda x: int(x))
    ordered = [mapping[key] for key in ordered_keys]
    return ordered, {label: idx for idx, label in enumerate(ordered)}


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z]+", "", str(label)).strip().lower()


def _aggregate_probability_vector(record: Dict, dataset: str, num_classes: int) -> Optional[np.ndarray]:
    raw_probs = record.get("probs")
    if raw_probs is None:
        return None

    if dataset in Config.TASK_TYPES.get("ordinal_classification", []):
        if not isinstance(raw_probs, Sequence):
            return None
        arr = np.asarray(raw_probs, dtype=float)
        if arr.size != num_classes:
            return None
        total = arr.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        return arr / total

    # Multilabel: raw_probs is a list of per-step class distributions.
    if isinstance(raw_probs, list) and raw_probs:
        arr = np.asarray(raw_probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != num_classes:
            return None
        mean_vector = arr.mean(axis=0)
        if not np.all(np.isfinite(mean_vector)):
            return None
        return np.clip(mean_vector, 0.0, 1.0)
    return None


def _collect_examples(results: Sequence[Dict], dataset: str) -> Tuple[List[CalibrationExample], List[str]]:
    class_labels, label_to_idx = _get_class_catalog(dataset)
    normalized_lookup = {_normalize_label(label): idx for label, idx in label_to_idx.items()}
    examples: List[CalibrationExample] = []
    for record in results:
        vec = _aggregate_probability_vector(record, dataset, len(class_labels))
        if vec is None:
            continue

        targets: List[int] = []
        if dataset in Config.TASK_TYPES.get("ordinal_classification", []):
            raw_true = str(record.get("true_value", ""))
            key = raw_true.split(":")[0].strip()
            idx = label_to_idx.get(key)
            if idx is None:
                continue
            targets.append(idx)
        else:
            raw = record.get("true_value")
            if isinstance(raw, str):
                candidates = [segment.strip() for segment in raw.split(",")]
            elif isinstance(raw, list):
                candidates = raw
            else:
                candidates = []
            for item in candidates:
                if not item:
                    continue
                key = label_to_idx.get(item)
                if key is None:
                    key = normalized_lookup.get(_normalize_label(item))
                if key is not None and key not in targets:
                    targets.append(key)
        if not targets:
            continue
        examples.append(CalibrationExample(vec.astype(float), targets))
    return examples, class_labels


def _split_examples(examples: List[CalibrationExample], rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[List[int]]]:
    if rng is None:
        rng = np.random.default_rng(seed=Config.SEED)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * Config.TRAIN_SET_SIZE)
    cal_end = train_end + int(n * Config.CALIBRATION_SET_SIZE)

    calibration = shuffled[train_end:cal_end]
    test = shuffled[cal_end:]

    cal_probs = np.stack([ex.probs for ex in calibration], axis=0) if calibration else np.empty((0, 0))
    test_probs = np.stack([ex.probs for ex in test], axis=0) if test else np.empty((0, 0))
    cal_targets = [ex.target_indices for ex in calibration]
    test_targets = [ex.target_indices for ex in test]
    return cal_probs, test_probs, cal_targets, test_targets


def _build_target_matrix(targets: Sequence[List[int]], num_classes: int, multilabel: bool) -> np.ndarray:
    n = len(targets)
    matrix = np.zeros((n, num_classes), dtype=np.float64)
    for i, indices in enumerate(targets):
        if multilabel:
            for idx in indices:
                if 0 <= idx < num_classes:
                    matrix[i, idx] = 1.0
        else:
            if not indices:
                continue
            idx = indices[0]
            if 0 <= idx < num_classes:
                matrix[i, idx] = 1.0
    return matrix


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, EPS, 1.0 - EPS)
    return np.log(clipped) - np.log(1.0 - clipped)


class PlattScaler:
    def __init__(self, multilabel: bool) -> None:
        self.multilabel = multilabel
        self._models: List[Optional[LogisticRegression]] = []

    def fit(self, probs: np.ndarray, targets: Sequence[List[int]]) -> None:
        if probs.ndim != 2:
            raise ValueError("Probability matrix must be 2D for Platt scaling")
        n_classes = probs.shape[1]
        target_matrix = _build_target_matrix(targets, n_classes, self.multilabel)
        self._models = []
        for class_idx in range(n_classes):
            y = target_matrix[:, class_idx]
            if y.sum() == 0 or y.sum() == y.size:
                self._models.append(None)
                continue
            x = _safe_logit(probs[:, class_idx]).reshape(-1, 1)
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            model.fit(x, y)
            self._models.append(model)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if probs.size == 0 or not self._models:
            return probs
        calibrated = probs.copy()
        for idx, model in enumerate(self._models):
            if model is None:
                continue
            x = _safe_logit(probs[:, idx]).reshape(-1, 1)
            if x.size == 0:
                continue
            class_probs = model.predict_proba(x)[:, 1]
            calibrated[:, idx] = class_probs
        if self.multilabel:
            return np.clip(calibrated, 0.0, 1.0)
        row_sum = calibrated.sum(axis=1, keepdims=True)
        safe_sum = np.where(row_sum > 0, row_sum, 1.0)
        normalized = calibrated / safe_sum
        normalized[~np.isfinite(normalized)] = 1.0 / calibrated.shape[1]
        return normalized


def _evaluate_single_label(probs: np.ndarray, targets: Sequence[List[int]]) -> Dict[str, float]:
    if probs.size == 0 or not targets:
        return {}
    labels = np.array([t[0] for t in targets], dtype=int)
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == labels))
    chosen = probs[np.arange(probs.shape[0]), labels]
    nll = float(-np.mean(np.log(np.clip(chosen, EPS, 1.0))))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    confidences = np.max(probs, axis=1)
    correctness = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    total = len(confidences)
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(correctness[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (count / total) * abs(bin_acc - bin_conf)
    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
    }


def _evaluate_multilabel(
    probs: np.ndarray,
    targets: Sequence[List[int]],
    class_labels: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if probs.size == 0 or not targets:
        return {}
    n_samples, n_classes = probs.shape
    y_true = np.zeros((n_samples, n_classes), dtype=int)
    true_names: List[List[str]] = []
    for i, indices in enumerate(targets):
        names = []
        for idx in indices:
            if 0 <= idx < n_classes:
                y_true[i, idx] = 1
                names.append(class_labels[idx])
        true_names.append(names)

    y_pred = (probs >= threshold).astype(int)
    f1_micro = float(f1_score(y_true, y_pred, average="micro"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    hamming = float(hamming_loss(y_true, y_pred))

    sequences = [[probs[i].tolist()] for i in range(n_samples)]
    ece, brier = compute_multilabel_calibration_metrics(true_names, sequences, class_labels)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": precision,
        "recall_micro": recall,
        "hamming_loss": hamming,
        "ece": float(ece) if ece is not None else float("nan"),
        "brier": float(brier) if brier is not None else float("nan"),
    }


def _load_results(results_path: Path, dataset: str) -> List[Dict]:
    with results_path.open("r", encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle]
    filtered = [entry for entry in raw if entry.get("ds_type") == dataset]
    return cleaning_results(filtered, dataset)


def _run_baselines(dataset: str, model: str, output_path: Path) -> Tuple[Dict, Optional[Dict], Optional[Dict], bool]:
    original_model = Config.MODEL_NAME_OR_PATH
    original_dataset = Config.DS_TYPE
    try:
        Config.update_model_and_dataset(model, dataset)
        Config.update_paths()

        results_path = Path(Config.RESULTS_FILE)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        raw_results = _load_results(results_path, dataset)
        examples, class_labels = _collect_examples(raw_results, dataset)
        if not examples:
            raise RuntimeError(f"No usable probability vectors found for dataset '{dataset}'")

        multilabel = _is_multilabel_dataset(dataset)
        metrics_history: Dict[str, List[Dict[str, float]]] = {"identity": [], "platt": []}
        identity_dict_first: Optional[Dict[str, object]] = None
        platt_dict_first: Optional[Dict[str, object]] = None

        def _label_name(idx: int) -> str:
            if 0 <= idx < len(class_labels):
                return class_labels[idx]
            return str(idx)

        def make_results_dict(
            probs_matrix: Optional[np.ndarray],
            targets: Sequence[List[int]],
        ) -> Optional[Dict[str, object]]:
            if probs_matrix is None or probs_matrix.size == 0:
                return None
            payload: Dict[str, object] = {}
            if multilabel:
                payload["probs"] = [[row.tolist()] for row in probs_matrix]
                payload["true_values"] = [
                    [_label_name(idx) for idx in indices] for indices in targets
                ]
                payload["predictions"] = []
            else:
                payload["probs"] = probs_matrix.tolist()
                true_values = [(_label_name(indices[0]), indices[0]) for indices in targets]
                pred_indices = np.argmax(probs_matrix, axis=1)
                predictions = [(_label_name(int(idx)), int(idx)) for idx in pred_indices]
                payload["true_values"] = true_values
                payload["predictions"] = predictions
            return payload

        num_repeats = max(1, int(getattr(Config, "NUM_REPEATS", 1)))
        for repeat_idx in range(num_repeats):
            rng = np.random.default_rng(Config.SEED + repeat_idx)
            cal_probs, test_probs, cal_targets, test_targets = _split_examples(examples, rng)
            if test_probs.size == 0 or not test_targets:
                continue

            identity_metrics = (
                _evaluate_multilabel(test_probs, test_targets, class_labels)
                if multilabel
                else _evaluate_single_label(test_probs, test_targets)
            )
            if identity_metrics:
                metrics_history["identity"].append(identity_metrics)
            if identity_dict_first is None:
                identity_dict_first = make_results_dict(test_probs, test_targets)

            if cal_probs.size == 0:
                continue

            platt = PlattScaler(multilabel=multilabel)
            platt.fit(cal_probs, cal_targets)
            platt_probs = platt.transform(test_probs)
            platt_metrics = (
                _evaluate_multilabel(platt_probs, test_targets, class_labels)
                if multilabel
                else _evaluate_single_label(platt_probs, test_targets)
            )
            if platt_metrics:
                metrics_history["platt"].append(platt_metrics)
            if platt_dict_first is None:
                platt_dict_first = make_results_dict(platt_probs, test_targets)

        if not metrics_history["identity"]:
            raise RuntimeError(f"No valid identity baseline evaluations for dataset '{dataset}'")

        baseline_stats = {
            scheme: _aggregate_metric_dicts(entries)
            for scheme, entries in metrics_history.items()
        }

        identity_dict = identity_dict_first
        platt_dict = platt_dict_first
        payload: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "results_file": str(results_path),
            "num_examples": len(examples),
            "class_labels": class_labels,
            "baselines": baseline_stats,
        }

        sanitized = _replace_nan_with_none(payload)
        serializable = convert_to_serializable(sanitized)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return serializable, identity_dict, platt_dict, multilabel
    finally:
        Config.MODEL_NAME_OR_PATH = original_model
        Config.DS_TYPE = original_dataset
        Config.update_paths()


def _flatten_metrics(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    dataset = payload.get("dataset")
    model = payload.get("model")
    baselines = payload.get("baselines", {})
    for scheme, metrics in baselines.items():
        row = {
            "dataset": dataset,
            "model": model,
            "scheme": scheme,
        }
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return rows


def _write_metrics_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        print("No baseline metrics to write.")
        return

    base_fields = ["dataset", "model", "scheme"]
    extra_fields = sorted({key for row in rows for key in row.keys() if key not in base_fields})
    fieldnames = base_fields + extra_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Baseline metrics CSV saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Platt baselines for emotion datasets/models")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Path to the aggregated CSV metrics file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = Config.get_baseline_datasets()
    models = Config.get_baseline_models()
    if not datasets:
        raise SystemExit("No baseline datasets defined in Config.BASELINE_DATASETS.")
    if not models:
        raise SystemExit("No baseline models defined in Config.BASELINE_MODEL_NAMES.")

    rows: List[Dict[str, object]] = []
    dataset_plot_entries: Dict[str, Dict[str, object]] = {}
    for model_name in models:
        for dataset in datasets:
            short_model = model_name.split("/")[-1]
            output_path = OUTPUT_DIR / dataset / f"{short_model}.json"
            try:
                payload, identity_dict, platt_dict, is_multilabel = _run_baselines(dataset, model_name, output_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Skipping {model_name} on {dataset}: {exc}")
                continue
            print(json.dumps(payload, indent=2))
            rows.extend(_flatten_metrics(payload))
            plot_bucket = dataset_plot_entries.setdefault(
                dataset,
                {
                    "dataset": dataset,
                    "multilabel": is_multilabel,
                    "entries": [],
                    "task_type": _infer_task_type(dataset),
                },
            )
            plot_bucket["dataset"] = dataset
            model_label = short_model
            if identity_dict:
                plot_bucket["entries"].append({"label": f"{model_label} · Identity", "results": identity_dict})
            if platt_dict:
                plot_bucket["entries"].append({"label": f"{model_label} · Platt", "results": platt_dict})

    _write_metrics_csv(rows, args.csv_output)

    task_groups: Dict[str, List[Dict[str, object]]] = {}
    for dataset, info in dataset_plot_entries.items():
        entries = info.get("entries") or []
        if not entries:
            continue
        task_type = info.get("task_type") or _infer_task_type(dataset)
        record = {
            "dataset": dataset,
            "entries": entries,
            "multilabel": bool(info.get("multilabel")),
        }
        task_groups.setdefault(task_type, []).append(record)

    panel_root = OUTPUT_DIR.parent / "figures" / "task_panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    for task_name, infos in task_groups.items():
        if not infos:
            continue
        out_path = panel_root / f"{_slug(task_name)}_baseline_panel.png"
        vis.plot_task_comparison_panel(task_name, infos, str(out_path))


if __name__ == "__main__":
    main()
