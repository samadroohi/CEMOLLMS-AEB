#!/usr/bin/env python3
"""Repeat-aware calibration analysis for conformal prediction outputs."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from config import Config  # noqa: E402
from analysis.analysis_utils import get_performance_metrics  # noqa: E402


@dataclass
class CalibrationAnalyzer:
    """Aggregate calibration metrics across repeated conformal runs."""

    conformal_dir: Path = REPO_ROOT / "results" / "conformal_results"
    temp_dir: str = "temp_0.9"
    output_dir: Path = REPO_ROOT / "analysis_output" / "calibration"
    confidence: float = 0.95

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_types = Config.TASK_TYPES

    def run(self) -> None:
        alpha_records, repeat_records = self._collect_records()

        if alpha_records.empty and repeat_records.empty:
            print("No conformal results found. Exiting.")
            return

        if not alpha_records.empty:
            alpha_records.to_csv(self.output_dir / "calibration_alpha_records.csv", index=False)
            alpha_summary = self._summarize_alpha_metrics(alpha_records)
            alpha_summary.to_csv(self.output_dir / "calibration_alpha_summary.csv", index=False)
            print(f"Saved per-alpha records and summary to {self.output_dir}")

        if not repeat_records.empty:
            repeat_records.to_csv(self.output_dir / "calibration_repeat_records.csv", index=False)
            repeat_summary = self._summarize_repeat_metrics(repeat_records)
            repeat_summary.to_csv(self.output_dir / "calibration_repeat_summary.csv", index=False)
            print(f"Saved per-repeat records and summary to {self.output_dir}")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------
    def _collect_records(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        alpha_rows: List[Dict[str, float]] = []
        repeat_rows: List[Dict[str, float]] = []

        for dataset_dir in sorted(self.conformal_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            temp_path = dataset_dir / self.temp_dir
            if not temp_path.exists():
                continue

            for json_path in sorted(temp_path.glob("*.json")):
                model = json_path.stem.split("__", 1)[0]
                payload = self._safe_load_json(json_path)
                if not payload:
                    continue

                mode_records: Dict[int, List[Dict]] = {}
                for record in payload.get("results", []):
                    repeat_idx = int(record.get("repeat_index", 0))
                    mode = record.get("mode") or self._infer_mode(json_path)
                    record["__mode"] = mode
                    mode_records.setdefault(repeat_idx, []).append(record)

                for repeat_idx, records in mode_records.items():
                    if not records:
                        continue
                    records_sorted = sorted(records, key=lambda r: float(r.get("alpha", 0)))
                    coverage_errors: List[float] = []

                    for record in records_sorted:
                        alpha = float(record.get("alpha"))
                        coverage = float(record.get("coverage"))
                        interval_size = float(record.get("interval_size"))
                        mode = record["__mode"]
                        coverage_error = abs(coverage - (1.0 - alpha))
                        alpha_rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "mode": mode,
                                "alpha": alpha,
                                "repeat_index": repeat_idx,
                                "coverage": coverage,
                                "interval_size": interval_size,
                                "ece": coverage_error,
                            }
                        )
                        coverage_errors.append(coverage_error)

                    ace_value = float(np.mean(coverage_errors)) if coverage_errors else np.nan
                    metrics_record = records_sorted[0]
                    repeat_metrics = self._compute_repeat_metrics(dataset, metrics_record)
                    repeat_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "mode": records_sorted[0]["__mode"],
                            "repeat_index": repeat_idx,
                            "accuracy": repeat_metrics.get("accuracy"),
                            "f1_micro": repeat_metrics.get("f1_micro"),
                            "f1_macro": repeat_metrics.get("f1_macro"),
                            "pcc": repeat_metrics.get("pcc"),
                            "ace": ace_value,
                        }
                    )

        alpha_df = pd.DataFrame(alpha_rows)
        repeat_df = pd.DataFrame(repeat_rows)
        return alpha_df, repeat_df

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_load_json(path: Path) -> Optional[Dict]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping {path}: {exc}")
            return None

    @staticmethod
    def _infer_mode(path: Path) -> str:
        stem = path.stem
        if "__" in stem:
            return stem.split("__", 1)[1]
        return "default"

    def _compute_repeat_metrics(self, dataset: str, record: Dict) -> Dict[str, Optional[float]]:
        payload = {
            "ds_type": dataset,
            "true_values": record.get("true_values", []),
            "predictions": record.get("predictions", []),
            "probs": record.get("probs", []),
            "prediction_sets": record.get("prediction_sets", []),
        }

        try:
            raw_metrics = get_performance_metrics(payload, dataset, self.task_types)
        except Exception as exc:  # noqa: BLE001
            print(f"Metric computation failed for {dataset}: {exc}")
            raw_metrics = {}

        metrics: Dict[str, Optional[float]] = {
            "accuracy": np.nan,
            "f1_micro": np.nan,
            "f1_macro": np.nan,
            "pcc": np.nan,
        }

        if dataset in self.task_types.get("regression", []):
            metrics["pcc"] = raw_metrics.get("pearson_correlation")

        elif dataset in self.task_types.get("ordinal_classification", []):
            metrics["accuracy"] = raw_metrics.get("accuracy")
            metrics["f1_micro"] = raw_metrics.get("micro_f1")
            metrics["f1_macro"] = raw_metrics.get("macro_f1")
            metrics["pcc"] = raw_metrics.get("average_pearson")

        elif dataset in self.task_types.get("multiclass_classification", []):
            metrics["f1_micro"] = raw_metrics.get("f1_micro")
            metrics["f1_macro"] = raw_metrics.get("f1_macro")
            metrics["accuracy"] = self._multilabel_subset_accuracy(
                payload["true_values"], payload["predictions"]
            )

        return metrics

    @staticmethod
    def _multilabel_subset_accuracy(true_values: Iterable[Iterable[str]],
                                    pred_values: Iterable[Iterable[str]]) -> Optional[float]:
        true_list = [list(labels) for labels in true_values]
        pred_list = [list(labels) for labels in pred_values]
        if not true_list:
            return None

        unique_labels = sorted({label for labels in true_list + pred_list for label in labels})
        if not unique_labels:
            return None
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        def to_binary(labels: List[str]) -> np.ndarray:
            vec = np.zeros(len(unique_labels), dtype=int)
            for label in labels:
                idx = label_to_index.get(label)
                if idx is not None:
                    vec[idx] = 1
            return vec

        y_true = np.array([to_binary(labels) for labels in true_list])
        y_pred = np.array([to_binary(labels) for labels in pred_list])
        if y_true.size == 0:
            return None
        return float(accuracy_score(y_true, y_pred))

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def _summarize_alpha_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        summaries: List[Dict] = []
        for metric_name in ("coverage", "interval_size", "ece"):
            summaries.extend(
                self._aggregate_metric(
                    df,
                    by_cols=["dataset", "model", "mode", "alpha"],
                    value_col=metric_name,
                    metric_name=metric_name,
                )
            )
        return pd.DataFrame(summaries)

    def _summarize_repeat_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        summaries: List[Dict] = []
        for metric_name in ("accuracy", "f1_micro", "f1_macro", "pcc", "ace"):
            summaries.extend(
                self._aggregate_metric(
                    df,
                    by_cols=["dataset", "model", "mode"],
                    value_col=metric_name,
                    metric_name=metric_name,
                )
            )
        return pd.DataFrame(summaries)

    def _aggregate_metric(
        self,
        df: pd.DataFrame,
        by_cols: List[str],
        value_col: str,
        metric_name: str,
    ) -> List[Dict]:
        rows: List[Dict] = []
        if value_col not in df:
            return rows

        grouped = df.dropna(subset=[value_col]).groupby(by_cols)[value_col]
        for keys, series in grouped:
            series = series.dropna()
            if series.empty:
                continue

            n = len(series)
            mean = series.mean()
            std = series.std(ddof=1) if n > 1 else 0.0
            ci_half_width = self._confidence_interval(std, n)

            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            row = {col: key for col, key in zip(by_cols, key_tuple)}
            row.update(
                {
                    "metric": metric_name,
                    "mean": float(mean),
                    "std": float(std),
                    "ci_lower": float(mean - ci_half_width),
                    "ci_upper": float(mean + ci_half_width),
                    "n": int(n),
                }
            )
            rows.append(row)

        return rows

    def _confidence_interval(self, std: float, n: int) -> float:
        if n <= 1 or std == 0:
            return 0.0
        sem = std / math.sqrt(n)
        t_score = stats.t.ppf((1 + self.confidence) / 2, n - 1)
        return float(sem * t_score)


def main() -> None:
    analyzer = CalibrationAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
