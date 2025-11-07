#!/usr/bin/env python3
"""Generate summary tables for conformal coverage and prediction-set sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "calibration"
DEFAULT_ALPHA_SUMMARY = OUTPUT_DIR / "calibration_alpha_summary.csv"
DEFAULT_TABLE_PATH = OUTPUT_DIR / "conformal_summary_table.csv"
DEFAULT_MARKDOWN_PATH = OUTPUT_DIR / "conformal_summary_table.md"
CONFIDENCE_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _format_entry(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.2f} ±{std:.2f}"


def _summarise_stats(df: pd.DataFrame,
                     dataset: str,
                     model: str,
                     mode: str | None,
                     metric: str) -> tuple[str, List[tuple[float, float, float]]]:
    entries: List[str] = []
    stats: List[tuple[float, float, float]] = []
    for conf in CONFIDENCE_LEVELS:
        alpha = round(1.0 - conf, 1)
        mask = (
            (df["dataset"] == dataset)
            & (df["model"] == model)
            & (df["mode"] == (mode or ""))
            & (df["alpha"].round(1) == alpha)
            & (df["metric"] == metric)
        )
        subset = df[mask]
        if subset.empty:
            entries.append(_format_entry(None, None))
            continue
        mean_val = float(subset.iloc[0]["mean"])
        std_val = float(subset.iloc[0]["std"])
        entries.append(_format_entry(mean_val, std_val))
        stats.append((conf, mean_val, std_val))
    formatted = f"({', '.join(entries)})" if entries else "()"
    return formatted, stats


def _compute_ace(coverage_stats: Iterable[tuple[float, float, float]]) -> str:
    entries = list(coverage_stats)
    if not entries:
        return "n/a"
    errors = [abs(mean_val - conf) for conf, mean_val, _ in entries]
    ace = float(np.mean(errors))
    if len(errors) > 1:
        ace_std = float(np.std(errors, ddof=0))
    else:
        ace_std = 0.0
    return f"{ace:.2f} ±{ace_std:.2f}"


def build_table(alpha_summary: pd.DataFrame) -> pd.DataFrame:
    sanitized = alpha_summary.copy()
    sanitized["mode"] = sanitized["mode"].fillna("")
    rows = []
    for (dataset, model, mode), _ in sanitized.groupby(["dataset", "model", "mode"]):
        coverage_str, coverage_stats = _summarise_stats(
            sanitized, dataset, model, mode, "coverage"
        )
        size_str, _ = _summarise_stats(
            sanitized, dataset, model, mode, "interval_size"
        )
        ace = _compute_ace(coverage_stats)
        row = {
            "dataset": dataset,
            "model": model,
            "mode": mode or "default",
            "coverage_stats": coverage_str,
            "size_stats": size_str,
            "ACE": ace,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values(["dataset", "model", "mode"]).reset_index(drop=True)
    return table


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = []
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conformal summary table")
    parser.add_argument("--alpha-summary", type=Path, default=DEFAULT_ALPHA_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--skip-markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha_summary = pd.read_csv(args.alpha_summary)
    table = build_table(alpha_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Saved conformal summary table to {args.output}")

    if not args.skip_markdown and args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_text = _dataframe_to_markdown(table)
        args.markdown_output.write_text(markdown_text, encoding="utf-8")
        print(f"Saved Markdown table to {args.markdown_output}")


if __name__ == "__main__":
    main()
