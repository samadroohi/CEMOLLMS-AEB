import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - optional dependency
    wilcoxon = None


def _to_float_array(values: Iterable) -> np.ndarray:
    arr = []
    for v in values or []:
        if isinstance(v, (list, tuple)):
            arr.append(float(v[0]))
        else:
            arr.append(float(v))
    return np.asarray(arr, dtype=float)


def _load_conformal_entries(results_dir: Path, alpha: float | None) -> Iterable[Dict]:
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for temp_dir in dataset_dir.iterdir():
            if not temp_dir.is_dir():
                continue
            for json_path in temp_dir.glob("*.json"):
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for rec in payload.get("results", []):
                    if alpha is not None and not np.isclose(rec.get("alpha"), alpha):
                        continue
                    try:
                        true_vals = _to_float_array(rec.get("true_values", []))
                        pred_vals = _to_float_array(rec.get("predictions", []))
                        lower = _to_float_array(rec.get("prediction_sets", [[], []])[0])
                        upper = _to_float_array(rec.get("prediction_sets", [[], []])[1])
                    except (ValueError, TypeError) as exc:
                        print(f"[WARN] Failed to parse arrays in {json_path}: {exc}")
                        continue

                    yield {
                        "dataset": payload.get("dataset_type", rec.get("dataset_type")),
                        "temperature": temp_dir.name,
                        "model": json_path.stem,
                        "mode": rec.get("mode"),
                        "repeat": int(rec.get("repeat_index", 0)),
                        "alpha": rec.get("alpha"),
                        "true": true_vals,
                        "pred": pred_vals,
                        "lower": lower,
                        "upper": upper,
                    }


def _summarize_widths(widths: np.ndarray) -> Dict[str, float]:
    q1 = float(np.percentile(widths, 25))
    q3 = float(np.percentile(widths, 75))
    return {
        "median": float(np.median(widths)),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "mean": float(np.mean(widths)),
        "std": float(np.std(widths)),
        "min": float(np.min(widths)),
        "max": float(np.max(widths)),
    }


def _plot_width_distributions(
    baseline: np.ndarray,
    cqr: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, max(baseline.max(), cqr.max()), 40)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].hist(
        baseline,
        bins=bins,
        alpha=0.6,
        label="Split Conformal",
        color="#377eb8",
        density=True,
    )
    axes[0].hist(
        cqr,
        bins=bins,
        alpha=0.6,
        label="CQR",
        color="#e41a1c",
        density=True,
    )
    axes[0].set_xlabel("Interval width")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Width distribution")
    axes[0].legend()

    for arr, label, color in [
        (np.sort(baseline), "Split Conformal", "#377eb8"),
        (np.sort(cqr), "CQR", "#e41a1c"),
    ]:
        axes[1].plot(arr, np.linspace(0, 1, arr.size), label=label, color=color)
    axes[1].set_xlabel("Interval width")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Width CDF")
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_residual_relationship(
    abs_resid: np.ndarray,
    widths: np.ndarray,
    title: str,
    output_path: Path,
    max_points: int = 4000,
    num_bins: int = 20,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if abs_resid.size > max_points:
        idx = rng.choice(abs_resid.size, size=max_points, replace=False)
        abs_resid = abs_resid[idx]
        widths = widths[idx]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(abs_resid, widths, s=12, alpha=0.2, color="#555555", edgecolors="none")

    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.unique(np.quantile(abs_resid, quantiles))
    if bins.size < 2:
        bins = np.linspace(abs_resid.min(), abs_resid.max(), num_bins + 1)
    centers = []
    medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (abs_resid >= lo) & (abs_resid <= hi)
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (lo + hi))
        medians.append(np.median(widths[mask]))
    if centers:
        ax.plot(centers, medians, color="#e41a1c", linewidth=2.0, label="Bin median")
        ax.legend()

    ax.set_xlabel("|y - y_hat|")
    ax.set_ylabel("CQR interval width")
    ax.set_title(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate interval-width diagnostics for CQR vs. split conformal."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/conformal_results"),
        help="Path to conformal results directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_output/adaptiveness"),
        help="Directory to store plots and summary CSVs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Significance level to analyze (set None for all alphas).",
    )
    args = parser.parse_args()

    records: Dict[Tuple[str, str, str, int], Dict[str, Dict]] = defaultdict(dict)
    for entry in _load_conformal_entries(args.results_dir, args.alpha):
        key = (entry["dataset"], entry["temperature"], entry["model"], entry["repeat"])
        records[key][entry["mode"]] = entry

    if not records:
        raise SystemExit("No conformal results found. Run conformal prediction first.")

    per_repeat_rows: List[Dict] = []
    wilcoxon_rows: List[Dict] = []
    aggregate: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"baseline": [], "cqr": [], "abs_resid": []}
    )

    for (dataset, temp, model, repeat), mode_map in records.items():
        if "quantilized_regression" not in mode_map or "regression" not in mode_map:
            continue
        cqr_entry = mode_map["quantilized_regression"]
        base_entry = mode_map["regression"]
        lower_cqr, upper_cqr = cqr_entry["lower"], cqr_entry["upper"]
        lower_base, upper_base = base_entry["lower"], base_entry["upper"]

        if not (lower_cqr.size and lower_base.size):
            continue

        widths_cqr = upper_cqr - lower_cqr
        widths_base = upper_base - lower_base

        if widths_cqr.size != widths_base.size:
            min_len = min(widths_cqr.size, widths_base.size)
            widths_cqr = widths_cqr[:min_len]
            widths_base = widths_base[:min_len]

        abs_resid = np.abs(cqr_entry["true"][: widths_cqr.size] - cqr_entry["pred"][: widths_cqr.size])

        for method, arr in [("CQR", widths_cqr), ("Split", widths_base)]:
            stats = _summarize_widths(arr)
            per_repeat_rows.append(
                {
                    "dataset": dataset,
                    "temperature": temp,
                    "model": model,
                    "repeat": repeat,
                    "alpha": cqr_entry["alpha"],
                    "method": method,
                    "n": int(arr.size),
                    **stats,
                }
            )

        if wilcoxon is not None and widths_base.size > 0:
            try:
                stat, pvalue = wilcoxon(widths_base, widths_cqr, zero_method="wilcox")
            except ValueError:
                stat, pvalue = np.nan, np.nan
        else:
            stat, pvalue = np.nan, np.nan

        wilcoxon_rows.append(
            {
                "dataset": dataset,
                "temperature": temp,
                "model": model,
                "repeat": repeat,
                "alpha": cqr_entry["alpha"],
                "n": int(widths_cqr.size),
                "median_reduction": float(np.median(widths_base - widths_cqr)),
                "mean_reduction": float(np.mean(widths_base - widths_cqr)),
                "wilcoxon_stat": stat,
                "wilcoxon_pvalue": pvalue,
            }
        )

        agg_key = (dataset, temp, model)
        aggregate[agg_key]["baseline"].extend(widths_base.tolist())
        aggregate[agg_key]["cqr"].extend(widths_cqr.tolist())
        aggregate[agg_key]["abs_resid"].extend(abs_resid.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_repeat_rows).to_csv(
        args.output_dir / "interval_width_summary.csv", index=False
    )
    pd.DataFrame(wilcoxon_rows).to_csv(
        args.output_dir / "interval_width_wilcoxon.csv", index=False
    )

    for (dataset, temp, model), data_dict in aggregate.items():
        baseline = np.asarray(data_dict["baseline"], dtype=float)
        cqr = np.asarray(data_dict["cqr"], dtype=float)
        abs_resid = np.asarray(data_dict["abs_resid"], dtype=float)
        if baseline.size == 0 or cqr.size == 0:
            continue
        title = f"{dataset} | {model} | {temp}"
        dist_path = (
            args.output_dir
            / dataset
            / temp
            / f"{model}_alpha{args.alpha}_width_distributions.png"
        )
        _plot_width_distributions(baseline, cqr, title, dist_path)

        rel_path = (
            args.output_dir
            / dataset
            / temp
            / f"{model}_alpha{args.alpha}_width_vs_residual.png"
        )
        _plot_residual_relationship(abs_resid, cqr, title, rel_path)

    print(
        f"Interval-width diagnostics written to {args.output_dir}. "
        "Figures are grouped by dataset/model."
    )


if __name__ == "__main__":
    main()
