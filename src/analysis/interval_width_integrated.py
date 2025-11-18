import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analysis_output.calibration.style import (
    FIG_WIDTH_2COL,
    GOLDEN_RATIO,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    apply_publication_style,
    COLOR_PALETTE,
)


def _to_float_array(values: Iterable) -> np.ndarray:
    out: List[float] = []
    for v in values or []:
        if isinstance(v, (list, tuple)):
            out.append(float(v[0]))
        else:
            out.append(float(v))
    return np.asarray(out, dtype=float)


def _load_entries(results_dir: Path, alpha: float | None) -> Iterable[Dict]:
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for temp_dir in sorted(dataset_dir.iterdir()):
            if not temp_dir.is_dir():
                continue
            for json_path in sorted(temp_dir.glob("*.json")):
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for rec in payload.get("results", []):
                    rec_alpha = rec.get("alpha")
                    if alpha is not None and not np.isclose(rec_alpha, alpha):
                        continue
                    try:
                        true_vals = _to_float_array(rec.get("true_values", []))
                        pred_vals = _to_float_array(rec.get("predictions", []))
                        lower = _to_float_array(rec.get("prediction_sets", [[], []])[0])
                        upper = _to_float_array(rec.get("prediction_sets", [[], []])[1])
                    except (ValueError, TypeError) as exc:
                        print(f"[WARN] Skipping {json_path} (repeat={rec.get('repeat_index')}): {exc}")
                        continue
                    yield {
                        "dataset": payload.get("dataset_type", rec.get("dataset_type")),
                        "temperature": temp_dir.name,
                        "model": json_path.stem,
                        "mode": rec.get("mode"),
                        "repeat": int(rec.get("repeat_index", 0)),
                        "alpha": rec_alpha,
                        "true": true_vals,
                        "pred": pred_vals,
                        "lower": lower,
                        "upper": upper,
                    }


def _aggregate_by_dataset(entries: Iterable[Dict]) -> Dict[str, Dict[Tuple[str, str], Dict[str, List[float]]]]:
    out: Dict[str, Dict[Tuple[str, str], Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {"widths": [], "abs_res": []}))
    for entry in entries:
        width = entry["upper"] - entry["lower"]
        n = min(width.size, entry["true"].size, entry["pred"].size)
        if n == 0:
            continue
        width = width[:n]
        abs_res = np.abs(entry["true"][:n] - entry["pred"][:n])
        dataset = entry["dataset"]
        key = (entry["model"], entry["mode"])
        out[dataset][key]["widths"].extend(width.tolist())
        out[dataset][key]["abs_res"].extend(abs_res.tolist())
    return out


def _marker_cycle() -> List[str]:
    return ["o", "s", "D", "^", "v", "P", "X", "*", "h"]


def _color_map() -> Dict[str, str]:
    return {
        "regression": COLOR_PALETTE[0],
        "quantilized_regression": COLOR_PALETTE[1],
    }


def _build_legends(models: List[str], ax):
    import matplotlib.lines as mlines

    color_map = _color_map()
    method_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None", label=label.capitalize())
        for label, color in [("regression", color_map["regression"]), ("quantilized_regression", color_map["quantilized_regression"])]
    ]
    marker_handles = [
        mlines.Line2D(
            [],
            [],
            color="#4d4d4d",
            marker=marker,
            linestyle="None",
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=model,
        )
        for model, marker in models
    ]
    handles = method_handles + marker_handles
    ax.legend(handles=handles, title="Method / Model", loc="upper right", ncol=2, fontsize=5.5)


def _plot_dataset(dataset: str, data: Dict[Tuple[str, str], Dict[str, List[float]]], output_dir: Path, alpha: float):
    apply_publication_style(FIG_WIDTH_2COL, GOLDEN_RATIO)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_2COL, FIG_WIDTH_2COL * GOLDEN_RATIO))

    color_map = _color_map()
    markers = _marker_cycle()
    model_order = sorted({model for model, _ in data.keys()})
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(model_order)}

    # Panel A: width summaries per model/method
    x_positions = np.arange(len(model_order))
    offsets = {"regression": -0.08, "quantilized_regression": 0.08}

    for (model, method), stats_dict in data.items():
        widths = np.asarray(stats_dict["widths"], dtype=float)
        if widths.size == 0:
            continue
        q1 = np.percentile(widths, 25)
        median = np.median(widths)
        q3 = np.percentile(widths, 75)
        idx = model_order.index(model)
        x = x_positions[idx] + offsets.get(method, 0.0)
        ax.errorbar(
            x,
            median,
            yerr=[[median - q1], [q3 - median]],
            fmt=marker_map[model],
            color=color_map.get(method, "#333333"),
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=MARKER_EDGE_WIDTH,
            ecolor=color_map.get(method, "#333333"),
            capsize=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_order, rotation=20, ha="right")
    ax.set_ylabel("Interval width (median ± IQR)")
    ax.set_title(f"{dataset}: width summaries (α={alpha:.2f})")

    _build_legends(list(marker_map.items()), ax)
    fig.suptitle(f"{dataset} · interval width summary", fontsize=7)
    fig.tight_layout()
    out_path = output_dir / f"{dataset}_integrated_alpha{alpha:.2f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create integrated interval adaptiveness figures per dataset.")
    parser.add_argument("--results_dir", type=Path, default=Path("results/conformal_results"))
    parser.add_argument("--output_dir", type=Path, default=Path("analysis_output/adaptiveness/combined"))
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    entries = list(_load_entries(args.results_dir, args.alpha))
    if not entries:
        raise SystemExit("No conformal entries found. Run conformal prediction first.")
    aggregated = _aggregate_by_dataset(entries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, grouped in aggregated.items():
        _plot_dataset(dataset, grouped, args.output_dir, args.alpha)
    print(f"Integrated figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
