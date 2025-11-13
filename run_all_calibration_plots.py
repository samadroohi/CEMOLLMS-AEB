#!/usr/bin/env python3
"""Run calibration plots for every model/dataset pair, skipping missing combos."""

from __future__ import annotations
from src.config import Config

import subprocess
import sys




PYTHON = sys.executable


def run_or_skip(cmd: list[str]) -> bool:
    """Execute command; print a message if it fails and continue."""
    print("->", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"   skipped (exit code {exc.returncode})")
        return False


def main() -> None:
    # Refresh aggregated summaries first; if this fails we bail out early.
    print("Running calibration analysis to refresh CSV summaries...")
    if not run_or_skip([PYTHON, "-m", "src.analysis.calibration_analysis"]):
        sys.exit(1)

    for model in Config.BASELINE_MODEL_NAMES:
        short_name = model.split("/")[-1]
        for dataset in Config.BASELINE_DATASETS:
            cmd = [
                PYTHON,
                "-m",
                "src.analysis.calibration_plots",
                "--dataset",
                dataset,
                "--model",
                short_name,
            ]
            run_or_skip(cmd)

    print("Generating merged coverage figures...")
    merge_cmd = [
        PYTHON,
        "-m",
        "src.analysis.merged_calibration_plots",
        "--datasets",
        *Config.BASELINE_DATASETS,
    ]
    run_or_skip(merge_cmd)

    print("Building conformal summary table...")
    table_cmd = [
        PYTHON,
        "-m",
        "src.analysis.generate_conformal_tables",
    ]
    run_or_skip(table_cmd)


if __name__ == "__main__":
    main()
