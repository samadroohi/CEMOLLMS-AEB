#!/usr/bin/env python3
"""Run calibration plots for every model/dataset pair, skipping missing combos."""

from __future__ import annotations

import subprocess
import sys

MODELS = [
    "lzw1008/Emollama-7b",
    "lzw1008/Emobloom-7b",
    "lzw1008/Emollama-chat-7b",
    "lzw1008/Emollama-chat-13b",
    "lzw1008/Emoopt-13b",
    
]

DATASETS = [
    "EI-oc",
    "TDT",
    "SST5",
    "V-oc",
    "EI-reg",
    "V-reg",
    "V-A,V-M,V-NYT,V-T",
    "Emobank",
    "SST",
    "GoEmotions",
    "E-c",
]

PYTHON = sys.executable


def run_or_skip(cmd: list[str]) -> bool:
    """Execute command; print a message if it fails and continue."""
    print("→", " ".join(cmd))
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

    for model in MODELS:
        short_name = model.split("/")[-1]
        for dataset in DATASETS:
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
        *DATASETS,
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
