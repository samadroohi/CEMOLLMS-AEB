#!/usr/bin/env python3
"""Regenerate conformal summary tables across all available datasets/models."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

PYTHON = sys.executable


def _run_step(cmd: Sequence[str]) -> bool:
    cmd_list = list(cmd)
    print("→", " ".join(cmd_list))
    try:
        subprocess.run(cmd_list, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"   failed (exit code {exc.returncode})")
        return False


def main() -> None:
    print("Refreshing calibration summaries...")
    if not _run_step([PYTHON, "-m", "src.analysis.calibration_analysis"]):
        sys.exit(1)

    print("Generating conformal tables...")
    _run_step([PYTHON, "-m", "src.analysis.generate_conformal_tables"])

    print("Generating performance tables...")
    _run_step([
        PYTHON,
        "-m",
        "src.analysis.generate_performance_tables",
    ])


if __name__ == "__main__":
    main()
