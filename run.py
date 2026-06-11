"""Cross-platform pipeline orchestrator.

Runs the full single- or double-stent virtual stenting pipeline in order:
  1. constructAneuGeom  — build vessel geometry and centreline
  2. constructInitFD    — build initial (crimped) outer stent
  3. deployStent        — deploy outer stent
  4. mergeMesh          — merge outer stent into vessel wall
  5. constructInitFD    — build initial (crimped) inner stent
  6. deployStent        — deploy inner stent

Equivalent to the original ``runSym.cmd`` + ``script/*.bat``, but runs on
both Linux and Windows.

Usage (Linux / macOS)::

    uv run python run.py --experiment_dir "experiment/experiment 0"
    uv run python run.py --experiment_dir "experiment/experiment 0" --clean

Usage (Windows PowerShell / cmd)::

    uv run python run.py --experiment_dir "experiment\\experiment 0"
    uv run python run.py --experiment_dir "experiment\\experiment 0" --clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import constructAneuGeom
import constructInitFD
import deployStent
import mergeMesh


def clean(dir_path: str) -> None:
    """Delete the ``results/`` directory for the given experiment.

    Args:
        dir_path: Experiment directory path.
    """
    results_dir = Path(dir_path) / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Removed {results_dir}")
    else:
        print(f"Nothing to clean: {results_dir} does not exist.")


def run(dir_path: str) -> None:
    """Execute all pipeline steps for the given experiment directory.

    Args:
        dir_path: Experiment directory path (must contain ``appSettings.json``).
    """
    print(f"=== Step 1: Build aneurysm geometry [{dir_path}] ===")
    constructAneuGeom.main(dir_path)

    print(f"=== Step 2: Build initial outer stent ===")
    constructInitFD.main(dir_path, "outer")

    print(f"=== Step 3: Deploy outer stent ===")
    deployStent.main(dir_path, "outer")

    print(f"=== Step 4: Merge outer stent into vessel wall ===")
    mergeMesh.main(dir_path)

    print(f"=== Step 5: Build initial inner stent ===")
    constructInitFD.main(dir_path, "inner")

    print(f"=== Step 6: Deploy inner stent ===")
    deployStent.main(dir_path, "inner")

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full double-stent virtual stenting pipeline."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default="experiment/experiment 0",
        help='Path to the experiment directory containing appSettings.json',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete the results/ directory instead of running the pipeline',
    )
    args = parser.parse_args()

    if args.clean:
        clean(args.experiment_dir)
    else:
        run(args.experiment_dir)
