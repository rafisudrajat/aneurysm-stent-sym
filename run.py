"""Cross-platform pipeline orchestrator — delegates to :mod:`stenting.pipeline`.

Equivalent to the original ``runSym.cmd`` + ``script/*.bat``, but runs on
both Linux and Windows.

For richer subcommand support (geometry-only, deploy-only, etc.) use the
``stenting`` CLI installed by ``uv sync`` / ``pip install -e .``::

    stenting run "experiment/experiment 0"
    stenting geometry "experiment/experiment 0"
    stenting deploy "experiment/experiment 0" --pos outer
    stenting clean "experiment/experiment 0"

This script is kept for users who prefer to run it directly without installing::

    uv run python run.py "experiment/experiment 0"
    uv run python run.py "experiment/experiment 0" --clean
    uv run python run.py "experiment/experiment 0" --single-stent
"""

from __future__ import annotations

import argparse

from stenting import pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full virtual stenting pipeline."
    )
    parser.add_argument(
        'experiment_dir',
        nargs='?',
        default="experiment/experiment 0",
        help='Path to the experiment directory containing config.json',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete the results/ directory instead of running the pipeline',
    )
    parser.add_argument(
        '--single-stent',
        action='store_true',
        help='Stop after the outer stent (skip merge and inner stent steps)',
    )
    args = parser.parse_args()

    if args.clean:
        pipeline.clean(args.experiment_dir)
    else:
        pipeline.run(args.experiment_dir, single_stent=args.single_stent)
