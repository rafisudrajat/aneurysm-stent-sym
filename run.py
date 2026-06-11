#!/usr/bin/env python
"""Entry point for the virtual stenting pipeline.

Usage
-----
Run the full double-stent pipeline (most common)::

    python run.py "experiment/experiment 0"

Other modes::

    python run.py "experiment/experiment 0" --single-stent
    python run.py "experiment/experiment 0" --clean
    python run.py "experiment/experiment 0" --geometry
    python run.py "experiment/experiment 0" --deploy
    python run.py "experiment/experiment 0" --deploy --pos outer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package,
# as long as the virtual environment's site-packages are active.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stenting import pipeline
from stenting.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Fast Virtual Stenting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("experiment", help="Path to the experiment directory")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--clean",
        action="store_true",
        help="Delete the results/ directory and exit",
    )
    mode.add_argument(
        "--geometry",
        action="store_true",
        help="Run Step 1 (vessel geometry) only",
    )
    mode.add_argument(
        "--deploy",
        action="store_true",
        help="Run stent build + deployment steps only (assumes geometry is done)",
    )

    parser.add_argument(
        "--single-stent",
        action="store_true",
        help="Stop after the outer stent (skip merge and inner stent); used with default run mode",
    )
    parser.add_argument(
        "--pos",
        choices=["outer", "inner", "both"],
        default="both",
        help="Which stent position(s) to deploy (used with --deploy; default: both)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.clean:
        pipeline.clean(args.experiment)
        return 0

    if args.geometry:
        cfg = load_config(args.experiment)
        pipeline.build_geometry(cfg, Path(args.experiment) / "results")
        return 0

    if args.deploy:
        cfg = load_config(args.experiment)
        results_dir = Path(args.experiment) / "results"
        pos: str = args.pos
        if pos in ("outer", "both"):
            pipeline.build_stent(cfg, "outer", results_dir)
            pipeline.deploy_stent(cfg, "outer", results_dir)
        if pos == "both":
            pipeline.merge_meshes(cfg, results_dir)
        if pos in ("inner", "both"):
            pipeline.build_stent(cfg, "inner", results_dir)
            pipeline.deploy_stent(cfg, "inner", results_dir)
        return 0

    # Default: full pipeline
    pipeline.run(args.experiment, single_stent=args.single_stent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
