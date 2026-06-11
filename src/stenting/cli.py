"""CLI entry point for the ``stenting`` command.

Registered in ``pyproject.toml`` as ``[project.scripts] stenting = "stenting.cli:main"``
so that ``pip install -e .`` (or ``uv sync``) places a ``stenting`` executable on PATH.

Subcommands
-----------
``stenting run <experiment>``
    Run the full double-stent pipeline (or ``--single-stent`` for outer only).

``stenting geometry <experiment>``
    Build the aneurysm vessel geometry only (Step 1).

``stenting deploy <experiment> [--pos outer|inner|both]``
    Build stent(s) and run deployment (Steps 2–3 and/or 5–6).
    Does NOT automatically run geometry or merge; those must already be done.

``stenting clean <experiment>``
    Delete the ``results/`` directory.

Examples::

    stenting run "experiment/experiment 0"
    stenting run "experiment/experiment 0" --single-stent
    stenting geometry "experiment/experiment 0"
    stenting deploy "experiment/experiment 0" --pos outer
    stenting clean "experiment/experiment 0"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline
from .config import load_config

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    pipeline.run(args.experiment, single_stent=args.single_stent)


def _cmd_geometry(args: argparse.Namespace) -> None:
    cfg = load_config(args.experiment)
    pipeline.build_geometry(cfg, Path(args.experiment) / "results")


def _cmd_deploy(args: argparse.Namespace) -> None:
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


def _cmd_clean(args: argparse.Namespace) -> None:
    pipeline.clean(args.experiment)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stenting",
        description="Fast Virtual Stenting simulation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Run the full pipeline (geometry → outer → merge → inner)",
    )
    p_run.add_argument("experiment", help="Path to experiment directory")
    p_run.add_argument(
        "--single-stent",
        action="store_true",
        help="Stop after deploying the outer stent (skip merge and inner stent)",
    )
    p_run.set_defaults(func=_cmd_run)

    # ---- geometry ----
    p_geo = sub.add_parser("geometry", help="Build the aneurysm vessel geometry (Step 1 only)")
    p_geo.add_argument("experiment", help="Path to experiment directory")
    p_geo.set_defaults(func=_cmd_geometry)

    # ---- deploy ----
    p_dep = sub.add_parser(
        "deploy",
        help="Build stent(s) and run deployment (Steps 2–3 and/or 5–6)",
    )
    p_dep.add_argument("experiment", help="Path to experiment directory")
    p_dep.add_argument(
        "--pos",
        choices=["outer", "inner", "both"],
        default="both",
        help="Which stent position(s) to deploy (default: both)",
    )
    p_dep.set_defaults(func=_cmd_deploy)

    # ---- clean ----
    p_cl = sub.add_parser("clean", help="Delete the results/ directory")
    p_cl.add_argument("experiment", help="Path to experiment directory")
    p_cl.set_defaults(func=_cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* (or ``sys.argv[1:]``) and dispatch the selected subcommand.

    Returns:
        Exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
