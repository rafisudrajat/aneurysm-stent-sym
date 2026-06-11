"""Driver script — Step 1: build the aneurysm vessel geometry and centreline.

Delegates to :func:`stenting.pipeline.build_geometry`.

Outputs written to ``<experiment_dir>/results/``:
  ``vessel_EX<id>.stl``       — vessel surface mesh
  ``centerline_EX<id>.vtk``   — stent deployment centreline
  ``inlet_EX<id>.stl``        — inlet cap (only if ``get_inlet_outlet`` is True)
  ``outlet_EX<id>.stl``       — outlet cap (only if ``get_inlet_outlet`` is True)

Configuration is loaded from ``config.json`` in the experiment directory
(falls back to ``appSettings.json`` for backward compatibility).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stenting.config import load_config
from stenting.pipeline import build_geometry


def main(dir_path: str) -> None:
    """Build the aneurysm geometry and write mesh files.

    Args:
        dir_path: Path to the experiment directory.
    """
    cfg = load_config(dir_path)
    build_geometry(cfg, Path(dir_path) / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build aneurysm vessel geometry for a stenting experiment."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing config.json',
    )
    args = parser.parse_args()
    main(args.experiment_dir)
