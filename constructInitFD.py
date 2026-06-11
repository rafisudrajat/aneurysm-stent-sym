"""Driver script — Step 2: build and save the initial (crimped) stent wireframe.

Delegates to :func:`stenting.pipeline.build_stent`.

Output written to ``<experiment_dir>/results/``:
  ``init_<pos>_stentEX<id>.vtp``  — initial wireframe mesh (visualisation only)

The VirtualStenting case is no longer pickled to disk.  :mod:`deployStent`
reconstructs it at deploy time from the config + mesh files in ``results/``.

Configuration is loaded from ``config.json`` in the experiment directory
(falls back to ``appSettings.json`` for backward compatibility).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stenting.config import load_config
from stenting.pipeline import build_stent


def main(dir_path: str, stent_pos: str) -> None:
    """Build the initial crimped stent and write the wireframe mesh.

    Args:
        dir_path: Path to the experiment directory.
        stent_pos: ``"inner"`` or ``"outer"``.
    """
    cfg = load_config(dir_path)
    build_stent(cfg, stent_pos, Path(dir_path) / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the initial crimped stent wireframe."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing config.json',
    )
    parser.add_argument(
        '--stent_pos',
        type=str,
        default='outer',
        help="Stent position: 'inner' or 'outer'",
    )
    args = parser.parse_args()
    main(args.experiment_dir, args.stent_pos)
