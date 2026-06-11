"""Driver script — Step 3/6: run the FVS deployment simulation and render struts.

Delegates to :func:`stenting.pipeline.deploy_stent`.

Outputs written to ``<experiment_dir>/results/``:
  ``deployed_<pos>_stentEX<id>.stl``          — rendered solid stent (if render_param set)
  ``deployed_<pos>_stentEX<id>_norender.vtp`` — wireframe stent (if render_param absent)

The VirtualStenting case is rebuilt directly from the config and mesh files in
``results/``; no ``.obj`` pickle file is needed or written.

Configuration is loaded from ``config.json`` in the experiment directory
(falls back to ``appSettings.json`` for backward compatibility).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stenting.config import load_config
from stenting.pipeline import deploy_stent


def main(dir_path: str, stent_pos: str) -> None:
    """Run deployment for one stent position and write the result mesh.

    Args:
        dir_path: Path to the experiment directory.
        stent_pos: ``"inner"`` or ``"outer"``.
    """
    cfg = load_config(dir_path)
    deploy_stent(cfg, stent_pos, Path(dir_path) / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the FVS deployment simulation and optionally render struts."
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
