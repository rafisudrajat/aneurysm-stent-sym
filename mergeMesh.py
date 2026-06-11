"""Driver script — Step 4: merge the deployed outer stent into the vessel mesh.

Delegates to :func:`stenting.pipeline.merge_meshes`.

Output written to ``<experiment_dir>/results/``:
  ``stented1x_vessel_EX<id>.stl`` — merged vessel + outer stent surface

Configuration is loaded from ``config.json`` in the experiment directory
(falls back to ``appSettings.json`` for backward compatibility).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stenting.config import load_config
from stenting.pipeline import merge_meshes


def main(dir_path: str) -> None:
    """Merge the vessel and deployed outer stent meshes.

    Args:
        dir_path: Path to the experiment directory.
    """
    cfg = load_config(dir_path)
    merge_meshes(cfg, Path(dir_path) / "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge vessel and deployed outer stent into a single surface mesh."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing config.json',
    )
    args = parser.parse_args()
    main(args.experiment_dir)
