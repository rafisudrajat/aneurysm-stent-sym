"""Driver script — Step 4: merge the deployed outer stent into the vessel mesh.

Combines ``vessel_EX<id>.stl`` and ``deployed_outer_stentEX<id>.stl`` into a
single mesh used as the modified vessel wall for inner-stent deployment.

Output written to ``<experiment_dir>/results/``:
  ``stented1x_vessel_EX<id>.stl`` — merged vessel + outer stent surface

The experiment ID is read from the ``"experiment_id"`` key in ``appSettings.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def _parse_config(dir_path: str) -> str:
    """Read the experiment ID from ``appSettings.json``.

    Args:
        dir_path: Experiment directory path.

    Returns:
        The ``experiment_id`` string from the config.
    """
    with open(Path(dir_path) / 'appSettings.json', 'r') as setting:
        return json.load(setting)["experiment_id"]


def main(dir_path: str) -> None:
    """Merge the vessel and deployed outer stent meshes, then save the result.

    Args:
        dir_path: Experiment directory path.  Must contain a ``results/``
            sub-directory with the vessel and stent STL files.
    """
    experiment_id = _parse_config(dir_path)
    results_dir = Path(dir_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    vessel = trimesh.load(str(results_dir / f"vessel_EX{experiment_id}.stl"))
    deployed_stent = trimesh.load(
        str(results_dir / f"deployed_outer_stentEX{experiment_id}.stl")
    )
    combined = trimesh.util.concatenate([vessel, deployed_stent])
    combined.export(file_obj=str(results_dir / f"stented1x_vessel_EX{experiment_id}.stl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge vessel and deployed outer stent into a single surface mesh."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing appSettings.json',
    )
    args = parser.parse_args()
    main(args.experiment_dir)
