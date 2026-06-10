"""Driver script — Step 4: merge the deployed outer stent into the vessel mesh.

Combines ``vessel_EX<N>.stl`` and ``deployed_outer_stentEX<N>.stl`` into a
single mesh used as the modified vessel wall for inner-stent deployment.

Output written to ``<experiment_dir>/results/``:
  ``stented1x_vessel_EX<N>.stl`` — merged vessel + outer stent surface

NOTE: experiment number is currently derived from the Windows-style path string
``dir_path.split('\\')[1].split()[1]``.  This breaks on Linux.
Fix is tracked in REFACTOR_PLAN.md Phase 1.
"""

from __future__ import annotations

import argparse

import trimesh


def main(dir_path: str) -> None:
    """Merge the vessel and deployed outer stent meshes, then save the result.

    Args:
        dir_path: Experiment directory path.  Must contain a ``results/``
            sub-directory with the vessel and stent STL files.
    """
    # TODO (Phase 1): replace Windows-only path parsing with pathlib.Path(dir_path).name
    experiment_number = dir_path.split('\\')[1].split()[1]

    vessel = trimesh.load("{}/results/vessel_EX{}.stl".format(dir_path, experiment_number))
    deployed_stent = trimesh.load(
        "{}/results/deployed_outer_stentEX{}.stl".format(dir_path, experiment_number)
    )
    combined = trimesh.util.concatenate([vessel, deployed_stent])
    combined.export(
        file_obj="{}/results/stented1x_vessel_EX{}.stl".format(dir_path, experiment_number)
    )


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
