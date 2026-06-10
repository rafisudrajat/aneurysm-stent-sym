"""Driver script — Step 1: build the aneurysm vessel geometry and centreline.

Reads parameters from ``<experiment_dir>/appSettings.json`` (key
``"constructAneuGeom"``), calls :func:`Utils.aneu_geom`, and saves the outputs
to ``<experiment_dir>/results/``.

Outputs written:
  ``vessel_EX<N>.stl``       — vessel surface mesh
  ``centerline_EX<N>.vtk``   — stent deployment centreline (must be .vtk for PyVista)
  ``inlet_EX<N>.stl``        — inlet cap (only if ``get_inlet_outlet`` is True)
  ``outlet_EX<N>.stl``       — outlet cap (only if ``get_inlet_outlet`` is True)

NOTE: experiment number is currently derived from the Windows-style path string
``dir_path.split('\\')[1].split()[1]``.  This breaks on Linux.
Fix is tracked in REFACTOR_PLAN.md Phase 1.
"""

import argparse
import json
import time

import numpy as np
import pyvista as pv

from Utils import *


def _parse_config(dir_path: str) -> tuple[dict, dict | bool]:
    """Load ``constructAneuGeom`` parameters from ``appSettings.json``.

    Args:
        dir_path: Path to the experiment directory containing ``appSettings.json``.

    Returns:
        ``(aneu_geom_param, filter_param)`` where *filter_param* is either a dict
        with ``nsub`` / ``kind`` keys or ``False`` if no filter is configured.
    """
    with open(dir_path + '/appSettings.json', 'r') as setting:
        data = json.load(setting)["constructAneuGeom"]
        aneu_geom_param = data["aneu_geom_param"]
        filter_param = data.get("filter", False)
        return aneu_geom_param, filter_param


def main(dir_path: str) -> None:
    """Build the aneurysm geometry and write mesh files to the results directory.

    Args:
        dir_path: Path to the experiment directory.  Must contain
            ``appSettings.json`` and a writable ``results/`` sub-directory.
    """
    t0 = time.time()
    aneu_param, filter_param = _parse_config(dir_path)
    dict_aneu_geom = aneu_geom(
        r=aneu_param['r'],
        h=aneu_param['h'],
        hstent=aneu_param['hstent'],
        overlap=aneu_param['overlap'],
        aneu_rad=aneu_param['aneu_rad'],
        cyl_res=aneu_param['cyl_res'],
        sph_res=aneu_param['sph_res'],
        angle=np.radians(aneu_param['angle']),
        extension_ratio=aneu_param['extension_ratio'],
        ext_res=aneu_param['ext_res'],
        get_inlet_outlet=aneu_param.get("get_inlet_outlet", False),
    )
    aneu = dict_aneu_geom["geom"]
    centerline_points = dict_aneu_geom["stent_centerline"]
    inlet_surface = dict_aneu_geom.get("inlet", None)
    outlet_surface = dict_aneu_geom.get("outlet", None)

    # TODO (Phase 1): replace Windows-only path parsing with pathlib.Path(dir_path).name
    experiment_number = dir_path.split('\\')[1].split()[1]

    if inlet_surface is not None and outlet_surface is not None:
        inlet_surface.save("{}/results/inlet_EX{}.stl".format(dir_path, experiment_number))
        outlet_surface.save("{}/results/outlet_EX{}.stl".format(dir_path, experiment_number))

    centerline_wrap = pv.wrap(centerline_points)
    centerline_wrap.save("{}/results/centerline_EX{}.vtk".format(dir_path, experiment_number))

    bound = aneu
    if filter_param:
        bound = aneu.subdivide(filter_param['nsub'], subfilter=filter_param['kind'])
    bound.save("{}/results/vessel_EX{}.stl".format(dir_path, experiment_number))

    tend = time.time()
    print("Finished to construct aneurism geometry with time= %.2f ms" % (tend - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build aneurysm vessel geometry for a stenting experiment."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing appSettings.json',
    )
    args = parser.parse_args()
    main(args.experiment_dir)
