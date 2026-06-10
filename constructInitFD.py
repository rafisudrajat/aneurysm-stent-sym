"""Driver script — Step 2: build and serialise the initial (crimped) stent case.

Reads parameters from ``<experiment_dir>/appSettings.json`` (key
``"constructInitFD"`` → ``"inner"`` or ``"outer"``), constructs the stent
wireframe and crimped :class:`VirtualStenting` case, then saves:

  ``results/init_<pos>_stentEX<N>.vtp``  — initial stent wireframe mesh
  ``results/init_<pos>_stentEX<N>.obj``  — pickled :class:`VirtualStenting` case

NOTE: experiment number is currently derived from the Windows-style path string
``dir_path.split('\\')[1].split()[1]``.  This breaks on Linux.
Fix is tracked in REFACTOR_PLAN.md Phase 1.

NOTE: :func:`selectPattern` returns the ``semienterprise`` *function object*
instead of calling it (missing parentheses).  Bug preserved intentionally;
fix tracked in REFACTOR_PLAN.md §2.2.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pyvista as pv
import PyStenting as ps
from Utils import *


def _parse_config(dir_path: str, pos: str) -> tuple[str, dict, dict, dict, dict | bool]:
    """Load ``constructInitFD`` parameters for stent position *pos*.

    Args:
        dir_path: Experiment directory path.
        pos: Stent position — must be ``"inner"`` or ``"outer"``.

    Returns:
        ``(kind_FD, pattern_param, stent_param, deploy_pos_param, filter_param)``

    Raises:
        ValueError: If *pos* is not ``"inner"`` or ``"outer"``.
    """
    with open(dir_path + '/appSettings.json', 'r') as setting:
        if pos not in ("inner", "outer"):
            raise ValueError("stent position value must be either 'inner' or 'outer'")
        kind_FD = pos
        data = json.load(setting)["constructInitFD"][kind_FD]
        pattern = data["pattern"]
        stent = data["stent"]
        deploy_pos_param = data["deploy_position_param"]
        filter_param = data.get("filter", False)
        return kind_FD, pattern, stent, deploy_pos_param, filter_param


def selectPattern(name: str, param: dict = {}) -> ps.Pattern:
    """Return the :class:`~PyStenting.Pattern` for the named stent design.

    Args:
        name: One of ``"helical"``, ``"semienterprise"``, ``"enterprise"``,
            ``"honeycomb"``.
        param: Optional keyword arguments forwarded to the pattern constructor
            (e.g. ``{"size": 3}`` for helical, ``{"N": 2}`` for enterprise).

    Returns:
        Constructed :class:`~PyStenting.Pattern` instance.

    Raises:
        ValueError: If *name* is not one of the recognised pattern names.
    """
    if name not in ["helical", "semienterprise", "enterprise", "honeycomb"]:
        raise ValueError("pattern name must be either [helical, semienterprise, enterprise, honeycomb]")
    if name == "helical":
        if "size" in param.keys():
            return ps.helical(param["size"])
        return ps.helical()
    elif name == "semienterprise":
        # BUG: returns the function object, not the result of calling it.
        # Fix tracked in REFACTOR_PLAN.md §2.2.
        return ps.semienterprise
    elif name == "enterprise":
        if "N" in param.keys():
            return ps.enterprise(param["N"])
        return ps.enterprise()
    elif name == "honeycomb":
        return ps.honeycomb()


def saveFDCase(filename: str, case: ps.VirtualStenting) -> None:
    """Serialise *case* to *filename* using pickle.

    Args:
        filename: Output path (conventionally ``.obj`` extension).
        case: :class:`~PyStenting.VirtualStenting` instance to serialise.

    Warning:
        Pickle files are Python-version and object-structure specific.
        Replacement with a portable format is tracked in REFACTOR_PLAN.md Phase 3.
    """
    with open(filename, 'wb') as config_object_file:
        pickle.dump(case, config_object_file)


def main(dir_path: str, stent_pos: str) -> None:
    """Build the initial crimped stent and write mesh + case files.

    Args:
        dir_path: Experiment directory path.
        stent_pos: ``"inner"`` or ``"outer"``.
    """
    t0 = time.time()
    kind_FD, pattern_param, stent_param, deploy_pos_param, filter_param = _parse_config(
        dir_path, stent_pos
    )
    pattern = selectPattern(pattern_param["name"], pattern_param.get("parameter", {}))
    stent = ps.FlowDiverter(
        pattern,
        radius=stent_param["radius"],
        height=stent_param["height"],
        tcopy=stent_param["tcopy"],
        hcopy=stent_param["hcopy"],
        strut_radius=stent_param["strut_radius"],
        offset_angle=stent_param.get("offset_angle", 0),
    )

    # TODO (Phase 1): replace Windows-only path parsing with pathlib.Path(dir_path).name
    experiment_number = dir_path.split('\\')[1].split()[1]

    centerline_load = pv.read("{}/results/centerline_EX{}.vtk".format(dir_path, experiment_number))
    path_to_bound = (
        "{}/results/vessel_EX{}.stl".format(dir_path, experiment_number)
        if kind_FD == "outer"
        else "{}/results/stented1x_vessel_EX{}.stl".format(dir_path, experiment_number)
    )
    bound = pv.read(path_to_bound)
    if filter_param:
        bound.subdivide(filter_param['nsub'], subfilter=filter_param['kind'])

    centerline = ps.VascCenterline(
        centerline_load.points,
        init_range=deploy_pos_param.get("range", np.array([])),
        point_spacing=deploy_pos_param.get("point_spacing", 5),
        reverse=deploy_pos_param.get("reverse", False),
    )
    case = ps.VirtualStenting(stent=stent, centerline=centerline, boundary=bound)

    filename = '{}/results/init_{}_stentEX{}'.format(dir_path, kind_FD, experiment_number)
    case.initial_stent.save("{}.vtp".format(filename))
    saveFDCase('{}.obj'.format(filename), case)

    tend = time.time()
    print("Finished to construct initial stent with time= %.2f ms" % (tend - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the initial crimped stent and save the deployment case."
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='./',
        help='Path to the experiment directory containing appSettings.json',
    )
    parser.add_argument(
        '--stent_pos',
        type=str,
        default='inner',
        help="Stent position: 'inner' or 'outer'",
    )
    args = parser.parse_args()
    main(args.experiment_dir, args.stent_pos)
