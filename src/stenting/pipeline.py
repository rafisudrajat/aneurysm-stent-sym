"""End-to-end virtual stenting pipeline.

All intermediate state is derived from the typed config and the mesh files
written to ``results/``.  No pickle files are written or read; the
VirtualStenting case is rebuilt from config + saved meshes at each step.

Typical usage::

    from stenting.pipeline import run
    run("experiment/experiment 0")

Or step by step::

    from stenting.config import load_config
    from stenting.pipeline import build_geometry, build_stent, deploy_stent, merge_meshes

    cfg = load_config("experiment/experiment 0")
    rd  = Path("experiment/experiment 0") / "results"
    build_geometry(cfg, rd)
    build_stent(cfg, "outer", rd)
    deploy_stent(cfg, "outer", rd)
    merge_meshes(cfg, rd)
    build_stent(cfg, "inner", rd)
    deploy_stent(cfg, "inner", rd)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from .centerline import VascCenterline
from .config import DeployLayerConfig, ExperimentConfig, StentLayerConfig, load_config
from .geometry import aneu_geom as _aneu_geom
from .simulation import VirtualStenting
from .stent import FlowDiverter, enterprise, helical, honeycomb, semienterprise

__all__ = [
    "build_geometry",
    "build_stent",
    "deploy_stent",
    "merge_meshes",
    "run",
    "clean",
]

_PATTERN_FACTORIES = {
    "helical": helical,
    "semienterprise": semienterprise,
    "enterprise": enterprise,
    "honeycomb": honeycomb,
}


def _make_flow_diverter(cfg: StentLayerConfig) -> FlowDiverter:
    """Construct a :class:`FlowDiverter` from a :class:`StentLayerConfig`."""
    pat = cfg.pattern
    if pat.name not in _PATTERN_FACTORIES:
        raise ValueError(
            f"Unknown pattern '{pat.name}'; expected one of {list(_PATTERN_FACTORIES)}"
        )
    pattern = _PATTERN_FACTORIES[pat.name](**pat.parameter)
    s = cfg.stent
    return FlowDiverter(
        pattern,
        radius=s.radius,
        height=s.height,
        tcopy=s.tcopy,
        hcopy=s.hcopy,
        strut_radius=s.strut_radius,
        offset_angle=s.offset_angle,
    )


def _load_boundary(cfg: StentLayerConfig, pos: str, results_dir: Path, eid: str) -> pv.PolyData:
    """Load the vessel-wall boundary mesh for *pos* and optionally subdivide."""
    path = (
        results_dir / f"vessel_EX{eid}.stl"
        if pos == "outer"
        else results_dir / f"stented1x_vessel_EX{eid}.stl"
    )
    boundary = pv.read(str(path))
    if cfg.filter:
        boundary = boundary.subdivide(cfg.filter.nsub, subfilter=cfg.filter.kind)
    return boundary


def build_geometry(cfg: ExperimentConfig, results_dir: Path) -> None:
    """Step 1 — build the aneurysm vessel geometry and centreline.

    Saves to *results_dir*:

    - ``vessel_EX{id}.stl``     — vessel surface mesh
    - ``centerline_EX{id}.vtk`` — stent deployment centreline
    - ``inlet_EX{id}.stl``      — inlet cap (only when ``get_inlet_outlet`` is True)
    - ``outlet_EX{id}.stl``     — outlet cap (only when ``get_inlet_outlet`` is True)

    Args:
        cfg: Validated experiment configuration.
        results_dir: Output directory; created if it does not exist.
    """
    t0 = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.constructAneuGeom.aneu_geom_param
    eid = cfg.experiment_id

    result = _aneu_geom(
        r=p.r,
        h=p.h,
        hstent=p.hstent,
        overlap=p.overlap,
        aneu_rad=p.aneu_rad,
        cyl_res=p.cyl_res,
        sph_res=p.sph_res,
        angle=np.radians(p.angle),
        extension_ratio=p.extension_ratio,
        ext_res=p.ext_res,
        get_inlet_outlet=p.get_inlet_outlet,
    )

    if p.get_inlet_outlet:
        result["inlet"].save(str(results_dir / f"inlet_EX{eid}.stl"))
        result["outlet"].save(str(results_dir / f"outlet_EX{eid}.stl"))

    pv.wrap(result["stent_centerline"]).save(str(results_dir / f"centerline_EX{eid}.vtk"))

    bound = result["geom"]
    flt = cfg.constructAneuGeom.filter
    if flt:
        bound = bound.subdivide(flt.nsub, subfilter=flt.kind)
    bound.save(str(results_dir / f"vessel_EX{eid}.stl"))

    print(f"[geometry] done in {time.time() - t0:.2f} s")


def build_stent(cfg: ExperimentConfig, pos: str, results_dir: Path) -> None:
    """Step 2/5 — build the initial crimped stent and save its wireframe mesh.

    Saves to *results_dir*:

    - ``init_{pos}_stentEX{id}.vtp`` — initial wireframe (for visualisation)

    No pickle file is written.  :func:`deploy_stent` reconstructs the case
    from config + the mesh files already in *results_dir*.

    Args:
        cfg: Validated experiment configuration.
        pos: ``"inner"`` or ``"outer"``.
        results_dir: Must already contain ``centerline_EX{id}.vtk`` and the
            appropriate vessel STL.
    """
    if pos not in ("inner", "outer"):
        raise ValueError(f"pos must be 'inner' or 'outer', got '{pos}'")
    t0 = time.time()
    eid = cfg.experiment_id
    layer = cfg.constructInitFD.outer if pos == "outer" else cfg.constructInitFD.inner

    stent = _make_flow_diverter(layer)
    centerline_pts = pv.read(str(results_dir / f"centerline_EX{eid}.vtk")).points
    dp = layer.deploy_position_param
    centerline = VascCenterline(
        centerline_pts,
        init_range=np.array(dp.range) if dp.range else np.array([]),
        point_spacing=dp.point_spacing,
        reverse=dp.reverse,
    )
    boundary = _load_boundary(layer, pos, results_dir, eid)

    case = VirtualStenting(stent=stent, centerline=centerline, boundary=boundary)
    case.initial_stent.save(str(results_dir / f"init_{pos}_stentEX{eid}.vtp"))

    print(f"[build_stent {pos}] done in {time.time() - t0:.2f} s")


def deploy_stent(cfg: ExperimentConfig, pos: str, results_dir: Path) -> None:
    """Step 3/6 — run the FVS deployment simulation and save the result mesh.

    Rebuilds the :class:`~stenting.simulation.VirtualStenting` case from config
    plus the mesh files in *results_dir* (no pickle).

    Saves to *results_dir*:

    - ``deployed_{pos}_stentEX{id}.stl``          — when ``render_param`` is set
    - ``deployed_{pos}_stentEX{id}_norender.vtp``  — when ``render_param`` is absent

    Args:
        cfg: Validated experiment configuration.
        pos: ``"inner"`` or ``"outer"``.
        results_dir: Must contain the centreline and vessel boundary files.
    """
    if pos not in ("inner", "outer"):
        raise ValueError(f"pos must be 'inner' or 'outer', got '{pos}'")
    t0 = time.time()
    eid = cfg.experiment_id
    init_layer = cfg.constructInitFD.outer if pos == "outer" else cfg.constructInitFD.inner
    deploy_layer: DeployLayerConfig = (
        cfg.deployStent.outer if pos == "outer" else cfg.deployStent.inner
    )
    dp = deploy_layer.deploy_param

    stent = _make_flow_diverter(init_layer)
    centerline_pts = pv.read(str(results_dir / f"centerline_EX{eid}.vtk")).points
    pos_p = init_layer.deploy_position_param
    centerline = VascCenterline(
        centerline_pts,
        init_range=np.array(pos_p.range) if pos_p.range else np.array([]),
        point_spacing=pos_p.point_spacing,
        reverse=pos_p.reverse,
    )
    boundary = _load_boundary(init_layer, pos, results_dir, eid)

    case = VirtualStenting(stent=stent, centerline=centerline, boundary=boundary)
    result = case.deploy(
        tol=dp.tol,
        add_tol=dp.add_tol,
        step=dp.step,
        fstop=dp.fstop,
        max_iter=dp.max_iter,
        alpha=dp.alpha,
        verbose=dp.verbose,
        OC=dp.OC,
        render_gif=dp.render_gif,
        deployment_name=dp.deployment_name,
    )
    t1 = time.time()
    print(f"[deploy_stent {pos}] deployment done in {t1 - t0:.2f} s")

    rp = deploy_layer.render_param
    if rp:
        rendered = result.render_strut(n=rp.n, h=rp.h, threshold=rp.threshold)
        flt = deploy_layer.filter
        if flt:
            rendered = rendered.subdivide(flt.nsub, subfilter=flt.kind)
        rendered.save(str(results_dir / f"deployed_{pos}_stentEX{eid}.stl"))
        print(f"[deploy_stent {pos}] render done in {time.time() - t1:.2f} s")
    else:
        result.mesh.save(str(results_dir / f"deployed_{pos}_stentEX{eid}_norender.vtp"))


def merge_meshes(cfg: ExperimentConfig, results_dir: Path) -> None:
    """Step 4 — merge the outer stent into the vessel wall.

    Saves to *results_dir*:

    - ``stented1x_vessel_EX{id}.stl`` — combined vessel + outer stent surface

    Args:
        cfg: Validated experiment configuration.
        results_dir: Must contain ``vessel_EX{id}.stl`` and
            ``deployed_outer_stentEX{id}.stl``.
    """
    eid = cfg.experiment_id
    vessel = trimesh.load(str(results_dir / f"vessel_EX{eid}.stl"))
    deployed = trimesh.load(str(results_dir / f"deployed_outer_stentEX{eid}.stl"))
    trimesh.util.concatenate([vessel, deployed]).export(
        str(results_dir / f"stented1x_vessel_EX{eid}.stl")
    )
    print("[merge_meshes] done")


def run(experiment_dir: str | Path, single_stent: bool = False) -> None:
    """Run the full double- (or single-) stent pipeline for one experiment.

    Steps:
    1. :func:`build_geometry`
    2. :func:`build_stent` ``"outer"``
    3. :func:`deploy_stent` ``"outer"``
    4. :func:`merge_meshes`  (skipped when *single_stent* is True)
    5. :func:`build_stent` ``"inner"``  (skipped when *single_stent* is True)
    6. :func:`deploy_stent` ``"inner"``  (skipped when *single_stent* is True)

    Args:
        experiment_dir: Path to the experiment directory (must contain
            ``config.json`` or ``appSettings.json``).
        single_stent: Stop after step 3, skipping merge and inner-stent steps.
    """
    cfg = load_config(experiment_dir)
    results_dir = Path(experiment_dir) / "results"

    print("=== Step 1: Build aneurysm geometry ===")
    build_geometry(cfg, results_dir)

    print("=== Step 2: Build initial outer stent ===")
    build_stent(cfg, "outer", results_dir)

    print("=== Step 3: Deploy outer stent ===")
    deploy_stent(cfg, "outer", results_dir)

    if single_stent:
        print("=== Pipeline complete (single-stent) ===")
        return

    print("=== Step 4: Merge outer stent into vessel wall ===")
    merge_meshes(cfg, results_dir)

    print("=== Step 5: Build initial inner stent ===")
    build_stent(cfg, "inner", results_dir)

    print("=== Step 6: Deploy inner stent ===")
    deploy_stent(cfg, "inner", results_dir)

    print("=== Pipeline complete ===")


def clean(experiment_dir: str | Path) -> None:
    """Delete the ``results/`` directory for the given experiment.

    Args:
        experiment_dir: Path to the experiment directory.
    """
    results_dir = Path(experiment_dir) / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Removed {results_dir}")
    else:
        print(f"Nothing to clean: {results_dir} does not exist.")
