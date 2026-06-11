"""Typed configuration schema for stenting experiments.

Load with :func:`load_config`::

    cfg = load_config("experiment/experiment 0")
    print(cfg.experiment_id)          # "0"
    print(cfg.constructInitFD.outer.stent.radius)  # 1.2

The config file (``config.yaml`` preferred; falls back to ``config.json``)
may use a ``"defaults"`` key inside ``"constructInitFD"``
and ``"deployStent"`` to eliminate inner/outer duplication — only the fields
that differ need to be listed in the ``"inner"``/``"outer"`` sub-objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ExperimentConfig",
    "AneuGeomConfig",
    "AneuGeomParam",
    "InitFDConfig",
    "DeployStentConfig",
    "StentLayerConfig",
    "DeployLayerConfig",
    "PatternParam",
    "StentParam",
    "DeployPositionParam",
    "DeployParam",
    "RenderParam",
    "FilterParam",
    "load_config",
]


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FilterParam:
    """Subdivision-filter parameters for mesh refinement."""

    nsub: int
    kind: str


@dataclass
class AneuGeomParam:
    """Geometric parameters for the aneurysm vessel model."""

    r: float
    h: float
    hstent: float
    overlap: float
    aneu_rad: float
    cyl_res: int
    sph_res: int
    angle: float  # degrees; converted to radians inside pipeline
    extension_ratio: float
    ext_res: int
    get_inlet_outlet: bool = False


@dataclass
class AneuGeomConfig:
    """Top-level config for :func:`~stenting.geometry.aneurysm.aneu_geom`."""

    aneu_geom_param: AneuGeomParam
    filter: FilterParam | None = None


@dataclass
class PatternParam:
    """Stent unit-cell pattern selector."""

    name: str
    parameter: dict[str, Any] = field(default_factory=dict)


@dataclass
class StentParam:
    """Nominal stent dimensions."""

    radius: float
    height: float
    tcopy: int
    hcopy: int
    strut_radius: float
    offset_angle: float = 0.0


@dataclass
class DeployPositionParam:
    """Centreline range and spacing for the deployment initial position."""

    range: list[float] = field(default_factory=list)
    point_spacing: float = 5.0
    reverse: bool = False


@dataclass
class StentLayerConfig:
    """All parameters needed to build one crimped stent layer (inner or outer)."""

    pattern: PatternParam
    stent: StentParam
    deploy_position_param: DeployPositionParam
    filter: FilterParam | None = None


@dataclass
class DeployParam:
    """FVS spring-relaxation solver settings."""

    tol: float = 1e-5
    add_tol: float = 0.0
    step: int | None = None
    fstop: float = 1.0
    max_iter: int = 300
    alpha: float = 1.0
    verbose: bool = True
    OC: bool = True
    render_gif: bool = False
    deployment_name: str = ""


@dataclass
class RenderParam:
    """Strut-inflation render settings."""

    n: int = 5
    h: float = 1.2
    threshold: float = 2.0


@dataclass
class DeployLayerConfig:
    """All parameters needed to deploy one stent layer (inner or outer)."""

    deploy_param: DeployParam
    render_param: RenderParam | None = None
    filter: FilterParam | None = None


# ---------------------------------------------------------------------------
# Top-level dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InitFDConfig:
    """Build-stent config for both layers."""

    outer: StentLayerConfig
    inner: StentLayerConfig


@dataclass
class DeployStentConfig:
    """Deploy config for both layers."""

    outer: DeployLayerConfig
    inner: DeployLayerConfig


@dataclass
class ExperimentConfig:
    """Full validated configuration for one stenting experiment."""

    experiment_id: str
    constructAneuGeom: AneuGeomConfig
    constructInitFD: InitFDConfig
    deployStent: DeployStentConfig


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *base* deep-merged with *override*.

    Nested dicts are merged recursively; all other value types are replaced.
    Neither input dict is mutated.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_filter(raw: dict | None) -> FilterParam | None:
    if not raw:
        return None
    return FilterParam(nsub=raw["nsub"], kind=raw["kind"])


def _parse_aneu_geom_config(raw: dict) -> AneuGeomConfig:
    p = raw["aneu_geom_param"]
    return AneuGeomConfig(
        aneu_geom_param=AneuGeomParam(
            r=p["r"],
            h=p["h"],
            hstent=p["hstent"],
            overlap=p["overlap"],
            aneu_rad=p["aneu_rad"],
            cyl_res=p["cyl_res"],
            sph_res=p["sph_res"],
            angle=p["angle"],
            extension_ratio=p["extension_ratio"],
            ext_res=p["ext_res"],
            get_inlet_outlet=p.get("get_inlet_outlet", False),
        ),
        filter=_parse_filter(raw.get("filter")),
    )


def _parse_stent_layer(raw: dict) -> StentLayerConfig:
    pat = raw["pattern"]
    s = raw["stent"]
    dp = raw["deploy_position_param"]
    return StentLayerConfig(
        pattern=PatternParam(name=pat["name"], parameter=pat.get("parameter", {})),
        stent=StentParam(
            radius=s["radius"],
            height=s["height"],
            tcopy=s["tcopy"],
            hcopy=s["hcopy"],
            strut_radius=s["strut_radius"],
            offset_angle=s.get("offset_angle", 0.0),
        ),
        deploy_position_param=DeployPositionParam(
            range=dp.get("range", []),
            point_spacing=dp.get("point_spacing", 5.0),
            reverse=dp.get("reverse", False),
        ),
        filter=_parse_filter(raw.get("filter")),
    )


def _parse_init_fd_config(raw: dict) -> InitFDConfig:
    defaults = raw.get("defaults", {})
    outer = _deep_merge(defaults, raw["outer"]) if defaults else raw["outer"]
    inner = _deep_merge(defaults, raw["inner"]) if defaults else raw["inner"]
    return InitFDConfig(outer=_parse_stent_layer(outer), inner=_parse_stent_layer(inner))


def _parse_deploy_layer(raw: dict) -> DeployLayerConfig:
    dp = raw["deploy_param"]
    rp = raw.get("render_param")
    return DeployLayerConfig(
        deploy_param=DeployParam(
            tol=dp.get("tol", 1e-5),
            add_tol=dp.get("add_tol", 0.0),
            step=dp.get("step"),
            fstop=dp.get("fstop", 1.0),
            max_iter=dp.get("max_iter", 300),
            alpha=dp.get("alpha", 1.0),
            verbose=dp.get("verbose", True),
            OC=dp.get("OC", True),
            render_gif=dp.get("render_gif", False),
            deployment_name=dp.get("deployment_name", ""),
        ),
        render_param=RenderParam(n=rp["n"], h=rp["h"], threshold=rp["threshold"]) if rp else None,
        filter=_parse_filter(raw.get("filter")),
    )


def _parse_deploy_stent_config(raw: dict) -> DeployStentConfig:
    defaults = raw.get("defaults", {})
    outer = _deep_merge(defaults, raw["outer"]) if defaults else raw["outer"]
    inner = _deep_merge(defaults, raw["inner"]) if defaults else raw["inner"]
    return DeployStentConfig(outer=_parse_deploy_layer(outer), inner=_parse_deploy_layer(inner))


def load_config(experiment_dir: str | Path) -> ExperimentConfig:
    """Load and parse an experiment config from *experiment_dir*.

    Probes in order: ``config.yaml``, then ``config.json``.
    YAML is preferred because it supports inline comments.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Fully-parsed and validated :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: If neither ``config.yaml`` nor ``config.json``
            is found in *experiment_dir*.
        KeyError: If a required config key is missing.
    """
    d = Path(experiment_dir)
    for name in ("config.yaml", "config.json"):
        p = d / name
        if p.exists():
            with open(p) as fh:
                if name.endswith(".yaml"):
                    data = yaml.safe_load(fh)
                else:
                    data = json.load(fh)
            break
    else:
        raise FileNotFoundError(
            f"No config.yaml or config.json found in '{experiment_dir}'"
        )

    return ExperimentConfig(
        experiment_id=data["experiment_id"],
        constructAneuGeom=_parse_aneu_geom_config(data["constructAneuGeom"]),
        constructInitFD=_parse_init_fd_config(data["constructInitFD"]),
        deployStent=_parse_deploy_stent_config(data["deployStent"]),
    )
