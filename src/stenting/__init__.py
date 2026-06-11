"""stenting — Fast Virtual Stenting simulation package.

Public API mirrors the original flat-module API so that ``import stenting``
can replace ``import PyStenting`` and ``from Utils import *`` in one step.
"""

from .centerline import VascCenterline, points2lines
from .config import ExperimentConfig, load_config
from .geometry import (
    aneu_geom,
    bent_tube,
    conical_boundary,
    cylinder_bound,
    rotate_layer,
    rugged_cylinder,
    s_curve,
)
from .io import frame
from .simulation import VirtualStenting
from .stent import FlowDiverter, Pattern, enterprise, helical, honeycomb, semienterprise

__all__ = [
    # Core classes
    "Pattern",
    "FlowDiverter",
    "VascCenterline",
    "VirtualStenting",
    # Pattern factories
    "helical",
    "semienterprise",
    "enterprise",
    "honeycomb",
    # Geometry utilities
    "rotate_layer",
    "points2lines",
    # Boundary generators
    "cylinder_bound",
    "conical_boundary",
    "bent_tube",
    "s_curve",
    "rugged_cylinder",
    "aneu_geom",
    # Animation
    "frame",
    # Config + pipeline
    "ExperimentConfig",
    "load_config",
]
