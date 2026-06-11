"""Compatibility shim — re-exports everything from ``stenting.geometry``.

The canonical implementation now lives in ``src/stenting/geometry/``.
Import ``stenting`` directly in new code; this file exists only to keep
legacy import paths working without changes.
"""

from stenting import (  # noqa: F401
    aneu_geom,
    bent_tube,
    conical_boundary,
    cylinder_bound,
    points2lines,
    rotate_layer,
    rugged_cylinder,
    s_curve,
)
