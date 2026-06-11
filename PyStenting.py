"""Compatibility shim — re-exports everything from the ``stenting`` package.

The canonical implementation now lives in ``src/stenting/``.
Import ``stenting`` directly in new code; this file exists only to keep
legacy import paths working without changes.
"""

from stenting import (  # noqa: F401
    FlowDiverter,
    Pattern,
    VascCenterline,
    VirtualStenting,
    enterprise,
    frame,
    helical,
    honeycomb,
    rotate_layer,
    semienterprise,
)
