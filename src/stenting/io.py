"""I/O helpers: GIF animation frame writer."""

from __future__ import annotations

import pyvista as pv

__all__ = ["frame"]

# Lazily-initialised off-screen plotter.  Created on the first frame(init=True)
# call so that importing this module does not instantiate PyVista at import time.
_plotter: pv.Plotter | None = None


def frame(
    init_mesh: pv.PolyData | None = None,
    mesh: pv.PolyData | None = None,
    init: bool = False,
    end: bool = False,
    ztrans: float = 0,
    fname: str = "Deploy.gif",
) -> int:
    """Write a single frame to the deployment animation GIF.

    Call sequence: ``frame(init=True, init_mesh=vessel)`` → N × ``frame(mesh=stent)``
    → ``frame(end=True)``.

    Args:
        init_mesh: Static vessel mesh shown as a translucent overlay; required when *init=True*.
        mesh: Current stent mesh for this frame; required for data frames.
        init: Open the GIF file and add the vessel overlay.
        end: Flush and close the GIF file.
        ztrans: Unused translation parameter (reserved for future camera animation).
        fname: Output GIF path. Defaults to ``"Deploy.gif"``.

    Returns:
        Always 0.
    """
    global _plotter
    if init:
        _plotter = pv.Plotter(off_screen=True)
        _plotter.open_gif(fname)
        _plotter.add_mesh(init_mesh, color='b', opacity=0.1)
    else:
        actor = _plotter.add_mesh(mesh)
        _plotter.write_frame()
        _plotter.remove_actor(actor)
    if end:
        _plotter.close()
        _plotter = None
    return 0
