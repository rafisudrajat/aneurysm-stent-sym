"""Strut inflation: converts a wireframe stent into solid geometry.

The :func:`render_strut` function is separated here from :class:`FlowDiverter`
because it contains non-trivial convex-hull geometry that benefits from its own
module and test coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from ..geometry.transforms import rotate_layer

if TYPE_CHECKING:
    from .flow_diverter import FlowDiverter

__all__ = ["render_strut"]


def render_strut(
    fd: FlowDiverter,
    n: int = 3,
    h: float = 1.2,
    threshold: int = 2,
    save_as: str | None = None,
) -> pv.PolyData:
    """Inflate the wireframe into solid geometry using convex-hull strut inflation.

    For each node: rotated cross-section polygons are placed along every
    connected edge, merged into a convex hull, then clipped with directional
    cones to produce smooth junctions.  The same procedure generates each
    strut body between adjacent nodes.

    Args:
        fd: Source :class:`FlowDiverter` whose wireframe is inflated.
        n: Sides of the strut cross-section polygon (3 = triangular). Defaults to 3.
        h: Offset multiplier: cross-sections are placed h × strut_radius away from
            the node centre. Defaults to 1.2.
        threshold: Histogram bin index; faces with area below bins[threshold] are
            discarded as degenerate. Defaults to 2.
        save_as: If given, save the rendered mesh to this path.

    Returns:
        Solid PolyData mesh of all inflated struts and junctions.
    """
    r = fd.strut_radius

    node_mesh = pv.PolyData([])
    line_mesh = pv.PolyData([])

    polygon = np.array([[r * np.cos(i * 2 * np.pi / n),
                         r * np.sin(i * 2 * np.pi / n), 0] for i in range(n)])
    # Small z-offset forces ConvexHull to treat this as a 3-D problem rather
    # than collapsing to a degenerate planar hull.
    polygon = np.append(np.array([[0, 0, 0.1 * r]]), polygon, axis=0)

    for idx in range(len(fd.mesh.points)):
        pref = fd.mesh.points[idx]
        cids = fd.connected[idx]

        cloud = np.zeros((1, 3))
        subt = []
        for cid in cids:
            t = fd.mesh.points[cid] - pref
            t /= np.linalg.norm(t)
            vertices = rotate_layer(pref + h * r * t, t, polygon)
            cloud = np.append(cloud, vertices, axis=0)
            cone = pv.Cone(center=pref + h * r * t, direction=-t,
                           height=2 * h * r, radius=2 * r, resolution=n)
            subt.append(cone)

        cloud = cloud[1:]
        hull = ConvexHull(cloud)
        faces = hull.simplices
        # Fixed: was np.append(3*np.ones((faces.shape[0],1), faces, axis=1).ravel())
        # which passed `faces` as dtype and `axis=1` as unknown keyword to np.ones.
        faces = np.append(3 * np.ones((faces.shape[0], 1), 'int'), faces, axis=1).ravel()

        add = pv.PolyData()
        add.points = cloud
        add.faces = faces

        for surf in subt:
            add = add.clip_surface(surf, invert=False)

        node_mesh += add

    polygon = np.array([[r * np.cos(i * 2 * np.pi / n),
                         r * np.sin(i * 2 * np.pi / n), 0] for i in range(n)])
    polygon = np.append(np.array([[0, 0, -0.1 * r]]), polygon, axis=0)

    for line in fd.lines:
        pref = [fd.mesh.points[line[0]], fd.mesh.points[line[1]]]

        cloud = np.zeros((1, 3))
        subt = []
        for i in range(2):
            t = pref[i - 1] - pref[i]
            t /= np.linalg.norm(t)
            vertices = rotate_layer(pref[i] + h * r * t, t, polygon)
            cloud = np.append(cloud, vertices, axis=0)
            cone = pv.Cone(center=pref[i] + h * r * t, direction=t,
                           height=2 * h * r, radius=2 * r, resolution=n)
            subt.append(cone)

        cloud = cloud[1:]
        hull = ConvexHull(cloud)
        faces = hull.simplices
        faces = np.append(3 * np.ones((faces.shape[0], 1), 'int'), faces, axis=1).ravel()

        add = pv.PolyData()
        add.points = cloud
        add.faces = faces

        for surf in subt:
            add = add.clip_surface(surf, invert=False)

        line_mesh += add

    strut = pv.PolyData(node_mesh + line_mesh)

    areas = strut.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    hist, bins = np.histogram(areas, bins=100)
    faces = strut.faces.reshape(-1, 4)[:, 1:]
    delete_cells = [i for i in range(len(faces)) if areas[i] < bins[threshold]]
    faces = np.delete(faces, delete_cells, axis=0)
    strut.faces = np.append(3 * np.ones((len(faces), 1), dtype='int'), faces, axis=1).ravel()

    if save_as:
        strut.save(save_as)

    return strut
