"""Parametric vessel boundary geometry generators.

All functions return a triangulated PyVista PolyData surface mesh together
with the stent deployment centreline as an (N, 3) NumPy array.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import splipy.curve_factory as sp

from .cylinder import _build_faces, _make_ring
from .transforms import rotate_layer

__all__ = [
    "points2lines",
    "cylinder_bound",
    "conical_boundary",
    "bent_tube",
    "s_curve",
    "rugged_cylinder",
]


def points2lines(points: np.ndarray) -> pv.PolyData:
    """Convert an ordered point sequence to a VTK polyline.

    Args:
        points: 3-D points array. Shape (N, 3).

    Returns:
        PolyData with all points connected as a single open polyline.
    """
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def cylinder_bound(
    R: float = 1,
    height: float = 10,
    hstent: float = 10,
    res_ang: int = 100,
    res_lon: int = 100,
    origin: np.ndarray = np.zeros(3),
    direction: np.ndarray = np.array([0, 0, 1]),
    get_inlet_outlet: bool = False,
) -> tuple[pv.PolyData, np.ndarray] | tuple[pv.PolyData, np.ndarray, dict[str, pv.PolyData]]:
    """Generate a closed cylindrical vessel surface and stent deployment centreline.

    The cylinder axis is aligned with *direction* and its base sits at *origin*.
    When *hstent* < *height*, the returned centreline is trimmed to the central
    stent-deployment region.

    Args:
        R: Cylinder radius. Defaults to 1.
        height: Total cylinder length. Defaults to 10.
        hstent: Length of the stent deployment region (centred on the vessel). Defaults to 10.
        res_ang: Number of nodes per circular ring. Defaults to 100.
        res_lon: Number of rings along the axis. Defaults to 100.
        origin: Position of the cylinder base. Defaults to ``[0, 0, 0]``.
        direction: Unit vector defining the cylinder axis. Defaults to ``[0, 0, 1]``.
        get_inlet_outlet: If ``True``, also return flat inlet/outlet cap meshes.

    Returns:
        ``(mesh, stent_centerline)`` when *get_inlet_outlet* is ``False``, or
        ``(mesh, stent_centerline, {"inlet": …, "outlet": …})`` when ``True``.
    """
    Nz = res_lon
    N = res_ang
    sep_angle = 2 * np.pi / N

    t = np.linspace(0, 1, res_lon)
    z = height * t
    points = np.column_stack([np.zeros(res_lon), np.zeros(res_lon), z])

    if hstent != height:
        start = int(0.5 * res_lon * (1 - hstent / height))
        stop = int(0.5 * res_lon * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = _make_ring(R, N, sep_angle, origin, direction)

    dz_vec = height * direction / (Nz - 1)                 # single step vector (3,)
    i_vals = np.arange(Nz)                                  # (Nz,)
    nodes = (circ_nodes[None, :, :] -
             i_vals[:, None, None] * dz_vec[None, None, :]).reshape(-1, 3)
    nodes = nodes + height * direction

    if get_inlet_outlet:
        inlet_pts = nodes[:res_ang]
        outlet_pts = nodes[len(nodes) - res_ang:]
        face_idx = np.insert(np.arange(res_ang, dtype='int'), 0, res_ang)
        dict_inlet_outlet: dict[str, pv.PolyData] = {
            "inlet": pv.PolyData(inlet_pts, face_idx),
            "outlet": pv.PolyData(outlet_pts, face_idx),
        }

    faces = _build_faces(Nz, N)
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()

    if get_inlet_outlet:
        return mesh, stent_centerline, dict_inlet_outlet
    return mesh, stent_centerline


def conical_boundary(
    Rbottom: float,
    Rtop: float,
    height: float = 10,
    hstent: float = 10,
    res_lon: int = 100,
    res_ang: int = 100,
    origin: np.ndarray = np.zeros(3),
) -> tuple[pv.PolyData, np.ndarray]:
    """Generate a conical vessel surface tapering from *Rtop* to *Rbottom*.

    Args:
        Rbottom: Radius at the distal (bottom) end.
        Rtop: Radius at the proximal (top) end.
        height: Total cone length. Defaults to 10.
        hstent: Stent deployment region length (centred). Defaults to 10.
        res_lon: Number of axial rings. Defaults to 100.
        res_ang: Number of nodes per ring. Defaults to 100.
        origin: Cone base position. Defaults to ``[0, 0, 0]``.

    Returns:
        ``(mesh, stent_centerline)`` — triangulated cone and deployment centreline.
    """
    direction = np.array([0, 0, 1])
    Nz = res_lon
    N = res_ang
    sep_angle = 2 * np.pi / N

    t = np.linspace(0, 1, 100)
    z = height * (1 - t)
    points = np.column_stack([np.zeros(100), np.zeros(100), z])

    if hstent != height:
        start = int(50 * (1 - hstent / height))
        stop = int(50 * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = _make_ring(Rtop, N, sep_angle, origin, direction)

    Rf = np.linspace(1, Rbottom / Rtop, Nz)                # (Nz,) radial scale per ring

    dz_vec = height * direction / (Nz - 1)
    i_vals = np.arange(Nz)
    nodes = (Rf[:, None, None] * circ_nodes[None, :, :] -
             i_vals[:, None, None] * dz_vec[None, None, :]).reshape(-1, 3)
    nodes = nodes + height * direction

    faces = _build_faces(Nz, N)
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()

    return mesh, stent_centerline


def bent_tube(
    r: float,
    angle: float,
    h: float = 10,
    hstent: float = 10,
    res_ang: int = 100,
    res_lon: int = 100,
    get_inlet_outlet: bool = False,
) -> tuple[pv.PolyData, np.ndarray] | tuple[pv.PolyData, np.ndarray, dict[str, pv.PolyData]]:
    """Generate a curved (bent) tubular vessel with a constant bending radius.

    The centreline is a circular arc of total arc angle *angle* (radians) and
    chord length *h*.  The bending radius is derived as R = h / angle.

    Args:
        r: Tube cross-section radius.
        angle: Total bending angle in radians.
        h: Approximate arc length (chord) of the tube. Defaults to 10.
        hstent: Stent deployment region length. Defaults to 10.
        res_ang: Number of nodes per cross-sectional ring. Defaults to 100.
        res_lon: Number of rings along the centreline. Defaults to 100.
        get_inlet_outlet: If ``True``, also return inlet/outlet cap meshes.

    Returns:
        ``(mesh, stent_centerline)`` or
        ``(mesh, stent_centerline, {"inlet": …, "outlet": …})``.
    """
    Nz = res_lon
    N = res_ang
    sep_angle = 2 * np.pi / N
    R = h / angle

    t = np.linspace(0, angle, 5)
    y = R * (1 - np.sin(t))
    z = R * np.cos(t)
    x = np.zeros(len(t))

    points = np.column_stack([x, y, z])

    if hstent != h:
        start = int(50 * (1 - hstent / h))
        stop = int(50 * (1 + hstent / h))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    centerline = sp.cubic_curve(points)

    t = np.linspace(centerline.start()[0], centerline.end()[0], Nz)
    spline_points = centerline.evaluate(t)
    tangents = centerline.tangent(t)

    angles = np.arange(N) * sep_angle
    circ_nodes = r * np.column_stack([np.sin(angles), np.cos(angles), np.zeros(N)])

    nodes = np.concatenate(
        [rotate_layer(spline_points[i], tangents[i], circ_nodes) for i in range(Nz)],
        axis=0,
    )

    if get_inlet_outlet:
        inlet_pts = nodes[:res_ang]
        outlet_pts = nodes[len(nodes) - res_ang:]
        face_idx = np.insert(np.arange(res_ang, dtype='int'), 0, res_ang).astype('int')
        dict_inlet_outlet: dict[str, pv.PolyData] = {
            "inlet": pv.PolyData(inlet_pts, face_idx),
            "outlet": pv.PolyData(outlet_pts, face_idx),
        }

    faces = _build_faces(Nz, N)
    mesh = pv.PolyData(nodes, faces)

    if get_inlet_outlet:
        return mesh, stent_centerline, dict_inlet_outlet
    return mesh, stent_centerline


def s_curve(
    A: float,
    r: float,
    height: float,
    hstent: float = 10,
    res_ang: int = 100,
    res_lon: int = 100,
) -> tuple[pv.PolyData, np.ndarray]:
    """Generate an S-shaped tubular vessel with a sinusoidal centreline.

    The centreline follows y = −A sin(2π t), z = height (1−t) for t ∈ [0, 1].

    Args:
        A: Lateral amplitude of the S-shape.
        r: Tube cross-section radius.
        height: Total axial length of the vessel.
        hstent: Stent deployment region length (centred). Defaults to 10.
        res_ang: Number of nodes per ring. Defaults to 100.
        res_lon: Number of rings along the centreline. Defaults to 100.

    Returns:
        ``(mesh, stent_centerline)`` — triangulated S-tube and deployment centreline.
    """
    Nz = res_lon
    N = res_ang
    sep_angle = 2 * np.pi / N

    t = np.linspace(0, 1, Nz)
    z = height * (1 - t)
    y = -A * np.sin(2 * np.pi * t)
    x = np.zeros(Nz)

    points = np.column_stack([x, y, z])

    if hstent != height:
        start = int(50 * (1 - hstent / height))
        stop = int(50 * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    centerline = sp.cubic_curve(points)

    t = np.linspace(centerline.start()[0], centerline.end()[0], Nz)
    spline_points = centerline.evaluate(t)
    tangents = centerline.tangent(t)

    angles = np.arange(N) * sep_angle
    circ_nodes = r * np.column_stack([np.sin(angles), np.cos(angles), np.zeros(N)])

    nodes = np.concatenate(
        [rotate_layer(spline_points[i], tangents[i], circ_nodes) for i in range(Nz)],
        axis=0,
    )

    faces = _build_faces(Nz, N)
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()

    return mesh, stent_centerline


def rugged_cylinder(
    R: float,
    maxVar: float = 0.1,
    height: float = 10,
    hstent: float = 10,
    seed: int = 1,
    Nsmooth: int = 500,
    Nsubdiv: int = 1,
    res_lon: int = 100,
    res_ang: int = 100,
    origin: np.ndarray = np.zeros(3),
) -> tuple[pv.PolyData, np.ndarray]:
    """Generate a straight cylinder with random radius perturbations per layer.

    Each axial ring is scaled by a random factor drawn from
    ``Uniform(1 − maxVar, 1 + maxVar)`` using the provided *seed* for
    reproducibility.  The surface is then subdivided and smoothed to avoid
    sharp ridges.

    Args:
        R: Nominal cylinder radius.
        maxVar: Maximum fractional radius variation (0–1). Defaults to 0.1.
        height: Total cylinder length. Defaults to 10.
        hstent: Stent deployment region length. Defaults to 10.
        seed: Random seed for the radius perturbations. Defaults to 1.
        Nsmooth: Laplacian smoothing iterations. Defaults to 500.
        Nsubdiv: Loop subdivision passes applied before smoothing. Defaults to 1.
        res_lon: Number of axial rings. Defaults to 100.
        res_ang: Number of nodes per ring. Defaults to 100.
        origin: Cylinder base position. Defaults to ``[0, 0, 0]``.

    Returns:
        ``(mesh, stent_centerline)`` — smoothed rugged cylinder and deployment centreline.
    """
    direction = np.array([0, 0, 1])
    Nz = res_lon
    N = res_ang
    sep_angle = 2 * np.pi / N

    t = np.linspace(0, 1, 100)
    z = height * (1 - t)
    points = np.column_stack([np.zeros(100), np.zeros(100), z])

    if hstent != height:
        start = int(50 * (1 - hstent / height))
        stop = int(50 * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = _make_ring(R, N, sep_angle, origin, direction)

    rng = np.random.default_rng(seed)
    Rf = 1 - maxVar * (1 - 2 * rng.random(Nz))

    dz_vec = height * direction / (Nz - 1)
    i_vals = np.arange(Nz)
    nodes = (Rf[:, None, None] * circ_nodes[None, :, :] -
             i_vals[:, None, None] * dz_vec[None, None, :]).reshape(-1, 3)
    nodes = nodes + height * direction

    faces = _build_faces(Nz, N)
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()
    mesh = mesh.subdivide(Nsubdiv, subfilter='loop')
    mesh = mesh.smooth(n_iter=Nsmooth)

    return mesh, stent_centerline
