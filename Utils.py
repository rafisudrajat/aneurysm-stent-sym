"""Vessel boundary geometry generators for virtual stenting simulations.

Provides parametric surface meshes (cylinder, cone, bent tube, S-curve,
rugged cylinder) and a composite aneurysm geometry builder.  All functions
return a triangulated PyVista PolyData mesh paired with the stent deployment
centreline as an Nx3 numpy array.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import splipy.curve_factory as sp
from PyStenting import rotate_layer


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
    y = np.zeros(len(t))
    x = np.zeros(len(t))

    points = np.append(x.reshape(len(x), 1), y.reshape(len(y), 1), axis=1)
    points = np.append(points, z.reshape(len(z), 1), axis=1)

    if hstent != height:
        start = int(0.5 * res_lon * (1 - hstent / height))
        stop = int(0.5 * res_lon * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = np.zeros((N, 3))
    for i in range(N):
        circ_nodes[i] = R * np.array([np.sin(i * sep_angle), np.cos(i * sep_angle), 0])
    circ_nodes = rotate_layer(origin, direction, circ_nodes)

    dz = height * direction / (Nz - 1)
    dz = np.array([dz for _ in range(N)])

    nodes = circ_nodes.copy()
    for i in range(1, Nz):
        nodes = np.append(nodes, circ_nodes - i * dz, axis=0)
    nodes += height * direction

    if get_inlet_outlet:
        inlet_pts = nodes[:res_ang]
        outlet_pts = nodes[len(nodes) - res_ang:]
        face_idx = np.insert(np.arange(res_ang, dtype='int'), 0, res_ang)
        # NOTE: all three prints below use inlet_pts instead of their named variable.
        # Bug is preserved intentionally; will be removed in Phase 2.
        print("inletPoints", inlet_pts)
        print("facesInletOutlet", inlet_pts)
        print("outletPoints", inlet_pts)
        dict_inlet_outlet: dict[str, pv.PolyData] = {
            "inlet": pv.PolyData(inlet_pts, face_idx),
            "outlet": pv.PolyData(outlet_pts, face_idx),
        }

    faces = np.array([])
    for i in range(Nz - 1):
        for j in range(N):
            f = np.array([4, i * N + j, i * N + (j + 1) % N,
                          (i + 1) * N + (j + 1) % N, (i + 1) * N + j])
            faces = np.append(faces, f)
    faces = faces.astype('int')
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
    y = np.zeros(len(t))
    x = np.zeros(len(t))

    points = np.append(x.reshape(len(x), 1), y.reshape(len(y), 1), axis=1)
    points = np.append(points, z.reshape(len(z), 1), axis=1)

    if hstent != height:
        start = int(50 * (1 - hstent / height))
        stop = int(50 * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = np.zeros((N, 3))
    for i in range(N):
        circ_nodes[i] = Rtop * np.array([np.sin(i * sep_angle), np.cos(i * sep_angle), 0])
    circ_nodes = rotate_layer(origin, direction, circ_nodes)

    Rf = np.linspace(1, Rbottom / Rtop, Nz)

    dz = height * direction / (Nz - 1)
    dz = np.array([dz for _ in range(N)])

    nodes = circ_nodes.copy()
    for i in range(1, Nz):
        nodes = np.append(nodes, Rf[i] * circ_nodes - i * dz, axis=0)
    nodes += height * direction

    faces = np.array([])
    for i in range(Nz - 1):
        for j in range(N):
            f = np.array([4, i * N + j, i * N + (j + 1) % N,
                          (i + 1) * N + (j + 1) % N, (i + 1) * N + j])
            faces = np.append(faces, f)
    faces = faces.astype('int')
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

    points = np.append(x.reshape(len(x), 1), y.reshape(len(y), 1), axis=1)
    points = np.append(points, z.reshape(len(z), 1), axis=1)

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

    circ_nodes = np.zeros((N, 3))
    for i in range(N):
        circ_nodes[i] = r * np.array([np.sin(i * sep_angle), np.cos(i * sep_angle), 0])

    nodes = np.array([[0, 0, 0]])
    for i in range(Nz):
        layer = rotate_layer(spline_points[i], tangents[i], circ_nodes)
        nodes = np.append(nodes, layer, axis=0)
    nodes = nodes[1:]

    if get_inlet_outlet:
        inlet_pts = nodes[:res_ang]
        outlet_pts = nodes[len(nodes) - res_ang:]
        face_idx = np.insert(np.arange(res_ang, dtype='int'), 0, res_ang).astype('int')
        dict_inlet_outlet: dict[str, pv.PolyData] = {
            "inlet": pv.PolyData(inlet_pts, face_idx),
            "outlet": pv.PolyData(outlet_pts, face_idx),
        }

    faces = np.array([])
    for i in range(Nz - 1):
        for j in range(N):
            f = np.array([4, i * N + j, i * N + (j + 1) % N,
                          (i + 1) * N + (j + 1) % N, (i + 1) * N + j])
            faces = np.append(faces, f)
    faces = faces.astype('int')
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

    points = np.append(x.reshape(len(x), 1), y.reshape(len(y), 1), axis=1)
    points = np.append(points, z.reshape(len(z), 1), axis=1)

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

    circ_nodes = np.zeros((N, 3))
    for i in range(N):
        circ_nodes[i] = r * np.array([np.sin(i * sep_angle), np.cos(i * sep_angle), 0])

    nodes = np.array([[0, 0, 0]])
    for i in range(Nz):
        layer = rotate_layer(spline_points[i], tangents[i], circ_nodes)
        nodes = np.append(nodes, layer, axis=0)
    nodes = nodes[1:]

    faces = np.array([])
    for i in range(Nz - 1):
        for j in range(N):
            f = np.array([4, i * N + j, i * N + (j + 1) % N,
                          (i + 1) * N + (j + 1) % N, (i + 1) * N + j])
            faces = np.append(faces, f)
    faces = faces.astype('int')
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
    y = np.zeros(len(t))
    x = np.zeros(len(t))

    points = np.append(x.reshape(len(x), 1), y.reshape(len(y), 1), axis=1)
    points = np.append(points, z.reshape(len(z), 1), axis=1)

    if hstent != height:
        start = int(50 * (1 - hstent / height))
        stop = int(50 * (1 + hstent / height))
        stent_centerline = points[start:stop + 1]
    else:
        stent_centerline = points.copy()

    circ_nodes = np.zeros((N, 3))
    for i in range(N):
        circ_nodes[i] = R * np.array([np.sin(i * sep_angle), np.cos(i * sep_angle), 0])
    circ_nodes = rotate_layer(origin, direction, circ_nodes)

    rng = np.random.default_rng(seed)
    Rf = 1 - maxVar * (1 - 2 * rng.random(Nz))

    dz = height * direction / (Nz - 1)
    dz = np.array([dz for _ in range(N)])

    nodes = circ_nodes.copy()
    for i in range(1, Nz):
        nodes = np.append(nodes, Rf[i] * circ_nodes - i * dz, axis=0)
    nodes += height * direction

    faces = np.array([])
    for i in range(Nz - 1):
        for j in range(N):
            f = np.array([4, i * N + j, i * N + (j + 1) % N,
                          (i + 1) * N + (j + 1) % N, (i + 1) * N + j])
            faces = np.append(faces, f)
    faces = faces.astype('int')
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()
    mesh = mesh.triangulate()
    mesh = mesh.clean()
    mesh = mesh.subdivide(Nsubdiv, subfilter='loop')
    mesh = mesh.smooth(n_iter=Nsmooth)

    return mesh, stent_centerline


def aneu_geom(
    r: float = 1,
    h: float = 20,
    hstent: float = 20,
    angle: float = 0,
    aneu_rad: float = 1.5,
    aneu_pos: float = 0.5,
    overlap: float = 0.25,
    cyl_res: int = 100,
    sph_res: int = 50,
    extension_ratio: float = 0,
    ext_res: int = 20,
    get_inlet_outlet: bool = False,
) -> dict[str, pv.PolyData | np.ndarray | None]:
    """Build a composite aneurysm geometry: parent vessel + spherical sac.

    The aneurysm sac (sphere of radius *aneu_rad*) is positioned along the
    centreline at fractional position *aneu_pos*, offset radially by
    ``r + aneu_rad − overlap``.  The sac is Boolean-merged with the parent
    vessel using surface clipping.  When *aneu_rad* is 0, only the vessel is
    returned.

    Args:
        r: Parent vessel radius. Defaults to 1.
        h: Parent vessel length. Defaults to 20.
        hstent: Stent deployment region length (centred on the vessel). Defaults to 20.
        angle: Vessel bending angle in radians; 0 produces a straight cylinder.
            Defaults to 0.
        aneu_rad: Aneurysm sac radius; 0 skips sac generation. Defaults to 1.5.
        aneu_pos: Fractional position of the sac centre along the stent centreline
            (0 = proximal end, 1 = distal end). Defaults to 0.5.
        overlap: Radial overlap between sac and vessel wall at the neck. Defaults to 0.25.
        cyl_res: Angular and longitudinal resolution of the vessel mesh. Defaults to 100.
        sph_res: Theta/phi resolution of the sphere mesh. Defaults to 50.
        extension_ratio: If > 0, add straight flow-extension tubes of length
            ``extension_ratio × 2r`` at each end (for CFD inlet/outlet boundaries).
            Defaults to 0.
        ext_res: Longitudinal resolution of the extension tubes. Defaults to 20.
        get_inlet_outlet: If ``True``, return inlet/outlet cap meshes from the
            underlying vessel generator.

    Returns:
        Dictionary with keys:
          ``"geom"`` — combined vessel + sac PolyData,
          ``"stent_centerline"`` — Nx3 deployment centreline,
          ``"inlet"`` — inlet cap mesh or ``None``,
          ``"outlet"`` — outlet cap mesh or ``None``.
    """
    dict_inlet_outlet = None
    if angle:
        if get_inlet_outlet:
            vessel, centerline_points, dict_inlet_outlet = bent_tube(
                r, angle, h=h, hstent=h, res_ang=cyl_res, res_lon=cyl_res,
                get_inlet_outlet=get_inlet_outlet)
        else:
            vessel, centerline_points = bent_tube(
                r, angle, h=h, hstent=h, res_ang=cyl_res, res_lon=cyl_res)
    else:
        if get_inlet_outlet:
            vessel, centerline_points, dict_inlet_outlet = cylinder_bound(
                r, h, hstent=h, res_ang=cyl_res, res_lon=cyl_res,
                get_inlet_outlet=get_inlet_outlet)
        else:
            vessel, centerline_points = cylinder_bound(
                r, h, hstent=h, res_ang=cyl_res, res_lon=cyl_res)

    centerline = sp.cubic_curve(centerline_points)
    t = np.linspace(centerline.start()[0], centerline.end()[0], 1000)
    centerline_points = centerline.evaluate(t)

    if hstent != h:
        start = int(500 * (1 - hstent / h))
        stop = int(500 * (1 + hstent / h))
        stent_centerline = centerline_points[start:stop + 1]
    else:
        stent_centerline = centerline_points.copy()

    if extension_ratio:
        h = extension_ratio * 2 * r

        org = centerline_points[0]
        tg = -centerline.tangent(t[0])
        inlet, _ = cylinder_bound(r, h, res_ang=cyl_res, res_lon=ext_res,
                                  origin=org, direction=tg)

        org = centerline_points[-1]
        tg = centerline.tangent(t[-1])
        outlet, _ = cylinder_bound(r, h, res_ang=cyl_res, res_lon=ext_res,
                                   origin=org, direction=tg)

    '''
    ta = int(t[0] + aneu_pos*(t[-1]-t[0]))
    tg = centerline.tangent(ta)
    d = np.array([0,-1,tg[1]/tg[2]])
    d *= (r+aneu_rad-overlap)/np.linalg.norm(d)

    aneu_center = centerline.evaluate(ta) + d
    '''

    d = np.array([0, -np.cos(angle * aneu_pos), np.sin(angle * aneu_pos)])
    d *= (r + aneu_rad - overlap) / np.linalg.norm(d)
    aneu_center = stent_centerline[int(len(stent_centerline) * aneu_pos)] + d
    sacc = pv.Sphere(radius=aneu_rad, center=aneu_center,
                     direction=d / np.linalg.norm(d),
                     theta_resolution=sph_res, phi_resolution=sph_res)

    if aneu_rad:
        geom = vessel.clip_surface(sacc, invert=False) + sacc.clip_surface(vessel, invert=bool(angle))
    else:
        geom = vessel

    geom = geom.clean()
    geom = geom.triangulate()
    geom = geom.clean()

    return {
        "geom": geom,
        "stent_centerline": stent_centerline,
        "inlet": None if dict_inlet_outlet is None else dict_inlet_outlet["inlet"],
        "outlet": None if dict_inlet_outlet is None else dict_inlet_outlet["outlet"],
    }


'''
Flow Extension

def extend_flow(surface, centerline_points, extension_ratio=10, inlet=True, outlet=True):
    init_points = surface.points
    xmax = max(init_points[:,0])
    ymax = max(init_points[:,1])
    zmax = max(init_points[:,2])
    size = max([abs(xmax-ymax),abs(xmax-zmax),abs(ymax-zmax)])

    def extend(suface,idx,reverse=False):
        origin = centerline_points[idx]
        tangent = centerline_points[idx+1]-centerline_points[idx]
        tangent *= (1-2*reverse)/np.linalg.norm(tangent)
        center = origin+tangent*size/2
        new_surface = surface-pv.Cylinder(center=center, direction=tangent,
                                          radius=size, height=size)
        tree = KDTree(new_surface.points)
        r, _ = tree.query(origin)
        new_surface = new_surface +
'''
