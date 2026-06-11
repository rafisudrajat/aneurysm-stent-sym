"""Composite aneurysm geometry builder."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import splipy.curve_factory as sp

from .boundaries import bent_tube, cylinder_bound

__all__ = ["aneu_geom"]


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
        h_ext = extension_ratio * 2 * r

        org = centerline_points[0]
        tg = -centerline.tangent(t[0])
        inlet, _ = cylinder_bound(r, h_ext, res_ang=cyl_res, res_lon=ext_res,
                                  origin=org, direction=tg)

        org = centerline_points[-1]
        tg = centerline.tangent(t[-1])
        outlet, _ = cylinder_bound(r, h_ext, res_ang=cyl_res, res_lon=ext_res,
                                   origin=org, direction=tg)

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
