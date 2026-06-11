"""Smoke tests — quick checks that the package imports and core objects instantiate.

These catch import-time regressions immediately.
See test_golden.py for the full deployment regression.
"""

import numpy as np


def test_import_stenting():
    import stenting  # noqa: F401


def test_public_api_importable():
    from stenting import (  # noqa: F401
        FlowDiverter,
        Pattern,
        VascCenterline,
        VirtualStenting,
        enterprise,
        helical,
        honeycomb,
        semienterprise,
    )


def test_helical_pattern_creates_without_error():
    from stenting import helical

    p = helical(size=2)
    assert p is not None


def test_flow_diverter_straight_creates_without_error():
    from stenting import FlowDiverter, helical

    pattern = helical(size=2)
    fd = FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)
    assert fd.mesh is not None


def test_flow_diverter_curved_creates_without_error():
    """FlowDiverter placed along a VascCenterline (curved mode)."""
    from stenting import FlowDiverter, VascCenterline, helical

    pts = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 10, 20)])
    c = VascCenterline(pts, point_spacing=2)
    pattern = helical(size=2)
    fd = FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3, centerline=c)
    assert fd.mesh is not None


def test_vasc_centerline_creates_without_error():
    from stenting import VascCenterline

    pts = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 10, 20)])
    cl = VascCenterline(pts, point_spacing=2)
    assert cl.interp is not None


def test_cylinder_bound_creates_without_error():
    from stenting import cylinder_bound

    mesh, cl = cylinder_bound(R=1.5, height=10, hstent=10, res_ang=20, res_lon=20)
    assert mesh is not None
    assert cl is not None


def test_tiny_deploy_runs():
    """End-to-end: build vessel + stent, deploy 5 iterations, confirm nodes moved."""
    from stenting import FlowDiverter, VascCenterline, VirtualStenting, cylinder_bound, helical

    mesh, cl = cylinder_bound(R=1.0, height=5.0, hstent=5.0, res_ang=10, res_lon=10)
    c = VascCenterline(cl, point_spacing=1)
    pattern = helical(size=2)
    stent = FlowDiverter(pattern, radius=0.8, height=4.5, tcopy=3, hcopy=2, centerline=c)
    sim = VirtualStenting(stent=stent, centerline=c, boundary=mesh)
    initial_pts = np.array(sim.initial_stent.mesh.points).copy()
    result = sim.deploy(tol=1e-4, max_iter=5, OC=False, verbose=False)
    final_pts = np.array(result.mesh.points)
    assert not np.allclose(initial_pts, final_pts), "deploy() left all nodes unmoved"
