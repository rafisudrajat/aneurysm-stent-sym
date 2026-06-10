"""
Smoke tests — can we import the modules and instantiate the core classes?
These are the fastest sanity checks; they run with no heavy computation.

NOTE: Full pipeline golden tests (end-to-end) are added in Phase 1 once the
Linux path-parsing bug is fixed and the pipeline can run on this OS.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_import_pystenting():
    import PyStenting  # noqa: F401


def test_import_utils():
    import Utils  # noqa: F401


def test_pattern_classes_importable():
    from PyStenting import Pattern, FlowDiverter, VascCenterline, VirtualStenting  # noqa: F401


def test_helical_pattern_creates_without_error():
    import PyStenting as ps

    p = ps.helical(size=2)
    assert p is not None


def test_flow_diverter_creates_without_error():
    import PyStenting as ps

    pattern = ps.helical(size=2)
    fd = ps.FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)
    assert fd.mesh is not None


def test_vasc_centerline_creates_without_error():
    import numpy as np
    import PyStenting as ps

    pts = np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 10, 20)])
    cl = ps.VascCenterline(pts, point_spacing=2)
    assert cl.interp is not None


def test_cylinder_bound_creates_without_error():
    from Utils import cylinder_bound

    mesh, cl = cylinder_bound(R=1.5, height=10, hstent=10, res_ang=20, res_lon=20)
    assert mesh is not None
    assert cl is not None
