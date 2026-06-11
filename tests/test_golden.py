"""Golden regression tests: deploy() output must be stable across refactors.

On first run (no golden file): generates and saves the reference, then skips.
On subsequent runs: loads the reference and asserts allclose(atol=1e-6).
Commit tests/data/golden/deploy_tiny.npy after generating it locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
GOLDEN_FILE = GOLDEN_DIR / "deploy_tiny.npy"


def _run_tiny_deployment() -> np.ndarray:
    """Deterministic tiny deployment; returns final node positions (N, 3)."""
    from stenting import FlowDiverter, VascCenterline, VirtualStenting, cylinder_bound, helical

    mesh, cl = cylinder_bound(R=1.0, height=5.0, hstent=5.0, res_ang=10, res_lon=10)
    c = VascCenterline(cl, point_spacing=1)
    pattern = helical(size=2)
    stent = FlowDiverter(pattern, radius=0.8, height=4.5, tcopy=3, hcopy=2, centerline=c)
    sim = VirtualStenting(stent=stent, centerline=c, boundary=mesh)
    result = sim.deploy(tol=1e-4, max_iter=5, OC=False, verbose=False)
    return np.array(result.mesh.points)


def test_deploy_golden():
    """Final node positions must match the stored golden within 1e-6."""
    result = _run_tiny_deployment()

    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN_FILE, result)
        pytest.skip(
            f"Golden file generated at {GOLDEN_FILE}. "
            "Commit it, then rerun to validate."
        )

    golden = np.load(GOLDEN_FILE)
    np.testing.assert_allclose(
        result,
        golden,
        atol=1e-6,
        err_msg="deploy() output drifted from the committed golden reference.",
    )
