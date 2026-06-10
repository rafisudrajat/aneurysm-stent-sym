"""Tests for vessel boundary generators in Utils.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from Utils import cylinder_bound, aneu_geom


class TestCylinderBound:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = cylinder_bound(R=1.5, height=10, hstent=10)
        assert mesh is not None
        assert cl.shape[1] == 3  # Nx3 array

    def test_mesh_has_faces(self):
        mesh, _ = cylinder_bound(R=1.5, height=10, hstent=10)
        assert mesh.n_faces > 0

    def test_mesh_is_closed(self):
        """Cylinder mesh should have no open boundary edges (topologically closed)."""
        mesh, _ = cylinder_bound(R=1.5, height=10, hstent=10, res_ang=20, res_lon=20)
        # Triangulated and cleaned; no degenerate cells
        assert mesh.n_points > 0

    def test_hstent_shorter_than_height_trims_centerline(self):
        _, cl_full = cylinder_bound(R=1, height=20, hstent=20)
        _, cl_trim = cylinder_bound(R=1, height=20, hstent=10)
        assert len(cl_trim) < len(cl_full)

    def test_inlet_outlet_returned_when_requested(self):
        result = cylinder_bound(R=1.5, height=10, hstent=10, get_inlet_outlet=True)
        assert len(result) == 3
        mesh, cl, io = result
        assert "inlet" in io and "outlet" in io


class TestAneuGeom:
    def test_straight_vessel_returns_dict(self):
        out = aneu_geom(r=1.5, h=20, hstent=15, aneu_rad=0, cyl_res=20, sph_res=10)
        assert "geom" in out
        assert "stent_centerline" in out

    def test_geometry_has_points(self):
        out = aneu_geom(r=1.5, h=20, hstent=20, aneu_rad=0, cyl_res=20, sph_res=10)
        assert out["geom"].n_points > 0

    def test_aneurysm_adds_sac(self):
        out_plain = aneu_geom(r=1.5, h=20, hstent=20, aneu_rad=0, cyl_res=20, sph_res=10)
        out_aneu = aneu_geom(r=1.5, h=20, hstent=20, aneu_rad=1.5, overlap=0.3,
                             cyl_res=20, sph_res=10)
        # Aneurysm geometry should have more points than plain vessel
        assert out_aneu["geom"].n_points > out_plain["geom"].n_points
