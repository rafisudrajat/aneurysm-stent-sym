"""Tests for vessel boundary generators in stenting.geometry.boundaries."""

import numpy as np
import pytest

from stenting import (
    aneu_geom,
    bent_tube,
    conical_boundary,
    cylinder_bound,
    rugged_cylinder,
    s_curve,
)


class TestCylinderBound:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = cylinder_bound(R=1.5, height=10, hstent=10)
        assert mesh is not None
        assert cl.shape[1] == 3

    def test_mesh_has_faces(self):
        mesh, _ = cylinder_bound(R=1.5, height=10, hstent=10)
        assert mesh.n_faces > 0

    def test_mesh_has_points(self):
        mesh, _ = cylinder_bound(R=1.5, height=10, hstent=10, res_ang=20, res_lon=20)
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

    @pytest.mark.parametrize("direction", [
        np.array([0., 0., 1.]),
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
    ])
    def test_direction_parameter(self, direction):
        mesh, _ = cylinder_bound(R=1, height=5, res_ang=10, res_lon=10, direction=direction)
        assert mesh.n_points > 0


class TestConicalBoundary:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = conical_boundary(Rbottom=0.8, Rtop=1.2, height=10)
        assert mesh is not None
        assert cl.shape[1] == 3

    def test_mesh_has_faces(self):
        mesh, _ = conical_boundary(Rbottom=0.8, Rtop=1.2, height=10)
        assert mesh.n_faces > 0

    def test_hstent_shorter_than_height_trims_centerline(self):
        _, cl_full = conical_boundary(Rbottom=0.8, Rtop=1.2, height=20, hstent=20)
        _, cl_trim = conical_boundary(Rbottom=0.8, Rtop=1.2, height=20, hstent=10)
        assert len(cl_trim) < len(cl_full)

    def test_symmetric_cone_builds(self):
        mesh, _ = conical_boundary(Rbottom=1.0, Rtop=1.0, height=5, res_lon=5, res_ang=10)
        assert mesh.n_points > 0


class TestBentTube:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = bent_tube(r=1.0, angle=0.3, h=10)
        assert mesh is not None
        assert cl.shape[1] == 3

    def test_mesh_has_faces(self):
        mesh, _ = bent_tube(r=1.0, angle=0.3, h=10, res_ang=20, res_lon=20)
        assert mesh.n_faces > 0

    def test_inlet_outlet_returned_when_requested(self):
        result = bent_tube(r=1.0, angle=0.3, h=10, get_inlet_outlet=True)
        assert len(result) == 3
        _, _, io = result
        assert "inlet" in io and "outlet" in io


class TestSCurve:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = s_curve(A=0.5, r=1.0, height=10)
        assert mesh is not None
        assert cl.shape[1] == 3

    def test_mesh_has_faces(self):
        mesh, _ = s_curve(A=0.5, r=1.0, height=10, res_ang=20, res_lon=20)
        assert mesh.n_faces > 0

    def test_zero_amplitude_builds(self):
        mesh, _ = s_curve(A=0.0, r=1.0, height=10, res_ang=10, res_lon=10)
        assert mesh.n_points > 0


class TestRuggedCylinder:
    def test_returns_mesh_and_centerline(self):
        mesh, cl = rugged_cylinder(R=1.0, seed=42, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        assert mesh is not None
        assert cl.shape[1] == 3

    def test_mesh_has_faces(self):
        mesh, _ = rugged_cylinder(R=1.0, seed=42, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        assert mesh.n_faces > 0

    def test_seed_reproducibility(self):
        """Same seed → identical output (smoothing is also deterministic)."""
        mesh1, _ = rugged_cylinder(R=1.0, seed=7, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        mesh2, _ = rugged_cylinder(R=1.0, seed=7, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        np.testing.assert_array_equal(mesh1.points, mesh2.points)

    def test_different_seeds_differ(self):
        mesh1, _ = rugged_cylinder(R=1.0, seed=1, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        mesh2, _ = rugged_cylinder(R=1.0, seed=2, Nsmooth=5, Nsubdiv=1, res_ang=10, res_lon=10)
        assert not np.allclose(mesh1.points, mesh2.points)


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
        out_aneu = aneu_geom(
            r=1.5, h=20, hstent=20, aneu_rad=1.5, overlap=0.3, cyl_res=20, sph_res=10
        )
        assert out_aneu["geom"].n_points > out_plain["geom"].n_points
