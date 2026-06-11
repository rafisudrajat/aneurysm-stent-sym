"""Tests for rotate_layer (stenting.geometry.transforms)."""

import numpy as np
import pytest

from stenting.geometry.transforms import rotate_layer


class TestRotateLayer:
    def test_identity_when_tangent_is_z_axis(self):
        """No rotation when tangent already points along z."""
        verts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        origin = np.zeros(3)
        result = rotate_layer(origin, np.array([0.0, 0.0, 1.0]), verts)
        np.testing.assert_allclose(result, verts, atol=1e-10)

    def test_translation_applied(self):
        """Origin is added as translation after rotation."""
        verts = np.array([[1.0, 0.0, 0.0]])
        origin = np.array([5.0, 3.0, 1.0])
        result = rotate_layer(origin, np.array([0.0, 0.0, 1.0]), verts)
        np.testing.assert_allclose(result, np.array([[6.0, 3.0, 1.0]]), atol=1e-10)

    def test_output_shape_preserved(self):
        """Output has the same shape as input vertices."""
        n = 12
        verts = np.random.default_rng(0).random((n, 3))
        result = rotate_layer(np.zeros(3), np.array([1.0, 0.0, 0.0]), verts)
        assert result.shape == (n, 3)

    def test_points_lie_on_plane_perpendicular_to_tangent(self):
        """After rotation, points lie in the plane normal to tangent."""
        tangent = np.array([0.0, 1.0, 0.0])
        r = 1.0
        n = 8
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        verts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(n)])
        result = rotate_layer(np.zeros(3), tangent, verts)
        dots = result @ (tangent / np.linalg.norm(tangent))
        np.testing.assert_allclose(dots, np.zeros(n), atol=1e-10)

    def test_distances_from_origin_preserved(self):
        """Rotation is rigid — distances from origin are unchanged."""
        tangent = np.array([1.0, 1.0, 1.0])
        r = 2.0
        n = 6
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        verts = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(n)])
        result = rotate_layer(np.zeros(3), tangent, verts)
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1), np.linalg.norm(verts, axis=1), atol=1e-10
        )
