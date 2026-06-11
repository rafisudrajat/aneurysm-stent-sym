"""Rotation helpers used when placing stent rings along a curved centreline."""

from __future__ import annotations

import numpy as np

__all__ = ["rotate_layer"]


def rotate_layer(
    origin: np.ndarray,
    tangent: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Rotate *vertices* from the z=0 plane onto the plane normal to *tangent*, then translate.

    Builds the rotation as M = M1ᵀ · M2 · M1, where:
      - M1 aligns the tangent's xy-projection with the x-axis (z-axis rotation).
      - M2 tilts by the angle between the tangent and [0,0,1] (y-axis rotation).

    Args:
        origin: Translation applied after rotation. Shape (3,).
        tangent: Normal of the target plane (need not be unit length). Shape (3,).
        vertices: Points in the z=0 plane. Shape (N, 3).

    Returns:
        Transformed points, same shape as *vertices*.
    """
    x = vertices.T
    t = tangent / np.linalg.norm(tangent)
    angle = np.arccos(np.dot(t, np.array([0, 0, 1])))

    if t[0] or t[1]:
        t /= np.linalg.norm(t[:-1])
        M1 = np.array([[t[0], -t[1], 0],
                       [t[1],  t[0], 0],
                       [0,     0,    1]])
        M2 = np.array([[ np.cos(angle), 0, np.sin(angle)],
                       [0,              1, 0             ],
                       [-np.sin(angle), 0, np.cos(angle)]])
        M = np.dot(M1.T, np.dot(M2, M1))
        x = np.dot(M, x)
        return origin + x.T
    else:
        return origin + vertices
