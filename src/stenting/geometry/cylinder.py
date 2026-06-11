"""Shared low-level builders for cylindrical vessel geometry.

These are internal helpers used by :mod:`boundaries`.  They are not
part of the public API.
"""

from __future__ import annotations

import numpy as np

from .transforms import rotate_layer


def _make_ring(
    R: float,
    N: int,
    sep_angle: float,
    origin: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Return (N, 3) ring of nodes at radius *R* in the z=0 plane.

    If *origin* and *direction* are given the ring is rotated/translated so
    that its normal aligns with *direction* and its centre sits at *origin*.
    """
    angles = np.arange(N) * sep_angle
    nodes = R * np.column_stack([np.sin(angles), np.cos(angles), np.zeros(N)])
    if origin is not None and direction is not None:
        return rotate_layer(origin, direction, nodes)
    return nodes


def _build_faces(Nz: int, N: int) -> np.ndarray:
    """Return a flat VTK quad connectivity array for *Nz* rings of *N* nodes each.

    Each quad is encoded as ``[4, i0, i1, i2, i3]`` where ``4`` is the cell size.
    Rings wrap around (modular column index), producing a closed cylinder topology.
    """
    i = np.repeat(np.arange(Nz - 1), N)
    j = np.tile(np.arange(N), Nz - 1)
    quads = np.column_stack([
        np.full(len(i), 4, dtype=np.int64),
        i * N + j,
        i * N + (j + 1) % N,
        (i + 1) * N + (j + 1) % N,
        (i + 1) * N + j,
    ])
    return quads.ravel().astype('int')
