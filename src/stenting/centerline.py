"""Vascular centreline representation using a cubic B-spline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv
from splipy.curve_factory import cubic_curve

__all__ = ["VascCenterline", "points2lines"]


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


class VascCenterline:
    """Cubic B-spline representation of a vascular centreline path.

    Wraps a raw point sequence in a Splipy curve for smooth evaluation of
    positions and tangent vectors at arbitrary parameter values.

    Attributes:
        centerline_full: Full raw centreline as a VTK polyline.
        interp:          Splipy ``Curve`` object (supports ``.evaluate(t)``,
                         ``.tangent(t)``, ``.start()``, ``.end()``).
        init_segment:    The selected sub-segment as a VTK polyline.
    """

    def __init__(
        self,
        points: np.ndarray,
        init_range: np.ndarray = np.array([]),
        point_spacing: int = 5,
        reverse: bool = False,
    ) -> None:
        """Build the centreline spline, optionally restricted to a sub-segment.

        Args:
            points: Raw 3-D centreline points. Shape (N, 3).
            init_range: If non-empty, ``[start_idx, end_idx]`` (inclusive) selects
                the stent deployment sub-segment.  The full path is still stored
                in *centerline_full*.
            point_spacing: Keep every *point_spacing*-th point before spline fitting
                to reduce the control-point count.
            reverse: Reverse the point order before fitting (flips deployment direction).
        """
        self.centerline_full: pv.PolyData = points2lines(points)

        if np.asarray(init_range).size > 0:
            points = points[init_range[0]:init_range[1] + 1]

        self.interp: Any = self.interp_cl(points, point_spacing, reverse)
        self.init_segment: pv.PolyData = points2lines(points)

    def interp_cl(
        self,
        points: np.ndarray,
        point_spacing: int,
        reverse: bool,
    ) -> Any:
        """Fit a cubic B-spline through a downsampled subset of *points*.

        Args:
            points: 3-D centreline points. Shape (N, 3).
            point_spacing: Decimation stride — keep every *point_spacing*-th point.
            reverse: Reverse point order before fitting.

        Returns:
            Splipy ``Curve`` with ``.evaluate(t)`` and ``.tangent(t)`` methods.
        """
        if reverse:
            points = points[::-1]
        points = points[::point_spacing]
        return cubic_curve(points)

    def points2lines(self, points: np.ndarray) -> pv.PolyData:
        """Convert an ordered point array to a VTK polyline.

        Args:
            points: 3-D points. Shape (N, 3).

        Returns:
            PolyData with N points connected as an open polyline.
        """
        return points2lines(points)
