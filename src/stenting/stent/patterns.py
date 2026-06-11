"""Stent unit-cell pattern definitions and factory functions.

A :class:`Pattern` stores a 2-D lattice description of one repeating cell of
a stent mesh.  The factory functions build the pre-defined clinical patterns.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Pattern",
    "helical",
    "semienterprise",
    "enterprise",
    "honeycomb",
]


class Pattern:
    """2-D stent unit-cell geometry used to instantiate a :class:`~stenting.stent.flow_diverter.FlowDiverter`.

    Stores line segments as lattice coordinates (row, col) for the repeating body
    and optional end caps.  :class:`~stenting.stent.flow_diverter.FlowDiverter` maps
    these to 3-D positions by wrapping them onto a cylindrical surface.

    Attributes:
        size_lon:      Number of node rows in one unit cell (longitudinal extent).
        size_tgn:      Number of node columns in one unit cell (circumferential extent).
        bot_cap_size:  Row height of the distal end cap (0 if absent).
        top_cap_size:  Row height of the proximal end cap (0 if absent).
    """

    def __init__(
        self,
        pattern: np.ndarray = np.array([]),
        bot_cap: np.ndarray = np.array([]),
        top_cap: np.ndarray = np.array([]),
    ) -> None:
        """Store pattern geometry and compute bounding dimensions.

        Args:
            pattern: Line segments of the repeating body. Shape (M, 2, 2) where each
                entry is ``[[row0, col0], [row1, col1]]``.
            bot_cap: Line segments for the distal (bottom) end cap. Same shape.
            top_cap: Line segments for the proximal (top) end cap. Same shape.
        """
        self.pattern = pattern
        self.bot_cap = bot_cap
        self.top_cap = top_cap

        if pattern.any():
            # +1 because max() returns the highest 0-based index.
            self.size_lon: int = 1 + pattern[:, :, 0].max()
            self.size_tgn: int = 1 + pattern[:, :, 1].max()

        self.bot_cap_size: int = (1 + bot_cap[:, :, 0].max()) if bot_cap.any() else 0
        self.top_cap_size: int = (1 + top_cap[:, :, 0].max()) if top_cap.any() else 0


def helical(size: int) -> Pattern:
    """Build a helical (PED/Silk-style) braided stent unit cell.

    Generates a diamond lattice with *size* - 1 repeats: each repeat adds a
    primary-diagonal strut (i,i)→(i+1,i+1) and an anti-diagonal strut
    (i, size-1-i)→(i+1, size-i-2).

    Args:
        size: Node count per layer in the tangential direction of one unit cell.
            Larger values produce finer braid density.

    Returns:
        Pattern with no end caps (helical pattern is periodically continuous).
    """
    unit_cell_lines = np.array([[[0, 0], [0, 0]]])
    for i in range(size - 1):
        unit_cell_lines = np.append(unit_cell_lines,
                                    [[[i, i], [i + 1, i + 1]]], axis=0)
        unit_cell_lines = np.append(unit_cell_lines,
                                    [[[i, size - 1 - i], [i + 1, size - i - 2]]], axis=0)
    return Pattern(pattern=unit_cell_lines)


def semienterprise() -> Pattern:
    """Build a semi-enterprise (tilted rectangular) stent unit cell.

    Returns:
        Pattern with a distal end cap.
    """
    pattern_lines = np.array([
        [[0, 0], [1, 1]],
        [[2, 0], [1, 1]],
        [[1, 1], [0, 2]],
    ])
    pattern_cap = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 2]]])
    return Pattern(pattern=pattern_lines, bot_cap=pattern_cap)


def enterprise(N: int = 1) -> Pattern:
    """Build an enterprise-style stent unit cell.

    Args:
        N: Complexity level; 1 = 6 struts per cell, 2 = 12 struts per cell.
            Any value other than 1 or 2 defaults to N=2.

    Returns:
        Pattern with both proximal and distal end caps.
    """
    if N != 1 and N != 2:
        return enterprise(2)
    elif N == 1:
        pattern_lines = np.array([[[0, 0], [1, 1]],
                                   [[1, 1], [0, 2]],
                                   [[0, 2], [1, 3]],
                                   [[2, 0], [1, 1]],
                                   [[2, 2], [1, 3]],
                                   [[1, 3], [2, 4]]])
        bot_cap = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 2]]])
        top_cap = np.array([[[1, 2], [0, 3]], [[0, 3], [1, 4]]])
    else:
        pattern_lines = np.array([[[0, 0], [1, 1]],
                                   [[1, 1], [2, 2]],
                                   [[2, 2], [1, 3]],
                                   [[1, 3], [0, 4]],
                                   [[0, 4], [1, 5]],
                                   [[1, 5], [2, 6]],
                                   [[4, 0], [3, 1]],
                                   [[3, 1], [2, 2]],
                                   [[4, 4], [3, 5]],
                                   [[3, 5], [2, 6]],
                                   [[2, 6], [3, 7]],
                                   [[3, 7], [4, 8]]])
        bot_cap = np.array([[[0, 0], [1, 1]],
                             [[1, 1], [2, 2]],
                             [[2, 2], [1, 3]],
                             [[1, 3], [0, 4]]])
        top_cap = np.array([[[2, 4], [1, 5]],
                             [[1, 5], [0, 6]],
                             [[0, 6], [1, 7]],
                             [[1, 7], [2, 8]]])
    return Pattern(pattern=pattern_lines, bot_cap=bot_cap, top_cap=top_cap)


def honeycomb() -> Pattern:
    """Build a hexagonal (honeycomb) stent unit cell.

    Returns:
        Pattern with a distal end cap.
    """
    pattern_lines = np.array([[[2, 0], [0, 1]],
                               [[0, 1], [0, 2]],
                               [[0, 2], [2, 3]],
                               [[2, 3], [2, 4]],
                               [[2, 3], [4, 2]],
                               [[4, 1], [2, 0]]])
    pattern_cap = np.array([[[0, 1], [0, 2]]])
    return Pattern(pattern=pattern_lines, bot_cap=pattern_cap)
