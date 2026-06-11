"""FlowDiverter: wireframe stent mesh built by tiling a unit-cell pattern on a cylinder."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from ..geometry.transforms import rotate_layer
from .patterns import Pattern

__all__ = ["FlowDiverter"]


class FlowDiverter:
    """Wireframe stent mesh built by tiling a 2-D unit-cell pattern around a cylinder.

    The pattern is repeated *tcopy* times circumferentially and *hcopy* times
    longitudinally.  Optional top/bottom caps from the Pattern are added at each
    end of the main body.

    Attributes:
        mesh:      PyVista PolyData wireframe (points + line connectivity).
        lines:     N×2 array of node index pairs extracted from ``mesh.lines``.
        connected: Adjacency list — ``connected[i]`` holds the neighbours of node i.
    """

    def __init__(
        self,
        unit_cell: Pattern,
        radius: float,
        height: float,
        tcopy: int,
        hcopy: int,
        strut_radius: float = 0.05,
        centerline: object = None,
        offset_angle: float = 0,
    ) -> None:
        """Build stent wireframe and adjacency list.

        Args:
            unit_cell: Pattern defining the repeating unit cell geometry.
            radius: Nominal stent radius (mm).
            height: Total deployed stent length (mm).
            tcopy: Unit-cell copies in the circumferential direction.
            hcopy: Unit-cell copies along the stent length.
            strut_radius: Cross-section radius for strut inflation. Defaults to 0.05.
            centerline: If provided (a :class:`~stenting.centerline.VascCenterline`),
                nodes are placed along this curved path instead of a straight axis.
            offset_angle: Rotational offset in units of π, applied to the first node
                of every layer; useful for aligning overlapping double-stent patterns.
        """
        self.Pattern = unit_cell
        self.unit_cell = unit_cell.pattern
        self.size_lon = unit_cell.size_lon
        self.size_tgn = unit_cell.size_tgn
        self.radius = radius
        self.height = height
        self.tcopy = tcopy
        self.hcopy = hcopy
        self.strut_radius = strut_radius
        self.centerline = centerline

        self.bot_cap = unit_cell.bot_cap
        self.top_cap = unit_cell.top_cap
        self.bot_cap_size = unit_cell.bot_cap_size
        self.top_cap_size = unit_cell.top_cap_size

        # +1 for the mandatory base layer; cap layers are additive on top.
        self.layers: int = 1 + hcopy * (self.size_lon - 1) + self.bot_cap_size + self.top_cap_size
        # Circular topology: each cell shares one edge with its neighbour, so
        # tcopy cells need tcopy*(size_tgn-1) unique nodes, not tcopy*size_tgn.
        self.nodes_per_layer: int = self.tcopy * (self.size_tgn - 1)
        self.layer_height: float = self.height / (self.layers - 1)
        self.sep_angle: float = 2 * np.pi / self.nodes_per_layer
        self.offset_angle: float = offset_angle

        self.mesh: pv.PolyData = self.pattern_wrap(radius, centerline)
        # VTK lines array is [2, i, j, 2, k, l, …]; reshape to N×2 index pairs.
        self.lines: np.ndarray = self.mesh.lines.reshape(-1, 3)[:, 1:]
        self.connected: list[list[int]] = self.connected_list()

    def cylinder_mesh(self, R: float, centerline: object) -> pv.PolyData:
        """Create a point cloud arranged on a cylindrical surface (no face connectivity).

        Straight mode (centerline is None): rings are stacked uniformly along z.
        Curved mode: each ring is placed at a spline sample point and rotated to be
        perpendicular to the local tangent via :func:`~stenting.geometry.transforms.rotate_layer`.

        Args:
            R: Cylinder radius.
            centerline: Spline path; ``None`` produces a straight cylinder.

        Returns:
            PolyData containing only node positions.
        """
        Nz = self.layers
        N = self.nodes_per_layer
        h = self.layer_height
        sep_angle = self.sep_angle
        offset_angle = self.offset_angle * np.pi

        circ_nodes = np.zeros((N, 3))
        for i in range(N):
            circ_nodes[i] = R * np.array([
                np.cos(i * sep_angle + offset_angle),
                np.sin(i * sep_angle + offset_angle),
                0,
            ])

        if not centerline:
            dz = np.zeros((N, 3))
            dz[:, 2] = h * np.ones(N)
            nodes = circ_nodes.copy()
            for i in range(1, Nz):
                nodes = np.append(nodes, circ_nodes + i * dz, axis=0)
        else:
            c = centerline.interp
            t = np.linspace(c.start()[0], c.end()[0], Nz)
            spline_points = c.evaluate(t)
            tangents = c.tangent(t)

            nodes = np.array([[0, 0, 0]])
            for i in range(Nz):
                layer = rotate_layer(spline_points[i], tangents[i], circ_nodes)
                nodes = np.append(nodes, layer, axis=0)
            nodes = nodes[1:]

        return pv.PolyData(nodes)

    def pattern_wrap(self, R: float, centerline: object) -> pv.PolyData:
        """Tile the unit cell around and along the cylinder to produce the stent wireframe.

        Maps 2-D pattern coordinates (row, col) to 3-D node indices as
        ``index = row * N + (col % N)``, where N is *nodes_per_layer*.
        Modular arithmetic handles the circular wrap-around.

        Args:
            R: Cylinder radius forwarded to :meth:`cylinder_mesh`.
            centerline: Curved path; ``None`` for a straight cylinder.

        Returns:
            Cleaned PolyData wireframe with isolated points removed.
        """
        Nz = self.layers - self.bot_cap_size - self.top_cap_size
        N = self.nodes_per_layer
        Ni = self.size_lon
        Nj = self.size_tgn
        mesh = self.cylinder_mesh(R, centerline)

        lines = np.array([])

        if self.top_cap.any():
            i = 0
            for j in range(0, N, Nj - 1):
                cell_lines = self.top_cap.copy()
                cell_lines += np.array([i, j])
                for k in range(len(cell_lines)):
                    p0, p1 = cell_lines[k, 0], cell_lines[k, 1]
                    ind0 = p0[0] * N + (p0[1] % N)
                    ind1 = p1[0] * N + (p1[1] % N)
                    lines = np.append(lines, [2, ind0, ind1])
            lines = lines.astype('int')

        start = (self.top_cap_size - 1) * self.top_cap.any()
        for i in range(start, Nz - 1, Ni - 1):
            for j in range(0, N, Nj - 1):
                cell_lines = self.unit_cell.copy()
                cell_lines += np.array([i, j])
                for k in range(len(cell_lines)):
                    p0, p1 = cell_lines[k, 0], cell_lines[k, 1]
                    ind0 = p0[0] * N + (p0[1] % N)
                    ind1 = p1[0] * N + (p1[1] % N)
                    lines = np.append(lines, [2, ind0, ind1])
        lines = lines.astype('int')

        if self.bot_cap.any():
            i += Ni - 1
            for j in range(0, N, Nj - 1):
                cell_lines = self.bot_cap.copy()
                cell_lines += np.array([i, j])
                for k in range(len(cell_lines)):
                    p0, p1 = cell_lines[k, 0], cell_lines[k, 1]
                    ind0 = p0[0] * N + (p0[1] % N)
                    ind1 = p1[0] * N + (p1[1] % N)
                    lines = np.append(lines, [2, ind0, ind1])
            lines = lines.astype('int')

        edges = pv.PolyData()
        edges.points = mesh.points
        edges.lines = lines
        return edges.clean()

    def show(self, cpos: list[float] = [1, 0, 0]) -> None:
        """Open an interactive PyVista window showing the stent wireframe.

        Args:
            cpos: Camera position vector. Defaults to ``[1, 0, 0]`` (view from +x).
        """
        p = pv.Plotter()
        p.add_mesh(self.mesh, color='black', line_width=2)
        p.show(cpos=cpos)

    def connected_nodes(self, idx: int) -> np.ndarray:
        """Return indices of all nodes directly connected to node *idx*.

        Args:
            idx: Node index into ``self.mesh.points``.

        Returns:
            Sorted 1-D array of neighbour indices (excludes *idx* itself).
        """
        cids = [i for i, line in enumerate(self.lines) if idx in line]
        connected = np.unique([self.lines[i].ravel() for i in cids])
        return np.delete(connected, np.argwhere(connected == idx))

    def connected_list(self) -> list[list[int]]:
        """Build the full adjacency list for every node.

        Returns:
            List of length N where element i contains the neighbours of node i.
        """
        return [[p for p in self.connected_nodes(i)] for i in range(len(self.mesh.points))]

    def save(self, fname: str) -> None:
        """Write the wireframe mesh to *fname* (format inferred from file extension).

        Args:
            fname: Output path, e.g. ``"stent.vtp"``.
        """
        self.mesh.save(fname)

    def render_strut(
        self,
        n: int = 3,
        h: float = 1.2,
        threshold: int = 2,
        save_as: str | None = None,
    ) -> pv.PolyData:
        """Inflate the wireframe into solid geometry (delegates to :mod:`~stenting.stent.render`).

        Args:
            n: Sides of the strut cross-section polygon. Defaults to 3.
            h: Cross-section offset multiplier. Defaults to 1.2.
            threshold: Degenerate-face histogram bin index. Defaults to 2.
            save_as: If given, save the rendered mesh to this path.

        Returns:
            Solid PolyData mesh of all inflated struts and junctions.
        """
        from . import render
        return render.render_strut(self, n=n, h=h, threshold=threshold, save_as=save_as)
