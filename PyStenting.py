"""Fast Virtual Stenting simulation — core geometry, stent patterns, and FVS algorithm.

Classes:
    Pattern         — 2-D unit-cell geometry (lattice coordinates).
    FlowDiverter    — Wireframe stent mesh built by tiling a Pattern on a cylinder.
    VascCenterline  — Cubic-spline wrapper around a raw vascular centreline.
    VirtualStenting — Spring-mass FVS deployment engine.

Factory functions:
    helical, semienterprise, enterprise, honeycomb  — pre-built Pattern instances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv
from splipy.curve_factory import cubic_curve
from scipy.spatial import ConvexHull, KDTree


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
        centerline: VascCenterline | None = None,
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
            centerline: If provided, nodes are placed along this curved path instead
                of a straight axis.
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

    def cylinder_mesh(self, R: float, centerline: VascCenterline | None) -> pv.PolyData:
        """Create a point cloud arranged on a cylindrical surface (no face connectivity).

        Straight mode (centerline is None): rings are stacked uniformly along z.
        Curved mode: each ring is placed at a spline sample point and rotated to be
        perpendicular to the local tangent via :func:`rotate_layer`.

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

    def pattern_wrap(self, R: float, centerline: VascCenterline | None) -> pv.PolyData:
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
        Ni = self.size_lon
        Nj = self.size_tgn
        N = self.nodes_per_layer
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
        """Inflate the wireframe into solid geometry using convex-hull strut inflation.

        For each node: rotated cross-section polygons are placed along every
        connected edge, merged into a convex hull, then clipped with directional
        cones to produce smooth junctions.  The same procedure generates each
        strut body between adjacent nodes.

        Args:
            n: Sides of the strut cross-section polygon (3 = triangular). Defaults to 3.
            h: Offset multiplier: cross-sections are placed h × strut_radius away from
                the node centre. Defaults to 1.2.
            threshold: Histogram bin index; faces with area below bins[threshold] are
                discarded as degenerate. Defaults to 2.
            save_as: If given, save the rendered mesh to this path.

        Returns:
            Solid PolyData mesh of all inflated struts and junctions.

        Note:
            The node-branch face array construction on the line marked with the
            misplaced-parentheses bug is intentionally left unchanged here.
            Fix tracked in REFACTOR_PLAN.md §2.2 for Phase 2.
        """
        r = self.strut_radius

        node_mesh = pv.PolyData([])
        line_mesh = pv.PolyData([])

        polygon = np.array([[r * np.cos(i * 2 * np.pi / n),
                             r * np.sin(i * 2 * np.pi / n), 0] for i in range(n)])
        # Small z-offset forces ConvexHull to treat this as a 3-D problem rather
        # than collapsing to a degenerate planar hull.
        polygon = np.append(np.array([[0, 0, 0.1 * r]]), polygon, axis=0)

        for idx in range(len(self.mesh.points)):
            pref = self.mesh.points[idx]
            cids = self.connected[idx]

            cloud = np.zeros((1, 3))
            subt = []
            for cid in cids:
                t = self.mesh.points[cid] - pref
                t /= np.linalg.norm(t)
                vertices = rotate_layer(pref + h * r * t, t, polygon)
                cloud = np.append(cloud, vertices, axis=0)
                cone = pv.Cone(center=pref + h * r * t, direction=-t,
                               height=2 * h * r, radius=2 * r, resolution=n)
                subt.append(cone)

            cloud = cloud[1:]
            hull = ConvexHull(cloud)
            faces = hull.simplices
            # BUG: misplaced parenthesis — second arg to np.ones should not include `faces`.
            # Correct form is at the strut loop below. See REFACTOR_PLAN.md §2.2.
            faces = np.append(3 * np.ones((faces.shape[0], 1), faces, axis=1).ravel())

            add = pv.PolyData()
            add.points = cloud
            add.faces = faces

            for surf in subt:
                add = add.clip_surface(surf, invert=False)

            node_mesh += add

        polygon = np.array([[r * np.cos(i * 2 * np.pi / n),
                             r * np.sin(i * 2 * np.pi / n), 0] for i in range(n)])
        polygon = np.append(np.array([[0, 0, -0.1 * r]]), polygon, axis=0)

        for line in self.lines:
            pref = [self.mesh.points[line[0]], self.mesh.points[line[1]]]

            cloud = np.zeros((1, 3))
            subt = []
            for i in range(2):
                t = pref[i - 1] - pref[i]
                t /= np.linalg.norm(t)
                vertices = rotate_layer(pref[i] + h * r * t, t, polygon)
                cloud = np.append(cloud, vertices, axis=0)
                cone = pv.Cone(center=pref[i] + h * r * t, direction=t,
                               height=2 * h * r, radius=2 * r, resolution=n)
                subt.append(cone)

            cloud = cloud[1:]
            hull = ConvexHull(cloud)
            faces = hull.simplices
            faces = np.append(3 * np.ones((faces.shape[0], 1), 'int'), faces, axis=1).ravel()

            add = pv.PolyData()
            add.points = cloud
            add.faces = faces

            for surf in subt:
                add = add.clip_surface(surf, invert=False)

            line_mesh += add

        strut = pv.PolyData(node_mesh + line_mesh)

        areas = strut.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
        hist, bins = np.histogram(areas, bins=100)
        faces = strut.faces.reshape(-1, 4)[:, 1:]
        delete_cells = [i for i in range(len(faces)) if areas[i] < bins[threshold]]
        faces = np.delete(faces, delete_cells, axis=0)
        strut.faces = np.append(3 * np.ones((len(faces), 1), dtype='int'), faces, axis=1).ravel()

        if save_as:
            strut.save(save_as)

        return strut


class Pattern:
    """2-D stent unit-cell geometry used to instantiate a :class:`FlowDiverter`.

    Stores line segments as lattice coordinates (row, col) for the repeating body
    and optional end caps.  :class:`FlowDiverter` maps these to 3-D positions by
    wrapping them onto a cylindrical surface.

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
        self.centerline_full: pv.PolyData = self.points2lines(points)

        if np.asarray(init_range).size > 0:
            points = points[init_range[0]:init_range[1] + 1]

        self.interp: Any = self.interp_cl(points, point_spacing, reverse)
        self.init_segment: pv.PolyData = self.points2lines(points)

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
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
        cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
        poly.lines = cells
        return poly


class VirtualStenting:
    """Fast Virtual Stenting (FVS) engine for deploying a stent inside a vessel wall.

    Models stent deployment as a spring-mass relaxation: each node is attracted
    toward its fully-expanded target position by linear springs, while a KDTree
    proximity check prevents wall penetration.

    Optimality Control (OC) scales the spring force by the ratio of the current
    to previous wall distance, damping oscillations as nodes approach the wall.

    Attributes:
        stent:         Nominal (expanded) stent geometry.
        centerline:    Deployment vessel centreline.
        boundary:      Vessel wall surface for collision detection.
        initial_stent: Crimped stent placed along the centreline.
        target_stent:  Fully expanded stent placed along the centreline.
        result:        Updated stent after :meth:`deploy` is called.
    """

    def __init__(
        self,
        stent: FlowDiverter | None = None,
        centerline: VascCenterline | None = None,
        boundary: pv.PolyData | None = None,
        initial_stent: FlowDiverter | None = None,
        target_stent: FlowDiverter | None = None,
        crimping: float = 0.2,
    ) -> None:
        """Set up the FVS case with compressed (initial) and expanded (target) states.

        Args:
            stent: Nominal expanded stent geometry.
            centerline: Centreline path inside the deployment vessel.
            boundary: Vessel wall surface mesh for collision detection.
            initial_stent: Pre-built crimped stent.  Built from *stent* and *crimping*
                if ``None``.
            target_stent: Pre-built target stent at full expansion.  Built at
                crimping=1 if ``None``.
            crimping: Radial compression ratio (0–1); 0.2 means 20 % of nominal radius.
        """
        if stent:
            self.stent = stent
            self.result = stent

        if centerline:
            self.centerline = centerline

        if boundary:
            self.boundary = boundary

        self.initial_stent: FlowDiverter = (
            initial_stent if initial_stent else self.initial(stent, centerline, crimping)
        )
        self.target_stent: FlowDiverter = (
            target_stent if target_stent else self.initial(stent, centerline, 1)
        )

    def initial(
        self,
        stent: FlowDiverter,
        c: VascCenterline,
        crimping: float,
    ) -> FlowDiverter:
        """Create a scaled copy of *stent* at radius ``stent.radius × crimping``.

        Args:
            stent: Template stent whose pattern and dimensions are reused.
            c: Centreline path for positioning the new stent.
            crimping: Radial scale factor (1 = fully expanded).

        Returns:
            New FlowDiverter at the scaled radius placed along *c*.
        """
        return FlowDiverter(
            stent.Pattern,
            stent.radius * crimping,
            stent.height,
            stent.tcopy,
            stent.hcopy,
            centerline=c,
            strut_radius=stent.strut_radius,
            offset_angle=stent.offset_angle,
        )

    def deploy(
        self,
        tol: float = 1e-5,
        add_tol: float = 0,
        step: int | None = None,
        fstop: float = 1,
        max_iter: int = 300,
        alpha: float = 1,
        verbose: bool = True,
        OC: bool = True,
        render_gif: bool = False,
        deployment_name: str = "",
    ) -> FlowDiverter:
        """Run the FVS spring-relaxation deployment simulation.

        Iterates until every node's per-step displacement drops below *tol* or
        *max_iter* is reached.  When *step* is set, layers are activated
        incrementally to simulate catheter pull-back deployment.

        Args:
            tol: Per-node displacement convergence threshold. Defaults to 1e-5.
            add_tol: Extra clearance added to strut_radius for wall contact. Defaults to 0.
            step: Number of layers to activate per incremental step.  ``None``
                deploys all layers simultaneously.
            fstop: Fraction of total layers to deploy (0–1). Defaults to 1.
            max_iter: Maximum iterations per layer batch. Defaults to 300.
            alpha: Relaxation factor applied to the spring-force update. Defaults to 1.
            verbose: Print ``(layer, iterations, max_error)`` after each batch. Defaults to True.
            OC: Apply Optimality Control force scaling. Defaults to True.
            render_gif: Capture per-iteration frames to a GIF file. Defaults to False.
            deployment_name: Output GIF filename; uses ``"Deploy.gif"`` if empty.

        Returns:
            Deployed :class:`FlowDiverter` with updated node positions.
        """
        Nz = self.stent.layers
        N = len(self.stent.mesh.points) / Nz

        result_mesh = pv.PolyData()
        result_mesh.points = self.initial_stent.mesh.points
        result_mesh.lines = self.initial_stent.mesh.lines

        if render_gif:
            frame(init_mesh=result_mesh, init=True,
                  fname="Deploy.gif" if deployment_name == "" else deployment_name)

        con_tol = self.stent.strut_radius + add_tol

        tree = KDTree(self.boundary.points)

        def proximity(point: np.ndarray) -> float:
            """Return distance from *point* to the nearest vessel-wall node."""
            d, _ = tree.query(point)
            return d

        connected = self.stent.connected
        p_ref = np.array(self.target_stent.mesh.points)
        p = np.array(result_mesh.points)
        p_new = p.copy()
        p_pred = p.copy()
        p_prev = p.copy()

        if step:
            layers = np.arange(step, int(fstop * Nz) + step, step)
            if fstop == 1:
                layers = np.append(layers, Nz)
        else:
            layers = [int(fstop * Nz)]

        for l in layers:
            err = np.ones(int(l * N))
            Niter = 0

            while max(err) > tol and Niter < max_iter:
                for i in range(int(l * N)):
                    F = 0
                    kt = 0
                    for j in connected[i]:
                        k = 1 / np.linalg.norm(p_ref[j] - p_ref[i])
                        F += k * ((p[j] - p[i]) - (p_ref[j] - p_ref[i]))
                        kt += k

                    if OC:
                        F *= proximity(p[i]) / proximity(p_prev[i])

                    p_pred[i] = p[i] + alpha * F / kt

                    if proximity(p_pred[i]) > con_tol:
                        p_new[i] = p_pred[i]

                    err[i] = np.linalg.norm(p_new[i] - p[i])
                    # NOTE: these copies are inside the per-node loop → O(N²) cost.
                    # Performance fix deferred to Phase 4 (REFACTOR_PLAN.md §4).
                    p_prev = p.copy()
                    p = p_new.copy()

                Niter += 1

            if verbose:
                print(l, Niter, max(err))

            if render_gif:
                result_mesh.points = p
                frame(mesh=result_mesh)

        if render_gif:
            frame(end=True)
        else:
            result_mesh.points = p

        self.result.mesh = result_mesh
        return self.result


# ---------------------------------------------------------------------------
# Stent pattern factory functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GIF rendering helpers
# ---------------------------------------------------------------------------

# Module-level off-screen plotter shared across all frame() calls in one run.
plotter = pv.Plotter(off_screen=True)


def frame(
    init_mesh: pv.PolyData | None = None,
    mesh: pv.PolyData | None = None,
    init: bool = False,
    end: bool = False,
    ztrans: float = 0,
    fname: str = "Deploy.gif",
) -> int:
    """Write a single frame to the deployment animation GIF.

    Call sequence: ``frame(init=True, init_mesh=vessel)`` → N × ``frame(mesh=stent)``
    → ``frame(end=True)``.

    Args:
        init_mesh: Static vessel mesh shown as a translucent overlay; required when *init=True*.
        mesh: Current stent mesh for this frame; required for data frames.
        init: Open the GIF file and add the vessel overlay.
        end: Flush and close the GIF file.
        ztrans: Unused translation parameter (reserved for future camera animation).
        fname: Output GIF path. Defaults to ``"Deploy.gif"``.

    Returns:
        Always 0.
    """
    if init:
        plotter.open_gif(fname)
        plotter.add_mesh(init_mesh, color='b', opacity=0.1)
    else:
        actor = plotter.add_mesh(mesh)
        plotter.write_frame()
        plotter.remove_actor(actor)
    if end:
        plotter.close()
    return 0
