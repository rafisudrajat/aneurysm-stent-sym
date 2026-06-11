"""Fast Virtual Stenting (FVS) spring-mass deployment simulation."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

from .centerline import VascCenterline
from .stent.flow_diverter import FlowDiverter

__all__ = ["VirtualStenting"]


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
        from .io import frame

        Nz = self.stent.layers
        N = len(self.stent.mesh.points) / Nz

        result_mesh = pv.PolyData()
        result_mesh.points = self.initial_stent.mesh.points.copy()
        result_mesh.lines = self.initial_stent.mesh.lines

        if render_gif:
            frame(init_mesh=result_mesh, init=True,
                  fname="Deploy.gif" if deployment_name == "" else deployment_name)

        con_tol = self.stent.strut_radius + add_tol
        tree = KDTree(self.boundary.points)

        p_ref = np.array(self.target_stent.mesh.points)
        p = np.array(result_mesh.points)
        n_total = len(p)

        # Precompute spring-constant sparse matrix K (constant throughout deployment).
        # K[i,j] = 1 / ||p_ref[i] - p_ref[j]||; symmetric.
        # Spring force on node i: F_i = (K @ p)[i] - kt[i]*p[i] - C[i]
        edges = self.stent.lines                              # (E, 2)
        valid = edges[:, 0] != edges[:, 1]                   # drop degenerate self-loops
        ei, ej = edges[valid, 0], edges[valid, 1]
        k_vals = 1.0 / np.linalg.norm(p_ref[ej] - p_ref[ei], axis=1)
        K = csr_matrix(
            (np.tile(k_vals, 2), (np.r_[ei, ej], np.r_[ej, ei])),
            shape=(n_total, n_total),
        )
        kt_vec = np.asarray(K.sum(axis=1)).ravel()           # (n_total,)
        C = K @ p_ref - kt_vec[:, None] * p_ref              # (n_total, 3) constant term

        p_prev = p.copy()

        if step:
            layers = np.arange(step, int(fstop * Nz) + step, step)
            if fstop == 1:
                layers = np.append(layers, Nz)
        else:
            layers = [int(fstop * Nz)]

        for layer in layers:
            n_active = int(layer * N)
            err = np.ones(n_active)
            Niter = 0

            while max(err) > tol and Niter < max_iter:
                # Vectorized spring force for all active nodes — O(E) sparse matvec.
                F = (K[:n_active] @ p) - kt_vec[:n_active, None] * p[:n_active] - C[:n_active]

                if OC:
                    # Batch wall-distance queries; scale force by current/prev ratio.
                    d_curr, _ = tree.query(p[:n_active])
                    d_prev_arr, _ = tree.query(p_prev[:n_active])
                    F *= (d_curr / np.maximum(d_prev_arr, 1e-12))[:, None]

                p_cand = p[:n_active] + alpha * F / kt_vec[:n_active, None]

                # Batch contact check; only accept moves that stay clear of wall.
                d_cand, _ = tree.query(p_cand)
                moved_idx = np.where(d_cand > con_tol)[0]

                p_prev[:] = p                     # save pre-update state for next OC
                p[moved_idx] = p_cand[moved_idx]  # in-place Jacobi update

                err = np.linalg.norm(p[:n_active] - p_prev[:n_active], axis=1)
                Niter += 1

            if verbose:
                print(layer, Niter, max(err))

            if render_gif:
                result_mesh.points = p.copy()
                frame(mesh=result_mesh)

        if render_gif:
            frame(end=True)
        else:
            result_mesh.points = p

        self.result.mesh = result_mesh
        return self.result
