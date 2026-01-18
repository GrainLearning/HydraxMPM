# # Copyright (c) 2024, Retiefasuarus
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# # -*- coding: utf-8 -*-
"""
Explanation:

    This module implements the USL-AFLIP solver for the Material Point Method (MPM).

    The USL-AFLIP has a state `USLAFLIPState` and solver logic `USLAFLIP`.

    Additional features:
        - **AFLIP/APIC**: Preserves angular momentum and reduces dissipation compared to PIC.
        - **CFL Limiting**: Clamps particle velocity to a fraction of cell_size/dt.
        - **Dynamic Alpha**: Blends FLIP/PIC based on particle support to prevent instability in empty cells.
        - **Position Correction**: Uses separate velocity fields for position update to avoid the "positional trap".


    References:
    - Fei, Yun, et al. "Revisiting integration in the material point method: a scheme for easier separation and less dissipation."
    - Jiang, Chenfanfu, et al. "The affine particle-in-cell method."

"""


import equinox as eqx

from typing import Self, Tuple

import jax.numpy as jnp

from ..grid.grid import GridState
from ..material_points.material_points import MaterialPointState

from .solver import BaseSolver, BaseSolverState

from ..common.simstate import SimState

from ..forces.force import Force

from .coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw

from ..shapefunctions.mapping import InteractionCache

from ..sdf.sdfobject import SDFObjectBase

from jaxtyping import Float, Array

from typing import Tuple, Optional

from .usl import USLSolver


class USLAFLIPState(BaseSolverState):
    """
    State required for the AFLIP/APIC solver.

    Attributes:
        Bp_stack: Affine momentum matrices for each material point.
    """

    Bp_stack: Float[Array, "num_points 3 3"]


class USLAFLIP(USLSolver):
    """
    Update Stress Last (USL) Solver with Affine-FLIP (AFLIP/APIC) transfer.

    Position Correction (Separable S-FLIP) uses beta_p which mixes the FLIP and PIC velocities
    specifically for the position update.

    Dynamic Alpha reduces the FLIP component when particles have low grid support.

    CFL Limiting clamps particle velocities to a fraction of cell_size/dt to ensure stability.

    Attributes:
        alpha: Blending factor between FLIP and PIC updates (0.0 = pure FLIP, 1.0 = pure PIC) (default 0.99).
        use_dynamic_alpha: Enables dynamic alpha scaling based on particle support (default True).
        alpha_support_min: Minimum particle support ratio for dynamic alpha scaling (default 1.1).
        alpha_support_max: Maximum particle support ratio for dynamic alpha scaling (default 1.5).
        cfl_limit: Max fraction of cell_size per step (default 0.5)
        beta_min: Mixes FLIP/PIC based on particle support (minimum), for position update (default 1.0).
        beta_max: Mixes FLIP/PIC based on particle support (maximum), for position update (default 0.5).
        small_mass_cutoff: Prevents updates from grid to particles if masses are too small (default 1e-7).
    """

    # FLIP/PIC blending ratio
    alpha: float = eqx.field(static=True)

    # Dynamic Alpha
    use_dynamic_alpha: bool = eqx.field(static=True)
    alpha_support_min: float = eqx.field(static=True, default=1.1)
    alpha_support_max: float = eqx.field(static=True, default=1.5)

    # Position Correction (Separable S-FLIP)
    beta_min: float = eqx.field(static=True, default=0.0)
    beta_max: float = eqx.field(static=True, default=0.5)

    # CFL Condition
    cfl_limit: float = eqx.field(static=True, default=0.5)

    small_mass_cutoff: float = eqx.field(static=True, default=1e-7)

    # Switch between MLS and APIC update for velocity gradient and affine matrices
    use_mls_update: bool = eqx.field(static=True)

    # Logic operations
    couplings: Tuple[BodyCoupling, ...]
    constitutive_laws: Tuple[ConstitutiveLaw, ...]
    forces: Tuple[Force, ...]
    sdf_logics: Tuple[SDFObjectBase, ...]

    active_p_ids: Tuple[int, ...] = eqx.field(static=True)
    active_g_ids: Tuple[int, ...] = eqx.field(static=True)

    def create_state(self, mp_state) -> Self:
        """Creates empty state with affine matrices"""
        return USLAFLIPState(Bp_stack=jnp.zeros((mp_state.num_points, 3, 3)))

    def __init__(
        self,
        *,
        constitutive_laws: Tuple[Optional[ConstitutiveLaw], ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Tuple[Optional[Force], ...] = (),
        sdf_logics: Optional[Tuple[SDFObjectBase, ...]] = (),
        alpha=0.99,
        use_dynamic_alpha: bool = True,
        alpha_support_min: float = 1.1,
        alpha_support_max: float = 1.5,
        beta_min: float = 0.0,
        beta_max: float = 0.5,
        use_mls_update: bool = True,
        small_mass_cutoff: float = 1.0e-7,
        cfl_limit: float = 0.5,
    ):

        # MLS Update in G2P
        # TODO add it in p2g as well?
        # Option to not compute shape function gradients?
        self.use_mls_update = use_mls_update

        # FLIP/ PIC
        self.alpha = alpha
        self.use_dynamic_alpha = use_dynamic_alpha
        self.alpha_support_min = alpha_support_min
        self.alpha_support_max = alpha_support_max

        # Seperable
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Stability
        self.small_mass_cutoff = small_mass_cutoff
        self.cfl_limit = cfl_limit

        # logic operations
        self.constitutive_laws = constitutive_laws
        self.couplings = couplings
        self.forces = forces
        self.sdf_logics = sdf_logics

        p_set = sorted(list(set(c.p_idx for c in couplings)))
        g_set = sorted(list(set(c.g_idx for c in couplings)))

        self.active_p_ids = tuple(p_set)
        self.active_g_ids = tuple(g_set)


    def _p2g(self, world, mechanics, dt, time):
        """Particle to Grid Transfer (AFLIP/APIC)."""

        for force in self.forces:
            world, mechanics = force.apply_pre_p2g(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        grid_states = list(world.grids)
        mp_states = list(world.material_points)

        solver_states = list(mechanics.solvers)

        for c in self.couplings:
            # Ignore non-MPM couplings
            if c.skip_mpm_logic:
                continue
            mp_state = mp_states[c.p_idx]
            grid_state = grid_states[c.g_idx]
            intr_cache = mechanics.interactions[(c.p_idx, c.g_idx)]
            solver_state = solver_states[c.s_idx]

            # --- Operations in interaction space ---
            # Get data
            intr_masses_stack = mp_state.mass_stack.at[intr_cache.point_ids].get()
            intr_velocities_stack = mp_state.velocity_stack.at[
                intr_cache.point_ids
            ].get()
            intr_volume_stack = mp_state.volume_stack.at[intr_cache.point_ids].get()
            intr_ext_forces_stack = mp_state.force_stack.at[intr_cache.point_ids].get()
            intr_stress_stack = mp_state.stress_stack.at[intr_cache.point_ids].get()

            # AFLIP compute affine velocity contribution,  C * (x_node - x_p)
            # with C @ dist over N interactions (batched matmul)
            intr_Bp = solver_state.Bp_stack.at[intr_cache.point_ids].get()
            dist_vec_phys = -1.0 * intr_cache.rel_dist * grid_state.cell_size
            affine_vel = jnp.einsum("nij,nj->ni", intr_Bp, dist_vec_phys)

            # Compute weighted momentum and mass contributions
            weighted_mass_stack = intr_cache.shape_vals * intr_masses_stack

            # For AFLIP modification, we add affine part to velocity
            total_intr_velocities_stack = (
                intr_velocities_stack + affine_vel[:, : grid_state.dim]
            )

            # Make affine part same dimension as velocity
            weighted_moment_stack = weighted_mass_stack[:, None] * (
                total_intr_velocities_stack
            )

            # Compute forces contributions
            # External forces
            weighted_ext_force_stack = (
                intr_cache.shape_vals[:, None] * intr_ext_forces_stack
            )
            # Internal forces
            intern_force_term_stack = (
                intr_stress_stack @ intr_cache.shape_grads[..., None]
            ).squeeze(-1)
            intern_force_term_stack = intern_force_term_stack[:, : grid_state.dim]
            weighted_intern_force_stack = (
                -1.0 * intr_volume_stack[:, None] * intern_force_term_stack
            )
            total_intr_force = weighted_intern_force_stack + weighted_ext_force_stack

            # --- Scatter to grid ---
            grid_mass_stack = grid_state.mass_stack.at[intr_cache.node_hashes].add(
                weighted_mass_stack
            )
            grid_moment_stack = grid_state.moment_stack.at[intr_cache.node_hashes].add(
                weighted_moment_stack
            )
            grid_force_stack = (
                jnp.zeros_like(grid_state.moment_stack)
                .at[intr_cache.node_hashes]
                .add(total_intr_force)
            )

            grid_states[c.g_idx] = eqx.tree_at(
                lambda s: (s.mass_stack, s.moment_stack, s.force_stack),
                grid_state,
                (
                    grid_mass_stack,
                    grid_moment_stack,
                    grid_force_stack,
                ),
            )

        world = eqx.tree_at(
            lambda w: (w.grids,),
            world,
            (tuple(grid_states),),
        )
        return world, mechanics

    def _g2p(
        self,
        world,
        mechanics,
        dt,
        time,
    ):
        """Grid to Particle Transfer (AFLIP/APIC with Position Correction and Dynamic Alpha)."""
        # Apply forces hook 4 to modify grid moments, e.g., grid contact
        for force in self.forces:
            world, mechanics = force.apply_grid_moments(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        grid_states = list(world.grids)
        mp_states = list(world.material_points)
        solver_states = list(mechanics.solvers)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue
            grid_state = grid_states[c.g_idx]
            mp_state = mp_states[c.p_idx]
            intr_cache = mechanics.interactions[(c.p_idx, c.g_idx)]
            solver_state = solver_states[c.p_idx]

            # --- Operations in interaction space ---
            # Gather grid data
            intr_mass_stack = grid_state.mass_stack.at[intr_cache.node_hashes].get()
            intr_momement_stack = grid_state.moment_stack.at[intr_cache.node_hashes].get()
            intr_momement_nt_stack = grid_state.moment_nt_stack.at[
                intr_cache.node_hashes
            ].get()

            # Small mass cutoff to prevent instabilities
            safe_masses = jnp.where(
                intr_mass_stack > self.small_mass_cutoff, intr_mass_stack, 1.0
            )[:, None]
            mask = (intr_mass_stack > self.small_mass_cutoff)[:, None]

            # Get old velocity from grid
            intr_vels = jnp.where(mask, intr_momement_stack / safe_masses, 0.0)

            # Get new velocity from grid
            intr_vels_nt = jnp.where(mask, intr_momement_nt_stack / safe_masses, 0.0)

            # Apply padding to velocities to compute shape function gradients in 3D
            # considering plane strain case
            padding = (0, 3 - grid_state.dim)
            intr_vels_nt_3d = jnp.pad(intr_vels_nt, ((0, 0), padding))
            weighted_vels = intr_cache.shape_vals[:, None] * intr_vels
            weighted_vels_nt = intr_cache.shape_vals[:, None] * intr_vels_nt

            # Update the affine term which relates to the velocity gradient
            if self.use_mls_update:
                weighted_Bp_term = jnp.einsum(
                    "ij,ik->ijk", intr_vels_nt_3d, intr_cache.shape_grads
                )
            else:
                # classic apic update
                dist_vec_phys = -1.0 * intr_cache.rel_dist * grid_state.cell_size

                # shape_vals (N,) * outer(v (N,3), dist (N,3)) -> (N,3,3)
                weighted_Bp_term = intr_cache.shape_vals[:, None, None] * jnp.einsum(
                    "ni,nj->nij", intr_vels_nt_3d, dist_vec_phys
                )

            # --- Gather operations to material points ---

            # Old velocity (gathered from grid)
            p_vel = (
                jnp.zeros((mp_state.num_points, grid_state.dim))
                .at[intr_cache.point_ids]
                .add(weighted_vels)
            )
            # New particles
            p_vel_nt = (
                jnp.zeros((mp_state.num_points, grid_state.dim))
                .at[intr_cache.point_ids]
                .add(weighted_vels_nt)
            )
            # Interpolated affine matrix
            p_Bp_intpol = (
                jnp.zeros((mp_state.num_points, 3, 3))
                .at[intr_cache.point_ids]
                .add(weighted_Bp_term)
            )

            # Get velocity gradient and affine matrix
            # If MLS is used we compute it directly from the interpolated value
            # otherwise we need to multiply by the inverse of Dp
            # which is related to the shapefunction gradients
            if self.use_mls_update:
                p_Bp = p_Bp_intpol
                p_L_next = p_Bp  # Approximation
            else:
                # classic APIC needs the inverse of Dp, where
                # Dp = 1/4 * dx^2 * I (Quadratic) or 1/3 * dx^2 * I (Cubic)
                if c.shapefunction == "cubic":
                    coeff = 1.0 / 3.0
                else:
                    coeff = 1.0 / 4.0

                Dp_inv = (1.0 / (coeff * grid_state.cell_size**2)) * jnp.eye(3)

                p_Bp = p_Bp_intpol @ Dp_inv
                p_L_next = (
                    p_Bp  # Approximation (Strictly L should be grad, but Bp is often used)
                )

            # Get velocity velocity fluctuation from grid
            vel_adj = mp_state.velocity_stack - p_vel

            # --- Dynamic Alpha ---
            # Here we apply dynamic alpha based on mass support enter the velocity update for particles
            if self.use_dynamic_alpha:

                p_mass_support = (
                    jnp.zeros_like(mp_state.mass_stack)
                    .at[intr_cache.point_ids]
                    .add(
                        intr_cache.shape_vals
                        * intr_mass_stack  # effectively density * cell_vol
                    )
                )

                support_ratio = p_mass_support / (mp_state.mass_stack + 1e-9)
                alpha_scale = jnp.clip((support_ratio - 1.1) / 0.4, 0.0, 1.0)
                dynamic_alpha = self.alpha * alpha_scale[:, None]
            else:
                dynamic_alpha = self.alpha

            # --- PIC/FLIP blended update ---
            # Update particle velocity
            p_velocity_next = p_vel_nt + dynamic_alpha * vel_adj

            # --- CFL Clamping ---
            # Clamp velocity magnitude to prevent particles crossing >50% of a cell in one step
            max_speed = self.cfl_limit * grid_state.cell_size / dt
            speed = jnp.linalg.norm(p_velocity_next, axis=1, keepdims=True)
            clamp_factor = jnp.minimum(1.0, max_speed / (speed + 1e-12))
            p_velocity_next = p_velocity_next * clamp_factor

            # --- Position Update with Separable Correction ---
            I = jnp.eye(3)
            if grid_state.dim == 2:
                p_L_next = p_L_next.at[:, 2, 2].set(0.0)

            # Deformation Gradient and volume update
            F_inc = I + p_L_next * dt
            p_F_next = jnp.einsum("ijk,ikl->ijl", F_inc, mp_state.F_stack)

            if grid_state.dim == 2:
                p_F_next = p_F_next.at[:, 2, 2].set(1.0)

            J_next = jnp.linalg.det(p_F_next)

            p_volume_next = (J_next[:, None] * mp_state.volume0_stack[:, None]).squeeze()

            # Separable correction to avoid avoid positional trap
            # Use a mix of PIC and FLIP specifically for position to avoid noise
            # If J < 1 (compression), use beta_min (usually 0 -> PIC) to prevent particle crossing
            # If J > 1 (expansion), use beta_max
            beta_p = jnp.where(J_next < 1.0, self.beta_min, self.beta_max)
            correction_term = self.alpha * beta_p[:, None] * vel_adj

            p_position_next = mp_state.position_stack + dt * (p_vel_nt + correction_term)

            mp_states[c.p_idx] = eqx.tree_at(
                lambda s: (
                    s.velocity_stack,
                    s.position_stack,
                    s.volume_stack,
                    s.F_stack,
                    s.L_stack,
                ),
                mp_state,
                (p_velocity_next, p_position_next, p_volume_next, p_F_next, p_L_next),
            )

            solver_states[c.p_idx] = eqx.tree_at(lambda s: s.Bp_stack, solver_state, p_Bp)

        world = eqx.tree_at(
            lambda w: (w.material_points,),
            world,
            (tuple(mp_states),),
        )

        mechanics = eqx.tree_at(
            lambda w: (w.solvers,),
            mechanics,
            (tuple(solver_states),),
        )
        return world, mechanics