# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM
# -*- coding: utf-8 -*-
"""
Explanation:

    This module contains a standard Update Stress Last (USL) Solver.

    Note that this solver should only be used for reference.

    We recommend the `USLAFLIP` solver for practical simulations as it is
    more stable and less dissipative.

    References:
         - De Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory, implementation, and applications."
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


class USLSolverState(BaseSolverState):
    """State class for USL Solver. Currently empty for type consistency."""

    pass


class USLSolver(BaseSolver):
    """Update Stress Last (USL) Solver for MPM simulations.


    Attributes:
        alpha: Blending factor between FLIP and PIC updates (0.0 = FLIP, 1.0 = PIC).
        couplings: Tuple of BodyCoupling instances defining interactions.
        constitutive_laws: Tuple of ConstitutiveLaw instances for material behavior.
        forces: Tuple of Force instances applied during simulation.
    """

    alpha: float = eqx.field(static=True)

    couplings: Tuple[BodyCoupling, ...]

    constitutive_laws: Tuple[ConstitutiveLaw, ...]

    forces: Tuple[Force, ...]

    def create_state(self, mp_state) -> Self:
        """Creates a default USLSolverState instance."""
        return USLSolverState()

    def __init__(
        self,
        *,
        constitutive_laws: Tuple[ConstitutiveLaw, ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Tuple[Force, ...] = (),
        alpha=0.99,
    ):
        self.constitutive_laws = constitutive_laws
        self.couplings = couplings
        self.forces = forces
        self.alpha = alpha

    def step(
        self: Self,
        state: SimState,
    ) -> SimState:
        """
        Update SimState (global simulation state) by one time step.
        """
        # unpack  mutable from previous global state

        dt = state.dt
        time = state.time

        # We modify the lists to update the states
        mp_states = list(state.material_points)
        grid_states = list(state.grids)
        solver_states = list(state.solvers)  # unused for now
        constitutive_law_states = list(state.constitutive_laws)
        force_states = list(state.forces)
        intr_caches = state.interactions

        # Get active grids and particles from body couplings
        active_grids = set(c.g_idx for c in self.couplings)
        active_particles = set(c.p_idx for c in self.couplings)

        # Reset grid
        # TODO is there a way not to store grid in global state?
        for g_idx in active_grids:
            grid_state = grid_states[g_idx]
            grid_state = eqx.tree_at(
                lambda state: (
                    state.mass_stack,
                    state.moment_stack,
                    state.moment_nt_stack,
                    state.force_stack,
                ),
                grid_state,
                (
                    jnp.zeros_like(grid_state.mass_stack),
                    jnp.zeros_like(grid_state.moment_stack),
                    jnp.zeros_like(grid_state.moment_nt_stack),
                    jnp.zeros_like(grid_state.force_stack),
                ),
            )
            grid_states[g_idx] = grid_state

        # Reset material point temporary states
        for p_id in active_particles:
            mp_state = mp_states[p_id]

            if isinstance(mp_state, MaterialPointState):
                mp_state = eqx.tree_at(
                    lambda state: (state.L_stack, state.force_stack),
                    mp_state,
                    (
                        mp_state.L_stack.at[:].set(0.0),
                        mp_state.force_stack.at[:].set(0.0),
                    ),
                )
            mp_states[p_id] = mp_state

        # Compute connectivity of material points to grid nodes, node hashes, shape functions etc.
        for c in self.couplings:

            # TODO move update for rigid particle away from here maybe?
            # since we want to compute it only once?
            # Or make a flag to not compute and skip.
            # Then compute on initiation?
            # if c.skip_mpm_logic:
            #     continue
            p_pos = mp_states[c.p_idx].position_stack
            intr_caches[(c.p_idx, c.g_idx)] = c.shape_map.compute(
                p_pos,
                origin=grid_states[c.g_idx].origin,
                grid_size=grid_states[c.g_idx].grid_size,
                inv_cell_size=grid_states[c.g_idx]._inv_cell_size,
                intr_cache=intr_caches[(c.p_idx, c.g_idx)],
            )

        # Apply forces hook 1 (e.g., move rigid body) over all couplings (not just active pairs)
        # forces can modify material point positions directly here
        for force in self.forces:
            force_states = force.apply_kinematics(
                mp_states,
                grid_states,
                force_states,
                intr_caches,
                self.couplings,
                dt,
                time,
            )

        # Apply forces hook 2 to modify material point forces (e.g., apply external forces on points)
        for force in self.forces:
            mp_states, force_states = force.apply_pre_p2g(
                mp_states,
                grid_states,
                force_states,
                intr_caches,
                self.couplings,
                dt,
                time,
            )

        # Particle to grid (scatter) update
        for c in self.couplings:

            # Ignore non-MPM couplings
            if c.skip_mpm_logic:
                continue

            grid_states[c.g_idx] = self._p2g(
                mp_states[c.p_idx],
                grid_states[c.g_idx],
                intr_caches[(c.p_idx, c.g_idx)],
                dt,
            )

        # Apply forces hook 3 to modify grid forces (e.g., body forces, gravity)
        for force in self.forces:
            grid_states, force_states = force.apply_grid_forces(
                mp_states,
                grid_states,
                force_states,
                intr_caches,
                self.couplings,
                dt,
                time,
            )

        # Integrate grid forces to moments
        for c in self.couplings:
            if c.skip_mpm_logic:
                continue
            grid_states[c.g_idx] = self._integrate_grid(grid_states[c.g_idx], dt)

        # Apply forces hook 4 to modify grid moments, e.g., grid contact
        for force in self.forces:
            grid_states, force_states = force.apply_grid_moments(
                mp_states,
                grid_states,
                force_states,
                intr_caches,
                self.couplings,
                dt,
                time,
            )

        # G2P Update and constitutive law update
        for i, c in enumerate(self.couplings):
            if c.skip_mpm_logic:
                continue
            grid_state = grid_states[c.g_idx]

            mp_state = mp_states[c.p_idx]
            constitutive_law_state = constitutive_law_states[c.c_idx]
            intr_cache = intr_caches[(c.p_idx, c.g_idx)]

            mp_state = self._g2p(mp_state, grid_state, intr_cache, dt)

            # Update material point, and internal variables via constitutive law
            mp_state, constitutive_law_state = self.constitutive_laws[c.c_idx].update(
                mp_state,
                constitutive_law_state,
                dt,
            )

            # Remove shear strain from deformation gradient if required
            # by certain constitutive laws
            mp_state = self.constitutive_laws[c.c_idx].remove_accumulated_shear(
                mp_state,
                grid_state.dim,
            )

            mp_states[c.p_idx] = mp_state
            constitutive_law_states[c.c_idx] = constitutive_law_state

        # Update SimState
        # TODO add error checks for nan/inf or solver convergence status?
        # Might make the code messy though
        return eqx.tree_at(
            lambda s: (
                s.material_points,
                s.grids,
                s.constitutive_laws,
                s.forces,
                s.time,
                s.step,
            ),
            state,
            (
                tuple(mp_states),
                tuple(grid_states),
                tuple(constitutive_law_states),
                tuple(force_states),
                state.time + dt,
                state.step + 1,
            ),
        )

    def _p2g(self, mp_state: MaterialPointState, grid_state: GridState, intr_cache: InteractionCache, dt):
        """Particle to Grid Transfer."""
        # --- Operations in interaction space ---
        # Gather particle data
        intr_masses_stack = mp_state.mass_stack.at[intr_cache.point_ids].get()
        intr_velocities_stack = mp_state.velocity_stack.at[intr_cache.point_ids].get()
        intr_volume_stack = mp_state.volume_stack.at[intr_cache.point_ids].get()
        intr_ext_forces_stack = mp_state.force_stack.at[intr_cache.point_ids].get()
        intr_stress_stack = mp_state.stress_stack.at[intr_cache.point_ids].get()

        # TODO can be used to sanitize the simulation
        # valid_mask = jnp.all(jnp.isfinite(intr_velocities_stack), axis=1) & jnp.all(
        #     jnp.isfinite(intr_stress_stack), axis=(1, 2)
        # )

        # Compute weighted momentum and mass contributions
        weighted_mass_stack = intr_cache.shape_vals * intr_masses_stack
        weighted_moment_stack = weighted_mass_stack[:, None] * intr_velocities_stack

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
            .add(weighted_intern_force_stack + weighted_ext_force_stack)
        )

        return eqx.tree_at(
            lambda s: (s.mass_stack, s.moment_stack, s.force_stack),
            grid_state,
            (
                grid_mass_stack,
                grid_moment_stack,
                grid_force_stack,
            ),
        )

    def _integrate_grid(self, grid_state: GridState, dt) -> GridState:
        """Integrate grid forces to update momenta. Explicit Euler integration m_new =m_old + f * dt."""
        grid_moment_nt_stack = grid_state.moment_stack + grid_state.force_stack * dt
        return eqx.tree_at(
            lambda s: s.moment_nt_stack, grid_state, grid_moment_nt_stack
        )

    def _g2p(self, mp_state: MaterialPointState, grid_state: GridState, intr_cache: InteractionCache, dt):
        """Grid to Particle Transfer."""

        # --- Operations in interaction space ---
        # Gather grid data
        intr_mass_stack = grid_state.mass_stack.at[intr_cache.node_hashes].get()
        intr_momement_stack = grid_state.moment_stack.at[intr_cache.node_hashes].get()
        intr_momement_nt_stack = grid_state.moment_nt_stack.at[
            intr_cache.node_hashes
        ].get()

        # apply operations in interaction space
        # We do not add small mass correction for USL? 
        # but keep it simple for reference?
        safe_masses = jnp.where(intr_mass_stack > 1e-12, intr_mass_stack, 1.0)[:, None]
        mask = (intr_mass_stack > 1e-12)[:, None]

        # Get old velocity from grid
        intr_vels = jnp.where(mask, intr_momement_stack / safe_masses, 0.0)
        
        # Get new velocity from grid
        intr_vels_nt = jnp.where(mask, intr_momement_nt_stack / safe_masses, 0.0)

        # Get weighted velocity, and velocity difference from grid 
        intr_delta_vels = intr_vels_nt - intr_vels
        weighted_delta_vels = intr_cache.shape_vals[:, None] * intr_delta_vels
        weighted_vels_nt = intr_cache.shape_vals[:, None] * intr_vels_nt
        
        # Apply padding to velocities to compute shape function gradients in 3D
        # considering plane strain case
        padding = (0, 3 - grid_state.dim)
        intr_vels_nt_3d = jnp.pad(intr_vels_nt, ((0, 0), padding))

        # Outer product to find velocity gradients (n_intr, 3, 3)
        # As L_ij = v_i * grad_j
        weighted_L = jnp.einsum("ij,ik->ijk", intr_cache.shape_grads, intr_vels_nt_3d)

        # --- Gather operations to material points ---
        # Particle velocity differences on grid
        p_delta_vel = (
            jnp.zeros((mp_state.num_points, grid_state.dim))
            .at[intr_cache.point_ids]
            .add(weighted_delta_vels)
        )
        
        # Particle new velocity
        p_vel_nt = (
            jnp.zeros((mp_state.num_points, grid_state.dim))
            .at[intr_cache.point_ids]
            .add(weighted_vels_nt)
        )

        # Particle velocity gradient
        p_L = (
            jnp.zeros((mp_state.num_points, 3, 3))
            .at[intr_cache.point_ids]
            .add(weighted_L)
        )

        # FLIP/PIC update
        p_velocity_next = (1.0 - self.alpha) * p_vel_nt + self.alpha * (
            mp_state.velocity_stack + p_delta_vel
        )
        p_position_next = mp_state.position_stack + p_vel_nt * dt

        # Update deformation Gradient and volume
        I = jnp.eye(3)
        if grid_state.dim == 2:
            p_L = p_L.at[:, 2, 2].set(0.0)

        F_inc = I + p_L * dt
        p_F_next = jnp.einsum("ijk,ikl->ijl", F_inc, mp_state.F_stack)

        if grid_state.dim == 2:
            p_F_next = p_F_next.at[:, 2, 2].set(1.0)

        J = jnp.linalg.det(p_F_next)
        p_volume_next = (J[:, None] * mp_state.volume0_stack[:, None]).squeeze()
        return eqx.tree_at(
            lambda s: (
                s.velocity_stack,
                s.position_stack,
                s.volume_stack,
                s.F_stack,
                s.L_stack,
            ),
            mp_state,
            (p_velocity_next, p_position_next, p_volume_next, p_F_next, p_L),
        )
