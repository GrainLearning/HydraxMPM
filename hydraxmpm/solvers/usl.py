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

from typing import Self, Tuple, List, Optional

import jax.numpy as jnp

from ..grid.grid import GridState
from ..material_points.material_points import MaterialPointState

from .solver import BaseSolver, BaseSolverState

from ..common.simstate import SimState

from ..forces.force import Force

from .coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw

from ..shapefunctions.mapping import InteractionCache

from ..sdf.sdfobject import SDFObjectBase, SDFObjectState


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

    sdf_logics: Tuple[SDFObjectBase, ...]

    active_p_ids: Tuple[int, ...] = eqx.field(static=True)
    active_g_ids: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        constitutive_laws: Tuple[ConstitutiveLaw, ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Optional[Tuple[Force, ...]] = (),
        sdf_logics: Optional[Tuple[SDFObjectBase, ...]] = (),
        alpha=0.99,
    ):
        self.constitutive_laws = constitutive_laws
        self.couplings = couplings
        self.forces = forces
        self.sdf_logics = sdf_logics
        self.alpha = alpha

        p_set = sorted(list(set(c.p_idx for c in couplings)))
        g_set = sorted(list(set(c.g_idx for c in couplings)))

        self.active_p_ids = tuple(p_set)
        self.active_g_ids = tuple(g_set)

    def create_state(self, mp_state) -> Self:
        """Creates a default USLSolverState instance."""
        return USLSolverState()


    def __call__(
        self: Self,
        state: SimState,
    ) -> SimState:
        """
        Advances the simulation by one timestep dt.
        Usage: new_state = solver(old_state)
        """
        # unpack  mutable from previous global state
        dt = state.dt
        time = state.time

        world = state.world
        mechanics = state.mechanics

        world, mechanics = self._kinematic_update(world, mechanics, dt, time)

        world, mechanics = self._reset_state(world, mechanics, dt, time)

        world, mechanics = self._compute_connectivity(world, mechanics, dt, time)

        world, mechanics = self._p2g(world, mechanics, dt, time)

        world, mechanics = self._integrate_grid(world, mechanics, dt, time)

        world, mechanics = self._g2p(world, mechanics, dt, time)

        world, mechanics = self._constitutive_update(world, mechanics, dt, time)

        time = time + dt

        step = state.step + 1
        state = eqx.tree_at(
            lambda s: (s.world, s.mechanics, s.step,s.time),
            state,
            (world, mechanics, step, time),
        )

        return state



    def _kinematic_update(self, world, mechanics, dt, time):

        # Apply forces hook 1 (e.g., move rigid body) over all couplings (not just active pairs)
        for force in self.forces:
            world, mechanics = force.apply_kinematics(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        new_sdf_states = []
        for i, sdf_logic in enumerate(self.sdf_logics):

            sdf_state = sdf_logic.update_kinematics(world.sdfs[i], dt)
            new_sdf_states.append(sdf_state)

        # Repack
        world = eqx.tree_at(lambda w: w.sdfs, world, tuple(new_sdf_states))
        return world, mechanics
    

    def _reset_state(self, world, mechanics, dt, time):

        grid_states = list(world.grids)
        mp_states = list(world.material_points)

        active_grid_ids = set(c.g_idx for c in self.couplings)
        active_particles_ids = set(c.p_idx for c in self.couplings)

        for g_idx in active_grid_ids:
            g = grid_states[g_idx]
            grid_states[g_idx] = eqx.tree_at(
                lambda s: (
                    s.mass_stack,
                    s.moment_stack,
                    s.moment_nt_stack,
                    s.force_stack,
                ),
                g,
                (
                    jnp.zeros_like(g.mass_stack),
                    jnp.zeros_like(g.moment_stack),
                    jnp.zeros_like(g.moment_nt_stack),
                    jnp.zeros_like(g.force_stack),
                ),
            )

        for p_id in active_particles_ids:
            mp = mp_states[p_id]
            if isinstance(mp, MaterialPointState):
                mp_states[p_id] = eqx.tree_at(
                    lambda s: (s.L_stack, s.force_stack),
                    mp,
                    (mp.L_stack.at[:].set(0.0), mp.force_stack.at[:].set(0.0)),
                )

        # Repack
        world = eqx.tree_at(
            lambda w: (w.grids, w.material_points),
            world,
            (tuple(grid_states), tuple(mp_states)),
        )
        return world, mechanics

    def _compute_connectivity(self, world, mechanics, dt, time):
        # Compute connectivity of material points to grid nodes, node hashes, shape functions etc.

        interactions = mechanics.interactions

        for c in self.couplings:

            # TODO move update for rigid particle away from here maybe?
            # since we want to compute it only once?
            # Or make a flag to not compute and skip.
            # Then compute on initiation?
            # if c.skip_mpm_logic:
            #     continue
            p_pos = world.material_points[c.p_idx].position_stack

            interactions[(c.p_idx, c.g_idx)] = c.shape_map.compute(
                p_pos,
                origin=world.grids[c.g_idx].origin,
                grid_size=world.grids[c.g_idx].grid_size,
                inv_cell_size=world.grids[c.g_idx]._inv_cell_size,
                intr_cache=interactions[(c.p_idx, c.g_idx)],
            )
        # Repack
        mechanics = eqx.tree_at(
            lambda m: (m.interactions,),
            mechanics,
            (interactions,),
        )
        return world, mechanics
    
    def _p2g(self, world, mechanics, dt, time):

        # Apply forces hook 2 to modify material point forces (e.g., apply external forces on points)
        for force in self.forces:
            world, mechanics = force.apply_pre_p2g(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        grid_states = list(world.grids)
        mp_states = list(world.material_points)

        for c in self.couplings:
            # Ignore non-MPM couplings
            if c.skip_mpm_logic:
                continue
            mp_state = mp_states[c.p_idx]
            grid_state = grid_states[c.g_idx]
            intr_cache = mechanics.interactions[(c.p_idx, c.g_idx)]

            """Particle to Grid Transfer."""
            # --- Operations in interaction space ---
            # Gather particle data
            intr_masses_stack = mp_state.mass_stack.at[intr_cache.point_ids].get()
            intr_velocities_stack = mp_state.velocity_stack.at[
                intr_cache.point_ids
            ].get()
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

    def _integrate_grid(self, world, mechanics, dt, time):
        """Integrate grid forces to update momenta. Explicit Euler integration m_new =m_old + f * dt."""
        
        for force in self.forces:
            world, mechanics = force.apply_grid_forces(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        # Integrate grid forces to moments
        grid_states = list(world.grids)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue
            # grid_states[c.g_idx] = self._integrate_grid(grid_states[c.g_idx], dt)

            grid_state = grid_states[c.g_idx]

            grid_moment_nt_stack = grid_state.moment_stack + grid_state.force_stack * dt
            grid_state = eqx.tree_at(
                lambda s: s.moment_nt_stack, grid_state, grid_moment_nt_stack
            )

            grid_states[c.g_idx] = grid_state

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
        """Grid to Particle Transfer."""

        # Apply forces hook 4 to modify grid moments, e.g., grid contact
        for force in self.forces:
            world, mechanics = force.apply_grid_moments(
                world, mechanics, self.sdf_logics, self.couplings, dt, time
            )

        grid_states = list(world.grids)

        mp_states = list(world.material_points)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue
            grid_state = grid_states[c.g_idx]
            mp_state = mp_states[c.p_idx]
            intr_cache = mechanics.interactions[(c.p_idx, c.g_idx)]

            # --- Operations in interaction space ---
            # Gather grid data
            intr_mass_stack = grid_state.mass_stack.at[intr_cache.node_hashes].get()
            intr_momement_stack = grid_state.moment_stack.at[
                intr_cache.node_hashes
            ].get()
            intr_momement_nt_stack = grid_state.moment_nt_stack.at[
                intr_cache.node_hashes
            ].get()

            # apply operations in interaction space
            # We do not add small mass correction for USL?
            # but keep it simple for reference?
            safe_masses = jnp.where(intr_mass_stack > 1e-12, intr_mass_stack, 1.0)[
                :, None
            ]
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
            weighted_L = jnp.einsum(
                "ij,ik->ijk", intr_cache.shape_grads, intr_vels_nt_3d
            )

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
            mp_state = eqx.tree_at(
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
            mp_states[c.p_idx] = mp_state

        world = eqx.tree_at(
            lambda w: (w.material_points,),
            world,
            (tuple(mp_states),),
        )

        return world, mechanics

    def _constitutive_update(self, world, mechanics, dt, time):

        mp_states = list(world.material_points)
        constitutive_law_states = list(mechanics.constitutive_laws)
        for i, c in enumerate(self.couplings):
            if c.skip_mpm_logic:
                continue

            mp_state = mp_states[c.p_idx]
            constitutive_law_state = constitutive_law_states[c.c_idx]

            # Update material point, and internal variables via constitutive law
            mp_state, constitutive_law_state = self.constitutive_laws[c.c_idx].update(
                mp_state,
                constitutive_law_state,
                dt,
            )

            # Remove shear strain from deformation gradient if required
            # by certain constitutive laws
            mp_state = self.constitutive_laws[c.c_idx].remove_accumulated_shear(
                mp_state
            )

            mp_states[c.p_idx] = mp_state
            constitutive_law_states[c.c_idx] = constitutive_law_state

        world = eqx.tree_at(lambda w: (w.material_points,), world, (tuple(mp_states),))
        mechanics = eqx.tree_at(
            lambda m: (m.constitutive_laws,),
            mechanics,
            (tuple(constitutive_law_states),),
        )
        return world, mechanics
    