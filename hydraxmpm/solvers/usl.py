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
         - Hu, Yuanming, et al. "A moving least squares material point method with displacement discontinuity and two-way rigid body coupling."
"""

import equinox as eqx

from typing import Self, Tuple, List, Optional, Dict

import jax.numpy as jnp
import jax

from jaxtyping import Array, Float, Int

from ..material_points.material_points import MaterialPointState

from .solver import BaseSolver, BaseSolverState

from ..common.simstate import SimState, SimCache, NodeGeometry, ParticleGeometry

from ..forces.force import Force

from .coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw

from ..shapefunctions.mapping import InteractionCache

from ..sdf.sdfobject import SDFObjectBase, SDFObjectState

from ..grid.grid import GridDomain, GridArrays


class USLSolverState(BaseSolverState):
    """State class for USL Solver. Currently empty for type consistency."""

    pass


class USLSolver(BaseSolver):
    """Update Stress Last (USL) Solver for MPM simulations.


    Attributes:
        alpha: Blending factor between FLIP and PIC updates (0.0 = FLIP, 1.0 = PIC).
        grid_domains: Spatial-computational spaces for grids.
        couplings: Defines how material connect to the spaces.
        constitutive_laws: Tuple of ConstitutiveLaw instances for material behavior.
        forces: Tuple of Force instances applied during simulation.
    """

    alpha: float = eqx.field(static=True)

    grid_domains: Tuple[GridDomain, ...] = eqx.field(static=True)

    couplings: Tuple[BodyCoupling, ...]

    constitutive_laws: Tuple[ConstitutiveLaw, ...]

    forces: Tuple[Force, ...]

    sdf_logics: Tuple[SDFObjectBase, ...]

    active_p_ids: Tuple[int, ...] = eqx.field(static=True)
    active_g_ids: Tuple[int, ...] = eqx.field(static=True)

    sdf_mp_sharpness: float = eqx.field(static=True, default=1000.0)

    def __init__(
        self,
        *,
        grid_domains: Tuple[GridDomain, ...],
        constitutive_laws: Tuple[ConstitutiveLaw, ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Optional[Tuple[Force, ...]] = (),
        sdf_logics: Optional[Tuple[SDFObjectBase, ...]] = (),
        alpha=0.99,
        sdf_mp_sharpness: float = 1000.0,
    ):
        self.constitutive_laws = constitutive_laws
        self.couplings = couplings
        self.forces = forces
        self.sdf_logics = sdf_logics
        self.grid_domains = grid_domains
        self.alpha = alpha

        self.sdf_mp_sharpness = sdf_mp_sharpness
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

        # kinematics (Move SDFs) & Reset
        world, mechanics = self._kinematic_update(world, mechanics, dt, time)

        world, mechanics = self._reset_material_points(world, mechanics, dt, time)

        # create fresh zeroed grids in cache
        grids = [GridArrays.new(domain) for domain in self.grid_domains]

        sim_cache = SimCache(grids=grids, interactions={}, node_geoms={}, mp_geoms={})

        # node-particle connectivity (shape Functions + correction)
        world, mechanics, sim_cache = self._compute_connectivity(
            world, mechanics, sim_cache, dt, time
        )

        # precompute SDF geometries, to avoid redudant computations
        # especially for AD
        # note that we also return the union of all SDF
        # to use for CPIC-style correction
        sim_cache, grid_union_masks = self._get_nodes_sdfs(world, sim_cache, dt, time)
        sim_cache, mp_union_masks = self._get_particle_sdfs(world, sim_cache, dt, time)
        sim_cache, mid_point_masks = self._get_mid_point_sdfs(
            world, sim_cache, dt, time
        )

        # combine logics to correct connectivity
        sim_cache = self._sdf_correction_connectivity(
            world,
            sim_cache,
            grid_union_masks,
            mp_union_masks,
            mid_point_masks,
            dt,
            time,
        )

        # Particle to Grid Transfer
        world, mechanics, sim_cache = self._p2g(world, mechanics, sim_cache, dt, time)

        # # Integrate grid forces
        world, mechanics, sim_cache = self._integrate_grid(
            world, mechanics, sim_cache, dt, time
        )

        # Grid to Particle Transfer
        world, mechanics, sim_cache = self._g2p(world, mechanics, sim_cache, dt, time)

        # # Constitutive update
        world, mechanics, sim_cache = self._constitutive_update(
            world, mechanics, sim_cache, dt, time
        )

        time = time + dt

        step = state.step + 1
        state = eqx.tree_at(
            lambda s: (s.world, s.mechanics, s.step, s.time),
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

    def _reset_material_points(self, world, mechanics, dt, time):

        mp_states = list(world.material_points)

        active_particles_ids = set(c.p_idx for c in self.couplings)

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
            lambda w: w.material_points,
            world,
            tuple(mp_states),
        )
        return world, mechanics

    def _compute_connectivity(self, world, mechanics, sim_cache, dt, time):
        # Compute connectivity of material points to grid nodes, node hashes, shape functions etc.

        for c in self.couplings:

            sim_cache.interactions[(c.p_idx, c.g_idx)] = c.shape_map.compute(
                world.material_points[c.p_idx].position_stack,
                origin=self.grid_domains[c.g_idx].origin,
                grid_size=self.grid_domains[c.g_idx].grid_size,
                inv_cell_size=self.grid_domains[c.g_idx]._inv_cell_size,
            )
        return world, mechanics, sim_cache

    def _get_nodes_sdfs(self, world, sim_cache, dt, time):

        # returns union of blockage masks over all SDFs for each grid
        grid_union_masks = {}
        for g_idx, domain in enumerate(self.grid_domains):
            node_pos = domain.position_stack

            # Accumulator: 0.0 = Out, 1.0 = In
            union_mask = jnp.zeros((domain.num_cells,))

            for s_idx, sdf_logic in enumerate(self.sdf_logics):
                sdf_state = world.sdfs[s_idx]

                dists = (
                    sdf_logic.get_signed_distance_stack(sdf_state, node_pos)
                    - domain.cell_size * 1.0
                )
                normals = sdf_logic.get_normal_stack(sdf_state, node_pos)
                wall_vels = sdf_logic.get_velocity_stack(sdf_state, node_pos, dt)
                friction = sdf_logic.get_surface_friction_stack(sdf_state, node_pos)

                sim_cache.node_geoms[(g_idx, s_idx)] = NodeGeometry(
                    dists=dists, normals=normals, wall_vels=wall_vels, friction=friction
                )

                # C. Accumulate Union (For CPIC)
                is_in = jax.nn.sigmoid(-dists * self.sdf_mp_sharpness).squeeze()
                union_mask = jnp.maximum(union_mask, is_in)

            grid_union_masks[g_idx] = union_mask

        return sim_cache, grid_union_masks

    def _get_particle_sdfs(self, world, sim_cache, dt, time):

        mp_union_masks = {}

        for p_idx, mp_state in enumerate(world.material_points):

            union_mask = jnp.zeros(mp_state.num_points)

            for s_idx, sdf_logic in enumerate(self.sdf_logics):
                sdf_state = world.sdfs[s_idx]

                dists = sdf_logic.get_signed_distance_stack(
                    sdf_state, mp_state.position_stack
                )
                normals = sdf_logic.get_normal_stack(sdf_state, mp_state.position_stack)
                wall_vels = sdf_logic.get_velocity_stack(
                    sdf_state, mp_state.position_stack, dt
                )
                frictions = sdf_logic.get_surface_friction_stack(
                    sdf_state, mp_state.position_stack
                )

                sim_cache.mp_geoms[(p_idx, s_idx)] = ParticleGeometry(
                    dists=dists,
                    normals=normals,
                    wall_vels=wall_vels,
                    friction=frictions,
                )

                # C. Accumulate Union (For CPIC)
                is_in = jax.nn.sigmoid(-dists * self.sdf_mp_sharpness)
                union_mask = jnp.maximum(union_mask, is_in)

            mp_union_masks[p_idx] = union_mask

        return sim_cache, mp_union_masks

    def _get_mid_point_sdfs(self, world, sim_cache, dt, time):
        """
        Computes Union Masks for Midpoints.
        Returns: sim_cache, midpoint_masks (Dict[(p_idx, g_idx), Array])
        """
        midpoint_masks = {}

        for c in self.couplings:

            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]
            mp_state = world.material_points[c.p_idx]
            grid_domain = self.grid_domains[c.g_idx]

            # 1. Reconstruct Positions
            node_multi_index = jnp.unravel_index(
                intr_cache.node_hashes, grid_domain.grid_size
            )
            indices_stacked = jnp.stack(node_multi_index, axis=-1)
            node_pos_batch = (
                jnp.array(grid_domain.origin) + indices_stacked * grid_domain.cell_size
            )
            flat_node_pos = node_pos_batch.reshape(-1, grid_domain.dim)

            p_pos_expanded = mp_state.position_stack.at[intr_cache.point_ids].get()

            # 2. Compute Midpoints
            midpoint_pos_stack = 0.5 * (flat_node_pos + p_pos_expanded)

            # 3. Accumulate Union
            # We don't store specific geometry for midpoints, just the mask
            mid_in_mask_flat = jnp.zeros((intr_cache.num_interactions,))

            for s_idx, sdf_logic in enumerate(self.sdf_logics):
                sdf_state = world.sdfs[s_idx]

                d_mid = sdf_logic.get_signed_distance_stack(
                    sdf_state, midpoint_pos_stack
                )
                is_in = jax.nn.sigmoid(-d_mid * self.sdf_mp_sharpness).squeeze()

                mid_in_mask_flat = jnp.maximum(mid_in_mask_flat, is_in)

            midpoint_masks[(c.p_idx, c.g_idx)] = mid_in_mask_flat

        return sim_cache, midpoint_masks

    def _sdf_correction_connectivity(
        self,
        world,
        sim_cache,
        grid_union_masks,
        mp_union_masks,
        midpoint_masks,
        dt,
        time,
    ):
        """
        CPIC-style correction of shape functions based on SDF objects.

        """
        if len(self.sdf_logics) == 0:
            return sim_cache

        for c in self.couplings:
            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]
            mp_state = world.material_points[c.p_idx]
            grid_domain = self.grid_domains[c.g_idx]

            # We act on the shape (N, stencil size e.g., 27 for quadratic 3D)
            # num_interactions
            stencil_size = intr_cache.shape_vals.size // mp_state.num_points
            point_interaction_shape = (mp_state.num_points, stencil_size)

            # is particle inside any SDF?
            mp_is_in = mp_union_masks[c.p_idx]
            mp_is_out = 1.0 - mp_is_in

            # is node inside any SDF?
            node_is_in_flat = grid_union_masks[c.g_idx]
            intr_node_is_in = node_is_in_flat.at[intr_cache.node_hashes].get()
            intr_node_is_in = intr_node_is_in.reshape(point_interaction_shape)

            # is any mid point inside any SDF?
            mid_in_mask_flat = midpoint_masks[(c.p_idx, c.g_idx)]
            mid_in_mask = mid_in_mask_flat.reshape(point_interaction_shape)

            # we block the interaction if node OR midpoint is inside
            path_blocked = jnp.maximum(intr_node_is_in, mid_in_mask)

            # interaction is incompatible if particle is outside and path is blocked
            incompatible = mp_is_out[:, None] * path_blocked

            compatible_mask = (1.0 - incompatible).reshape(-1)

            new_interactions = sim_cache.interactions.copy()
            # new_interactions[(c.p_idx, c.g_idx)] = eqx.tree_at(
            #     lambda i: i.cpic_mask, intr_cache, compatible_mask
            # )

            new_interactions[(c.p_idx, c.g_idx)] = eqx.tree_at(
                lambda i: i.cpic_mask, intr_cache, jnp.ones_like(compatible_mask)
            )
            sim_cache = eqx.tree_at(
                lambda s: s.interactions, sim_cache, new_interactions
            )
        return sim_cache

    def _p2g(self, world, mechanics, sim_cache, dt, time):

        # Apply forces hook 2 to modify material point forces (e.g., apply external forces on points)
        for force in self.forces:
            world, mechanics, sim_cache = force.apply_pre_p2g(
                world,
                mechanics,
                sim_cache,
                self.sdf_logics,
                self.couplings,
                self.grid_domains,
                dt,
                time,
            )

        grids = list(sim_cache.grids)
        mp_states = list(world.material_points)

        for c in self.couplings:
            # Ignore non-MPM couplings
            if c.skip_mpm_logic:
                continue
            mp_state = mp_states[c.p_idx]
            grid_cache = grids[c.g_idx]
            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]

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

            # Apply CPIC mask to shape functions
            effective_shape_vals = intr_cache.shape_vals * intr_cache.cpic_mask

            # Compute weighted momentum and mass contributions
            weighted_mass_stack = effective_shape_vals * intr_masses_stack
            weighted_moment_stack = weighted_mass_stack[:, None] * intr_velocities_stack

            # Compute forces contributions
            # External forces
            weighted_ext_force_stack = (
                effective_shape_vals[:, None] * intr_ext_forces_stack
            )
            # Internal forces
            intern_force_term_stack = (
                intr_stress_stack @ intr_cache.shape_grads[..., None]
            ).squeeze(-1)
            intern_force_term_stack = intern_force_term_stack[:, : grid_cache.dim]

            # compression positive for forces
            weighted_intern_force_stack = (
                1.0 * intr_volume_stack[:, None] * intern_force_term_stack
            ) * intr_cache.cpic_mask[:, None]

            # --- Scatter to grid ---
            grid_mass_stack = grid_cache.mass_stack.at[intr_cache.node_hashes].add(
                weighted_mass_stack
            )
            grid_moment_stack = grid_cache.moment_stack.at[intr_cache.node_hashes].add(
                weighted_moment_stack
            )

            grid_force_stack = (
                jnp.zeros_like(grid_cache.moment_stack)
                .at[intr_cache.node_hashes]
                .add(weighted_intern_force_stack + weighted_ext_force_stack)
            )

            grids[c.g_idx] = eqx.tree_at(
                lambda s: (s.mass_stack, s.moment_stack, s.force_stack),
                grid_cache,
                (
                    grid_mass_stack,
                    grid_moment_stack,
                    grid_force_stack,
                ),
            )

        sim_cache = eqx.tree_at(
            lambda s: s.grids,
            sim_cache,
            tuple(grids),
        )
        return world, mechanics, sim_cache

    def _integrate_grid(self, world, mechanics, sim_cache, dt, time):
        """Integrate grid forces to update momenta. Explicit Euler integration m_new =m_old + f * dt."""

        for force in self.forces:
            world, mechanics, sim_cache = force.apply_grid_forces(
                world,
                mechanics,
                sim_cache,
                self.sdf_logics,
                self.couplings,
                self.grid_domains,
                dt,
                time,
            )

        # Integrate grid forces to moments
        grids = list(sim_cache.grids)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue

            grid_cache = grids[c.g_idx]

            grid_moment_nt_stack = grid_cache.moment_stack + grid_cache.force_stack * dt
            grid_cache = eqx.tree_at(
                lambda s: s.moment_nt_stack, grid_cache, grid_moment_nt_stack
            )

            grids[c.g_idx] = grid_cache

        sim_cache = eqx.tree_at(lambda s: s.grids, sim_cache, tuple(grids))

        for force in self.forces:
            world, mechanics, sim_cache = force.apply_grid_moments(
                world,
                mechanics,
                sim_cache,
                self.sdf_logics,
                self.couplings,
                self.grid_domains,
                dt,
                time,
            )

        return world, mechanics, sim_cache

    def _g2p(
        self,
        world,
        mechanics,
        sim_cache,
        dt,
        time,
    ):
        """Grid to Particle Transfer."""

        grids = list(sim_cache.grids)

        mp_states = list(world.material_points)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue

            grid_cache = grids[c.g_idx]
            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]
            mp_state = mp_states[c.p_idx]

            # --- Operations in interaction space ---
            # Gather grid data
            intr_mass_stack = grid_cache.mass_stack.at[intr_cache.node_hashes].get()
            intr_momement_stack = grid_cache.moment_stack.at[
                intr_cache.node_hashes
            ].get()
            intr_momement_nt_stack = grid_cache.moment_nt_stack.at[
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

            # --------- cpic-----------
            #  ghost velocity correction
            num_sdfs = len(self.sdf_logics)
            if num_sdfs > 0:

                # Find Closest SDF for each Particle (Particle Space)
                dists_stack = jnp.stack(
                    [
                        sim_cache.mp_geoms[(c.p_idx, s)].dists.squeeze()
                        for s in range(num_sdfs)
                    ]
                )
                norms_stack = jnp.stack(
                    [sim_cache.mp_geoms[(c.p_idx, s)].normals for s in range(num_sdfs)]
                )
                vels_stack = jnp.stack(
                    [
                        sim_cache.mp_geoms[(c.p_idx, s)].wall_vels
                        for s in range(num_sdfs)
                    ]
                )

                closest_idx = jnp.argmin(dists_stack, axis=0, keepdims=True)

                p_normal_best = jnp.take_along_axis(
                    norms_stack, closest_idx[..., None], axis=0
                ).squeeze(0)
                p_wall_vel_best = jnp.take_along_axis(
                    vels_stack, closest_idx[..., None], axis=0
                ).squeeze(0)

                v_rel = mp_state.velocity_stack - p_wall_vel_best

                v_dot_n = jnp.einsum("ij,ij->i", v_rel, p_normal_best)[:, None]

                v_rel_slip = v_rel - jnp.minimum(0.0, v_dot_n) * p_normal_best
                v_ghost_p = p_wall_vel_best + v_rel_slip

                v_ghost_intr = v_ghost_p.at[intr_cache.point_ids].get()

                cpic_mask = intr_cache.cpic_mask[:, None]

                intr_vels = intr_vels * cpic_mask + v_ghost_intr * (1.0 - cpic_mask)
                intr_vels_nt = intr_vels_nt * cpic_mask + v_ghost_intr * (
                    1.0 - cpic_mask
                )
            # end cpic

            # Get weighted velocity, and velocity difference from grid
            intr_delta_vels = intr_vels_nt - intr_vels
            weighted_delta_vels = intr_cache.shape_vals[:, None] * intr_delta_vels
            weighted_vels_nt = intr_cache.shape_vals[:, None] * intr_vels_nt

            # Apply padding to velocities to compute shape function gradients in 3D
            # considering plane strain case
            padding = (0, 3 - grid_cache.dim)
            intr_vels_nt_3d = jnp.pad(intr_vels_nt, ((0, 0), padding))

            # Outer product to find velocity gradients (n_intr, 3, 3)
            # As L_ij = v_i * grad_j
            # compression positive convention for L, 
            weighted_L = jnp.einsum(
                "ij,ik->ijk", intr_cache.shape_grads, -intr_vels_nt_3d
            )

  
            # Particle velocity differences on grid
            p_delta_vel = (
                jnp.zeros((mp_state.num_points, grid_cache.dim))
                .at[intr_cache.point_ids]
                .add(weighted_delta_vels)
            )

            # Particle new velocity
            p_vel_nt = (
                jnp.zeros((mp_state.num_points, grid_cache.dim))
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
            if grid_cache.dim == 2:
                p_L = p_L.at[:, 2, 2].set(0.0)

            # compression positive
            F_inc = I - p_L * dt
            p_F_next = jnp.einsum("ijk,ikl->ijl", F_inc, mp_state.F_stack)

            if grid_cache.dim == 2:
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

        return world, mechanics, sim_cache

    def _constitutive_update(self, world, mechanics, sim_cache, dt, time):

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
        return world, mechanics, sim_cache
