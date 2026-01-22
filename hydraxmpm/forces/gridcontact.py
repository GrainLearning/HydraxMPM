# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM
"""
Explanation:
    This module contains grid-to-grid contact model.

    It serves two purposes:
    1. Handle collision between two deformable bodies (Multi-Grid FSI).
    2. Handle collision between rigid body and deformable body

    It uses mass gradient to estimate surface normals. 

    Two way coupling is supported via reaction forces.

"""
import equinox as eqx
import jax
import jax.numpy as jnp
from .force import Force

from ..sdf.sdfcollection import MaterialPointCloudSDF

from typing_extensions import Optional

class GridContact(Force):
    """
    Grid-to-Grid Contact Model.
    
    Attributes:
        couple_idx_actor: Index of the coupling representing the actor (obstacle).
        couple_idx_receiver: Index of the coupling representing the receiver (material body).
        friction: Coefficient of friction for contact response.
        is_reaction: Whether to apply reaction forces to the actor.
        is_rigid: Whether the actor grid represents a rigid body.
        mass_ratio_limit: Limit on mass ratio for reaction force scaling (stability)
    """

    # connectivity
    couple_idx_actor: int = eqx.field(static=True) # Obstacle
    couple_idx_receiver: int = eqx.field(static=True)  # Material Body

    # physics
    friction: float = 0.0
    is_reaction: bool = eqx.field(static=True, default =False)
    is_rigid: bool = eqx.field(static=True, default =False)
    
    # stability
    mass_ratio_limit: float = eqx.field(static=True, default=2.0)
    def apply_grid_moments(self, world, mechanics, sim_cache, sdf_logics, couplings, dt, time):
        """"
        Apply grid contact between two grids.
        """
        # Get actor and receivor indices and unpack grid
        p_idx_actor = couplings[self.couple_idx_actor].p_idx
        g_idx_actor = couplings[self.couple_idx_actor].g_idx

        p_idx_receiver = couplings[self.couple_idx_receiver].p_idx
        g_idx_receiver = couplings[self.couple_idx_receiver].g_idx

        grid_state_act = world.grids[g_idx_actor] 
        grid_state_rec = world.grids[g_idx_receiver]

        # Buffers for actor quantities (Mass, Momentum, Normal)
        # We work on copies/buffers because we might alter rigid Body data
        act_mass_stack = grid_state_act.mass_stack
        act_mom_stack = grid_state_act.moment_nt_stack

        # buffer for accumulating explicit normals from Rigid Bodies (if given by user)
        act_normal_stack = jnp.zeros_like(act_mom_stack)

        # update actor grid if its rigid body
        if self.is_rigid:
            # Here we calculate rigid momentum and mass
            rb_mp_state = world.material_points[p_idx_actor]
            rb_intr_cache = mechanics.interactions[(p_idx_actor, g_idx_actor)]
            
            rb_intr_masses_stack = rb_mp_state.mass_stack.at[rb_intr_cache.point_ids].get()
            rb_intr_mom_nt_stack = (rb_mp_state.mass_stack[:, None] * rb_mp_state.velocity_stack).at[rb_intr_cache.point_ids].get()
            
            
            act_mass_stack = jnp.zeros_like(grid_state_act.mass_stack).at[rb_intr_cache.node_hashes].add(rb_intr_masses_stack)
            act_mom_stack = jnp.zeros_like(grid_state_act.moment_nt_stack).at[rb_intr_cache.node_hashes].add(rb_intr_mom_nt_stack)
            
            if self.is_reaction:
                grid_state_act = eqx.tree_at(
                lambda g: (g.mass_stack, g.moment_nt_stack), 
                grid_state_act, (act_mass_stack, act_mom_stack),
                )
            
            # explicit projections of normals to prevent "blobby" artifacts
            if hasattr(rb_mp_state, 'normals_stack') and rb_mp_state.normals_stack is not None:
                rb_intr_normal_stack = ( rb_mp_state.mass_stack[:, None] * rb_mp_state.normals_stack).at[self.shape_map.point_ids].get()
                act_normal_stack = jnp.zeros_like(grid_state_rec.moment_nt_stack).at[self.shape_map.node_ids].add(rb_intr_normal_stack)
            else:
                # Fallback if user didn't provide normals on particles
                act_normal_stack = -self._compute_mass_gradient(
                    grid_state_act,
                    act_mass_stack
                    )
          
        else:
            # act_normal_stack = -self._compute_mass_gradient(grid_state_act)
            particle_cloud = MaterialPointCloudSDF(smooth_k=0.0)
            
            indices = jnp.indices(grid_state_act.grid_size, dtype=jnp.float32)
            coords = jnp.moveaxis(indices, 0, -1) * grid_state_act.cell_size + jnp.array(
                grid_state_act.origin
            )
            flat_coords = coords.reshape(-1, grid_state_act.dim)

            dis_stack = particle_cloud.get_signed_distance_stack(
                flat_coords,
                world.material_points[p_idx_actor],
            )
            act_normal_stack = particle_cloud.get_normal_stack(
                    flat_coords,
                    world.material_points[p_idx_actor],
                )
            # act_normal_stack = particle_cloud.get_normal_stack(
            #     particle_cloud_state,
            #     flat_coords
            # )
            # dis_stack = particle_cloud.get_signed_distance_stack(particle_cloud_state, flat_coords)


        # Calculate grid velocities of actor and receiver
        # norm_mag = jnp.linalg.norm(act_normal_stack, axis=1, keepdims=True) + 1e-12

        # ensure unit length
        # act_normal_stack = jnp.where(norm_mag > 1e-4, act_normal_stack / norm_mag, 0.0)


        new_rec_mom, new_act_mom = jax.vmap(self._solve_node_collision)(
            grid_state_rec.moment_nt_stack, 
            grid_state_rec.mass_stack, 
            act_mom_stack,
            act_mass_stack,     
            act_normal_stack,
            dis_stack
        )

        new_grids = list(world.grids)
        
        # always update receiver moments
        new_grids[g_idx_receiver] = eqx.tree_at(
            lambda g: g.moment_nt_stack, grid_state_rec, new_rec_mom
        )

        # Only update Actor if it's NOT rigid and reaction is enabled
        if self.is_reaction and not self.is_rigid:
            new_grids[g_idx_actor] = eqx.tree_at(
                lambda g: g.moment_nt_stack, grid_state_act, new_act_mom
            )
        world =  eqx.tree_at(
            lambda w: (w.grids,),
            world,
            (tuple(new_grids),),
        )
        return world, mechanics, sim_cache

    def _compute_mass_gradient(self, grid_state,mass_grid= None ):
        """
        Computes finite difference gradient of mass on the grid.
        """
        # This can be improved with AD?
        
        
        grid_size = grid_state.grid_size
        
        dim = grid_state.dim
        
        # Reshape to grid
        if mass_grid is None:
            mass_grid = grid_state.mass_stack

        mass_grid = mass_grid.reshape(grid_size)
        
        grads = []
        for d in range(dim):
            # Central difference: (f(x+1) - f(x-1)) / 2dx
            # we use jnp.gradient which handles boundaries nicely
            grad_d = jnp.gradient(mass_grid, axis=d)
            grads.append(grad_d.flatten())
            
        return jnp.stack(grads, axis=-1)

    def _solve_node_collision(self, mom_rec, mass_rec, mom_act, mass_act, normal_act, dist):
        """
        Compute node-level collision response between receiver and actor.
        """
        # Check if mass is sufficient
        # has_contact = (mass_rec > 1e-9) & (mass_act > 1e-9)

        has_contact = dist <= 1e-6
        
        def apply_collision(_):
            # Get velocities, relative Velocity (Rec - Act)
            v_rec = mom_rec / (mass_rec + 1e-12)
            v_act = mom_act / (mass_act + 1e-12)
            
            v_rel = v_rec - v_act
            
            # penetration (v_rel projected onto Normal)
            # Normal points OUT of Actor. v_rel opposing normal means entering.
            v_n = jnp.dot(v_rel, normal_act)
            
            is_penetrating = v_n < 0.0
            
            def resolve_penetration(_):
                # We want v_n_new = 0 (Slide)
                # delta_v_n = -v_n

                if self.is_rigid:
                    # Rigid Wall. Wall doesn't move, so receiver does 100% of the correction.
                    v_rec_new = v_rec - (v_n * normal_act)
                    v_act_new = v_act 
                else:
                    # With two soft bodies.We distribute correction based on mass.
                    m_ratio = mass_rec / (mass_act + 1e-12)
                    # safe_ratio = jnp.minimum(m_ratio, self.mass_ratio_limit)
                    safe_ratio = m_ratio
                    
                    # Impulse                     
                    v_rec_new = v_rec - (v_n * normal_act)
                    
                    # Action and reaction
                    v_act_new = v_act + (v_n * normal_act) * safe_ratio

                return v_rec_new * mass_rec, v_act_new * mass_act

            return jax.lax.cond(is_penetrating, resolve_penetration, lambda _: (mom_rec, mom_act), None)

        return jax.lax.cond(
            has_contact, 
            apply_collision, 
            lambda _: (mom_rec, mom_act), 
            None
        )