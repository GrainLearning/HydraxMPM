# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
"""
Explaination:
    This module contains the Signed Distance Function (SDF) based collider logic for MPM simulations.

    It uses `SDFObjects` to define boundaries and applies collision responses on grid nodes.

"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Self, Optional

from jaxtyping import Float, Array

from ..sdf.sdfobject import SDFObjectBase, SDFObjectState
from .force import Force


import jax
import jax.numpy as jnp

def apply_frictional_contact(
    v_in, dist, normal, v_wall, friction_coeff, dt, gap, bias_factor=0.0
):
    """
    Standard MPM Frictional Contact Logic.
    Works for both Grid Nodes and Particle Ghost Velocities.
    """
    v_rel = v_in - v_wall
    v_n_mag = jnp.dot(v_rel, normal)

    # 1. Normal Impulse (Kinematic)
    # The velocity needed to reach the 'gap' distance in one timestep
    v_bias = jnp.maximum(0.0, (gap - dist) / dt * bias_factor)
    delta_v_kinematic = jnp.maximum(0.0, v_bias - v_n_mag)


    # 3. Resolve Normal Velocity (Non-penetration)
    v_corrected_n = v_in + delta_v_kinematic * normal

    # 4. Resolve Tangential Velocity (Friction)
    v_rel_corrected = v_corrected_n - v_wall
    v_n_vec = jnp.dot(v_rel_corrected, normal) * normal
    v_t_vec = v_rel_corrected - v_n_vec
    vt_mag = jnp.linalg.norm(v_t_vec)

    # Coulomb Law: Friction Limit
    friction_limit = friction_coeff * delta_v_kinematic
    reduction = jnp.where(vt_mag > 1e-12, jnp.minimum(vt_mag, friction_limit), 0.0)
    
    # Apply reduction safely
    v_t_frictional = v_t_vec * (1.0 - reduction / (vt_mag + 1e-12))

    # Final Velocity = Wall + Normal Component + Frictional Tangent
    return v_wall + v_n_vec + v_t_frictional

class SDFCollider(Force):
    """
    SDF Collider Force for MPM Simulations.

    Attributes:
        sdf_logic: The SDF object defining the collider shape.
        g_idx_list: List of grid indices to apply the collider on.
        f_idx: Index of the SDF object state in the force states.
        friction: Coefficient of friction for collision response.
        gap: Margin of safety distance for collision detection.
    """

    # Indices
    g_idx_list: list[int] = eqx.field(static=True)
    sdf_idx: int = eqx.field(static=True)

    # margin of safety distance
    gap: float = eqx.field(static=True)


    base_friction: float = eqx.field(static=True, default=1.0)

    def __init__(
        self,
        sdf_idx: int = 0,
        g_idx_list: list[int] = None,
        gap: float = 1e-4,
        friction: float = 1.0,


    ):
        """Initialize the SDFCollider with the given parameters."""
        if g_idx_list is None:
            g_idx_list = [0]  # select only first grid

        self.g_idx_list = g_idx_list
        self.sdf_idx = sdf_idx
        self.gap = gap
        self.base_friction = friction

     


    def create_state(
        self,
        center_of_mass: Float[Array, "dim"],
        velocity: Optional[Float[Array, "dim"]] = None,
        angular_velocity: Optional[float | Float[Array, ""] | Float[Array, "3"]] = None,
        rotation: Optional[float | Float[Array, ""] | Float[Array, "4"]] = None,
    ) -> Self:
        """Helper function to create default SDFObjectState"""
        dim = center_of_mass.shape[0]
        if velocity is None:
            velocity = jnp.zeros(dim)

        if dim == 2:
            if angular_velocity is None:
                angular_velocity = 0.0
            if rotation is None:
                rotation = 0.0
        else:
            if angular_velocity is None:
                angular_velocity = jnp.zeros(3)
            if rotation is None:
                rotation = jnp.array([1.0, 0.0, 0.0, 0.0])
        return SDFObjectState(
            center_of_mass=center_of_mass,
            velocity=velocity,
            angular_velocity=angular_velocity,
            rotation=rotation,
        )

    def apply_grid_moments(
      self, world, mechanics, sim_cache, sdf_logics, couplings, grid_domains, dt, time
            ):
        """
        Projects grid momentum to satisfy the boundary condition.
        """
        # return world, mechanics
        if self.sdf_idx == -1:
            return world, mechanics, sim_cache

        grid_caches = list(sim_cache.grids)
    
        sdf_logic = sdf_logics[self.sdf_idx]
        
        sdf_state = list(world.sdfs)[self.sdf_idx]


        for g_idx in self.g_idx_list:

            grid_domain = grid_domains[g_idx]
            grid_cache = grid_caches[g_idx]

            node_geom =sim_cache.node_geoms[(g_idx, self.sdf_idx)]

            # Generate node coordinates compatible with C-Contiguous (Row-Major) layout
            # (Nx, Ny, Nz, 3)
            # indices = jnp.indices(grid_domain.grid_size, dtype=jnp.float32)
            # coords = jnp.moveaxis(indices, 0, -1) * grid_domain.cell_size + jnp.array(
            #     grid_domain.origin
            # )
            # flat_coords = coords.reshape(-1, grid_domain.dim)

            # Get current  grid velocity
            inv_mass = jnp.where(grid_cache.mass_stack > 1e-14, 1.0 / grid_cache.mass_stack, 0.0)
            vel = grid_cache.moment_nt_stack * inv_mass[:, None]

            # Compute quantities from SDF object

            # SDF check Penetration
            # dis_stack = sdf_logic.get_signed_distance_stack(sdf_state, node_geom.coords) 
            dis_stack = node_geom.dists
            # Uses AD to find normal by default
            # normals_stack = sdf_logic.get_normal_stack(sdf_state, flat_coords)
            normals_stack = node_geom.normals
            # Handles linear and angular velocity, possibly other velocity like morphing
            # v_object_stack = sdf_logic.get_velocity_stack(
            #     sdf_state, flat_coords, dt
            # )
            v_object_stack = node_geom.wall_vels

            # query the SDF for material property at this location (possibly spatial varying)
            # local_friction_stack = sdf_logic.get_surface_friction_stack(sdf_state, flat_coords)
            local_friction_stack = node_geom.friction
            friction_stack = self.base_friction * local_friction_stack
            
            # Apply contact via vmap over all points to cover while domain
            new_vel = jax.vmap(self._collide_node, in_axes=(0, 0, 0, 0, 0, None))(
                dis_stack, vel, normals_stack, v_object_stack, friction_stack, dt
            )

            # reconstruct momentum
            new_mom = new_vel * grid_cache.mass_stack[:, None]

            # update grid State
            new_grid = eqx.tree_at(lambda g: g.moment_nt_stack, grid_cache, new_mom)

            # update global grid state
            grid_caches[g_idx] = new_grid

    
        sim_cache = eqx.tree_at(
            lambda s: (s.grids,),
            sim_cache,
            (tuple(grid_caches),),
        )

        return world, mechanics, sim_cache

    def _collide_node(self, dist, v_node, normal, v_object, friction, dt):
        """
        Calculates collision for a single node.
        """

        # v_n < 0 means moving INTO the wall
        v_rel = v_node - v_object
        v_n_mag = jnp.dot(v_rel, normal)
        # Check Inside/Touching AND Moving Inward
        # dist <= 0 implies we are behind the plane

        is_inside = dist <= 0.0

        is_approaching = (dist <= 0.0) & (v_n_mag < 0.0)
        should_collide = is_inside | is_approaching

        def handle_collision(v_in):
            return apply_frictional_contact(
            v_in, dist, normal, v_object, friction, dt, self.gap, bias_factor=0.0
            )

        return jax.lax.cond(
            should_collide,
            handle_collision,  # Updates velocity
            lambda v: v,  # No collision, return original velocity
            v_node,
        )

