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
from .force import Force, BaseForceState
from ..utils.math_helpers import safe_norm


import jax
import jax.numpy as jnp


def apply_frictional_contact(
    v_in, dist, normal, v_wall, friction_coeff, dt, gap, bias_factor=0.0
):

    # relative velocity
    v_rel = v_in - v_wall

    # Normal component magnitude
    #  (positive if moving away, negative if approaching)
    v_n_mag = jnp.dot(v_rel, normal)

    # 2. Collision Criteria (Logic Gate)
    # We collide if we are within the gap AND moving towards the wall
    is_inside = dist <= gap
    # is_approaching = v_n_mag < 0.0
    should_collide = is_inside

    # ghost particles need to check this?
    # is_approaching = v_n_mag < 0.0
    # should_collide = is_inside & is_approaching

    def handle_collision():
        penalty_scale = 0.0
        # bias pushes out if we are penetrating
        # v_bias = (how much we are inside) / dt * factor
        v_push_out = jnp.maximum(0.0, (gap - dist) * 0.0)

        v_push_out = jnp.minimum(v_push_out, 1.0)
        dv_n_mag = jnp.maximum(0.0, -v_n_mag + v_push_out)

        # resolve normal velocity (non-penetration)
        v_corrected_n = v_in + dv_n_mag * normal

        v_rel_corrected = v_corrected_n - v_wall
        v_n_vec = jnp.dot(v_rel_corrected, normal) * normal

        v_t_vec = v_rel_corrected - v_n_vec
        vt_mag = safe_norm(v_t_vec)

        # Coulomb Law: Friction Limit
        friction_limit = friction_coeff * dv_n_mag
        reduction = jnp.where(vt_mag > 1e-20, jnp.minimum(vt_mag, friction_limit), 0.0)

        # Apply reduction safely
        v_t_frictional = v_t_vec * (1.0 - reduction / (vt_mag + 1e-20))

        return v_wall + v_n_vec + v_t_frictional

    return jax.lax.cond(should_collide, handle_collision, lambda: v_in)


class SDFColliderState(BaseForceState):
    pass


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

    def create_state(self) -> Self:
        return SDFColliderState()

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

            node_geom = sim_cache.node_geoms[(g_idx, self.sdf_idx)]

            # Get current  grid velocity
            inv_mass = jnp.where(
                grid_cache.mass_stack > 1e-14, 1.0 / grid_cache.mass_stack, 0.0
            )
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

        return apply_frictional_contact(
            v_node, dist, normal, v_object, friction, dt, self.gap, bias_factor=0.0
        )
