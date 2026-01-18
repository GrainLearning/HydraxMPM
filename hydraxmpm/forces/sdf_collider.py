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
        friction: float = 1.0
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

    def apply_grid_moments(self, world, mechanics, sdf_logics, couplings, dt, time):
        """
        Projects grid momentum to satisfy the boundary condition.
        """
        # return world, mechanics

        grid_states = list(world.grids)
    
        sdf_logic = sdf_logics[self.sdf_idx]
        
        sdf_state = list(world.sdfs)[self.sdf_idx]

        for g_idx in self.g_idx_list:

            grid = grid_states[g_idx]

            # Generate node coordinates compatible with C-Contiguous (Row-Major) layout
            # (Nx, Ny, Nz, 3)
            indices = jnp.indices(grid.grid_size, dtype=jnp.float32)
            coords = jnp.moveaxis(indices, 0, -1) * grid.cell_size + jnp.array(
                grid.origin
            )
            flat_coords = coords.reshape(-1, grid.dim)

            # Get current  grid velocity
            inv_mass = jnp.where(grid.mass_stack > 1e-14, 1.0 / grid.mass_stack, 0.0)
            vel = grid.moment_nt_stack * inv_mass[:, None]

            # Compute quantities from SDF object

            # SDF check Penetration
            dis_stack = sdf_logic.get_signed_distance_stack(sdf_state, flat_coords)

            # Uses AD to find normal by default
            normals_stack = sdf_logic.get_normal_stack(sdf_state, flat_coords)

            # Handles linear and angular velocity, possibly other velocity like morphing
            v_object_stack = sdf_logic.get_velocity_stack(
                sdf_state, flat_coords, dt
            )

            # query the SDF for material property at this location (possibly spatial varying)
            local_friction_stack = sdf_logic.get_surface_friction_stack(sdf_state, flat_coords)

            friction_stack = self.base_friction * local_friction_stack

            # Apply contact via vmap over all points to cover while domain
            new_vel = jax.vmap(self._collide_node, in_axes=(0, 0, 0, 0, 0, None))(
                dis_stack, vel, normals_stack, v_object_stack, friction_stack, dt
            )

            # reconstruct momentum
            new_mom = new_vel * grid.mass_stack[:, None]

            # update grid State
            new_grid = eqx.tree_at(lambda g: g.moment_nt_stack, grid, new_mom)

            # update global grid state
            grid_states[g_idx] = new_grid

    
        world = eqx.tree_at(
            lambda w: (w.grids,),
            world,
            (tuple(grid_states),),
        )

        return world, mechanics

    def _collide_node(self, dist, v_node, normal, v_object, friction, dt):
        """
        Calculates collision for a single node.
        """

        # v_n < 0 means moving INTO the wall

        # Check Inside/Touching AND Moving Inward
        # dist <= 0 implies we are behind the plane
        is_colliding = dist <= self.gap
        # & (v_n_mag < 0.0)

        def handle_collision(v_in):
            
            # Check if we  are  inside the gap
            # add velocity bias to push us out
            # by next timestep
            # bias = (overlap)/dt * stiffness
            # ensure we dont apply negative bias (suction)
            bias_factor = 0.0  # good values (0.1 - 0.5)
            overlap = self.gap - dist
            v_bias = (overlap / dt) * bias_factor
            v_bias = jnp.maximum(0.0, v_bias)

            # Solve normal velocity
            # Get velocity direction relative Velocity to object
            v_rel = v_node - v_object
            v_n_mag = jnp.dot(v_rel, normal)
            delta_v_stop = jnp.maximum(0.0, -v_n_mag)

            # calculate the impulse to reach v_bias
            # if current velocity is already > v_bias,
            # we are moving fast enough
            delta_v_n = v_bias - v_n_mag
            delta_v_n = jnp.maximum(0.0, delta_v_n)

            # apply impuse to velocity
            v_corrected = v_in + delta_v_n * normal

            # Friction
            # Decompose normal and tangential velocity
            v_rel_corrected = v_corrected - v_object
            v_n_new = jnp.dot(v_rel_corrected, normal)
            v_t_vec = v_rel_corrected - v_n_new * normal

            vt_mag = jnp.linalg.norm(v_t_vec)

            # Friction limits based on the Normal Impulse we just applied
            # The "Normal Force" is proportional to delta_v_n / dt
            friction_impulse_max = friction * delta_v_stop

            # Calculate how much we can reduce tangential velocity
            reduction = jnp.minimum(vt_mag, friction_impulse_max)

            # Apply reduction
            v_t_new = v_t_vec * (1.0 - reduction / (vt_mag + 1e-12))

            # Reassemble
            return v_object + v_n_new * normal + v_t_new

            # Friction limits based on the Normal Impulse we just applied
            # The "Normal Force" is proportional to delta_v_n / dt
            # So we can just scale velocities directly.
            # friction_impulse_max = self.friction * delta_v_n

            # # Friction (Coulomb)
            # # We apply an impulse to stop normal motion (Delta_vn = -v_n_mag)
            # # Max Tangential Impulse <= mu * Normal Impulse
            # # => Delta_vt <= mu * |v_n_mag|
            # vt_mag = jnp.linalg.norm(v_t_vec)
            # vn_impulse = jnp.abs(v_n_mag)

            # # Calculate slip reduction
            # # If friction is high, we subtract the full magnitude (Stick)
            # # If friction is low, we subtract mu * vn
            # reduction = self.friction * vn_impulse

            # # New tangential magnitude (cannot go below zero)
            # vt_new_mag = jnp.maximum(0.0, vt_mag - reduction)

            # # rescale tangent vector
            # v_t_new = v_t_vec * (vt_new_mag / (vt_mag + 1e-12))

            # return v_t_new + v_object

        return jax.lax.cond(
            is_colliding,
            handle_collision,  # Updates velocity
            lambda v: v,  # No collision, return original velocity
            v_node,
        )
