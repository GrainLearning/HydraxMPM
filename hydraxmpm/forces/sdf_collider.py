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
        sdf_object: The SDF object defining the collider shape.
        g_idx_list: List of grid indices to apply the collider on.
        f_idx: Index of the SDF object state in the force states.
        friction: Coefficient of friction for collision response.
        gap: Margin of safety distance for collision detection.
    """

    sdf_object: SDFObjectBase = eqx.field(static=True)

    friction: float = eqx.field(static=True)

    # Indices
    g_idx_list: list[int] = eqx.field(static=True)
    f_idx: int = eqx.field(static=True)

    # margin of safety distance
    gap: float = eqx.field(static=True)

    def __init__(
        self,
        sdf_object,
        f_idx: int = 0,
        g_idx_list: list[int] = None,
        friction: float = 0.0,
        gap: float = 1e-4,
    ):
        """Initialize the SDFCollider with the given parameters."""
        if g_idx_list is None:
            g_idx_list = [0]  # select only first grid

        self.sdf_object = sdf_object

        self.g_idx_list = g_idx_list
        self.f_idx = f_idx
        self.gap = gap
        friction = jnp.clip(friction, 0.0, 1e9)
        self.friction = friction

    def create_state(
        self,
        center_of_mass: Float[Array, "dim"],
        velocity: Optional[Float[Array, "dim"]] = None,
        angular_velocity: Optional[float | Float[Array, ""] | Float[Array, "3"]] = None,
        rotation: Optional[float | Float[Array, ""] | Float[Array, "4"]] = None,
    ) -> Self:
        """ Helper function to create default SDFObjectState """
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

    
    def apply_kinematics(
        self, mp_states, grid_states, f_states, intr_caches, couplings, dt, time
    ):
        """Update the SDF object's state based on its velocity and angular velocity."""
        f_state = f_states[self.f_idx]
        dim = f_state.center_of_mass.shape[0]

        new_pos = f_state.center_of_mass + f_state.velocity * dt

        if dim == 2:
            new_rot = f_state.rotation + f_state.angular_velocity * dt
        else:
            # Quaternion Update to be implemented
            new_rot = f_state.rotation

        f_state = eqx.tree_at(
            lambda s: (s.center_of_mass, s.rotation), f_state, (new_pos, new_rot)
        )
        f_states[self.f_idx] = f_state
        return f_states

    def apply_grid_moments(
        self, mp_states, grid_states, f_states, intr_caches, couplings, dt, time
    ):
        """
        Projects grid momentum to satisfy the boundary condition.
        """
        f_state = f_states[self.f_idx]

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
            dis_stack = self.sdf_object.get_signed_distance_stack(f_state, flat_coords)

            # Uses AD to find normal by default
            normals_stack = self.sdf_object.get_normal_stack(f_state, flat_coords)

            # Handles linear and angular velocity, possibly other velocity like morphing
            v_object_stack = self.sdf_object.get_velocity_stack(f_state, flat_coords,dt)

            # Apply contact via vmap over all points to cover while domain
            new_vel = jax.vmap(self._collide_node)(
                dis_stack, vel, normals_stack, v_object_stack
            )

            # reconstruct momentum
            new_mom = new_vel * grid.mass_stack[:, None]

            # update grid State
            new_grid = eqx.tree_at(lambda g: g.moment_nt_stack, grid, new_mom)

            # update global grid state
            grid_states[g_idx] = new_grid

        return grid_states, f_states

    def _collide_node(self, dist, v_node, normal, v_object):
        """
        Calculates collision for a single node.
        """

        # Get velocity direction relative Velocity to object
        v_rel = v_node - v_object

        # v_n < 0 means moving INTO the wall
        v_n_mag = jnp.dot(v_rel, normal)

        # Check Inside/Touching AND Moving Inward
        # dist <= 0 implies we are behind the plane
        is_colliding = (dist <= self.gap) & (v_n_mag < 0.0)

        def handle_collision(v_in):
            # Decompose normal and tangential velocity
            v_n_vec = v_n_mag * normal
            v_t_vec = v_in - v_n_vec

            # Friction (Coulomb)
            # We apply an impulse to stop normal motion (Delta_vn = -v_n_mag)
            # Max Tangential Impulse <= mu * Normal Impulse
            # => Delta_vt <= mu * |v_n_mag|
            vt_mag = jnp.linalg.norm(v_t_vec)
            vn_impulse = jnp.abs(v_n_mag)

            # Calculate slip reduction
            # If friction is high, we subtract the full magnitude (Stick)
            # If friction is low, we subtract mu * vn
            reduction = self.friction * vn_impulse

            # New tangential magnitude (cannot go below zero)
            vt_new_mag = jnp.maximum(0.0, vt_mag - reduction)

            # rescale tangent vector
            v_t_new = v_t_vec * (vt_new_mag / (vt_mag + 1e-12))

            return v_t_new + v_object

        return jax.lax.cond(
            is_colliding,
            handle_collision, # Updates velocity
            lambda v: v,  # No collision, return original velocity
            v_node,
        )
