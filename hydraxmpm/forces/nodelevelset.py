"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from .forces import Forces


class NodeLevelSet(Forces):
    id_stack: chex.Array
    velocity_stack: chex.Array
    mu: float

    num_selected_cells: int = eqx.field(static=True, converter=lambda x: int(x))

    def __init__(
        self,
        config: MPMConfig,
        id_stack: chex.Array = None,
        velocity_stack: chex.Array = None,
        mu: float = 0.0,
        thickness=2,
    ):
        """Initialize the rigid particles."""

        if id_stack is not None:
            self.id_stack = id_stack
        else:  # creata a domain
            all_id_stack = (
                jnp.arange(config.num_cells)
                .reshape(config.grid_size)
                .astype(jnp.uint32)
            )

            mask_id_stack = jnp.zeros_like(all_id_stack).astype(jnp.bool_)

            if config.dim == 2:
                # boundary layers
                mask_id_stack = mask_id_stack.at[0:thickness, :].set(True)  # x0
                mask_id_stack = mask_id_stack.at[:, 0:thickness].set(True)  # y0
                mask_id_stack = mask_id_stack.at[
                    config.grid_size[0] - thickness :, :
                ].set(True)  # x1
                mask_id_stack = mask_id_stack.at[
                    :, config.grid_size[1] - thickness :
                ].set(True)  # y1
            else:
                mask_id_stack = mask_id_stack.at[0:thickness, :, :].set(True)  # x0
                mask_id_stack = mask_id_stack.at[:, 0:thickness, :].set(True)  # y0
                mask_id_stack = mask_id_stack.at[:, :, 0:thickness].set(True)  # z0
                mask_id_stack = mask_id_stack.at[
                    config.grid_size[0] - thickness :, :, :
                ].set(True)  # x1
                mask_id_stack = mask_id_stack.at[
                    :, config.grid_size[1] - thickness :, :
                ].set(True)  # y1
                mask_id_stack = mask_id_stack.at[
                    :, :, config.grid_size[2] - thickness :
                ].set(True)  # z1

            non_zero_ids = jnp.where(mask_id_stack.reshape(-1))[0]
            self.id_stack = all_id_stack.reshape(-1).at[non_zero_ids].get()

        self.num_selected_cells = self.id_stack.shape[0]

        if velocity_stack is None:
            velocity_stack = jnp.zeros((self.num_selected_cells, config.dim))

        self.velocity_stack = velocity_stack

        self.mu = mu

        super().__init__(config)

    def apply_on_nodes(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        step: int = 0,
    ):
        
        @partial(jax.vmap, in_axes=(0, 0))
        def vmap_selected_nodes(n_id, levelset_vel):
            normal = nodes.normal_stack.at[n_id].get()
            moment_nt = nodes.moment_nt_stack.at[n_id].get()
            mass = nodes.mass_stack.at[n_id].get()

            # skip the nodes with small mass, due to numerical instability
            vel_nt = jax.lax.cond(
                mass > nodes.small_mass_cutoff,
                lambda x: x / mass,
                lambda x: jnp.zeros_like(x),
                moment_nt,
            )

            # normalize the normals
            normal = jax.lax.cond(
                mass > nodes.small_mass_cutoff,
                lambda x: x / jnp.linalg.vector_norm(x),
                lambda x: jnp.zeros_like(x),
                normal,
            )

            # check if the velocity direction of the normal and apply contact
            # dot product is 0 when the vectors are orthogonal
            # and 1 when they are parallel
            # if othogonal no contact is happening
            # if parallel the contact is happening
            delta_vel = vel_nt - levelset_vel

            delta_vel_dot_normal = jnp.dot(delta_vel, normal)

            delta_vel_padded = jnp.pad(
                delta_vel,
                self.config.padding,
                mode="constant",
                constant_values=0,
            )

            norm_padded = jnp.pad(
                normal,
                self.config.padding,
                mode="constant",
                constant_values=0,
            )
            delta_vel_cross_normal = jnp.cross(
                delta_vel_padded, norm_padded
            )  # works only for vectors of len 3
            norm_delta_vel_cross_normal = jnp.linalg.vector_norm(delta_vel_cross_normal)

            omega = delta_vel_cross_normal / norm_delta_vel_cross_normal
            mu_prime = jnp.minimum(
                self.mu, norm_delta_vel_cross_normal / delta_vel_dot_normal
            )

            normal_cross_omega = jnp.cross(
                norm_padded, omega
            )  # works only for vectors of len 3

            tangent = (
                (norm_padded + mu_prime * normal_cross_omega)
                .at[: self.config.dim]
                .get()
            )

            # sometimes tangent become nan if velocity is zero at initialization
            # which causes problems
            tangent = jnp.nan_to_num(tangent)

            new_nodes_vel_nt = jax.lax.cond(
                delta_vel_dot_normal > 0.0,
                lambda x: x - delta_vel_dot_normal * tangent,
                # lambda x: x - delta_vel_dot_normal*node_normals, # no friction debug
                lambda x: x,
                vel_nt,
            )

            node_moments_nt = new_nodes_vel_nt * mass
            return node_moments_nt

        levelset_moment_stack = vmap_selected_nodes(self.id_stack, self.velocity_stack)

        new_nodes = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            nodes,
            (nodes.moment_nt_stack.at[self.id_stack].set(levelset_moment_stack)),
        )

        return new_nodes, self
