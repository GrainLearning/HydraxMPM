"""Module for imposing zero/non-zero boundaries via rigid particles."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax import Array

from jax.sharding import Sharding

from pymudokon.config.mpm_config import MPMConfig

from ..partition.grid_stencil_map import GridStencilMap
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction

import equinox as eqx

from pymudokon.shapefunctions import shapefunctions


class NodeLevelSet(eqx.Module):
    id_stack: chex.Array
    velocity_stack: chex.Array
    mu: float

    num_selected_cells: int = eqx.field(static=True, converter=lambda x: int(x))

    dim: int = eqx.field(static=True, converter=lambda x: int(x))

    dt: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self,
        config: MPMConfig,
        id_stack: chex.Array= None,
        velocity_stack: chex.Array = None,
        mu: float = 0.0,
        dt: float = 0.0,
        thickness = 2,
    ):
        """Initialize the rigid particles."""
        if config:
            dim = config.dim
            dt = config.dt
        else:
            if id_stack:
                dim = id_stack.shape[1]
            else:
                raise ValueError("Need to give either config or id_stack")

        if id_stack:
            self.id_stack = id_stack
        else: #creata a domain
            all_id_stack = (
                jnp.arange(config.num_cells).reshape(config.grid_size).astype(jnp.int32)
            )

            mask_id_stack = jnp.zeros_like(all_id_stack).astype(jnp.bool_)

            if dim == 2:
                # boundary layers
                mask_id_stack = mask_id_stack.at[0:thickness, :].set(True)  # x0
                mask_id_stack = mask_id_stack.at[:, 0:thickness].set(True)  # y0
                mask_id_stack = mask_id_stack.at[config.grid_size[0] - thickness :, :].set(
                    True
                )  # x1
                mask_id_stack = mask_id_stack.at[:, config.grid_size[1] - thickness :].set(
                    True
                )  # y1
            else:
                mask_id_stack = mask_id_stack.at[0:thickness, :, :].set(True)  # x0
                mask_id_stack = mask_id_stack.at[:, 0:thickness, :].set(True)  # y0
                mask_id_stack = mask_id_stack.at[:, :, 0:thickness].set(True)  # z0
                mask_id_stack = mask_id_stack.at[config.grid_size[0] - thickness :, :, :].set(
                    True
                )  # x1
                mask_id_stack = mask_id_stack.at[:, config.grid_size[1] - thickness :, :].set(
                    True
                )  # y1
                mask_id_stack = mask_id_stack.at[:, :, config.grid_size[2] - thickness :].set(
                    True
                )  # z1

            non_zero_ids = jnp.where(mask_id_stack.reshape(-1))[0]
            self.id_stack = all_id_stack.reshape(-1).at[non_zero_ids].get()


        self.dt = dt
        self.dim = dim
        self.num_selected_cells = self.id_stack.shape[0]

        if velocity_stack is None:
            velocity_stack = jnp.zeros((self.num_selected_cells, self.dim))

        self.velocity_stack = velocity_stack

        self.mu = mu

    def __call__(
        self: Self,
        nodes: Nodes,
        grid: GridStencilMap,
        particles: Particles,
        shapefunctions: ShapeFunction,
        step: int = 0,
    ):
        def vmap_p2g_grid_normals(p_id, c_id, w_id, carry):
            normal_prev = carry

            mass = particles.mass_stack.at[p_id].get()

            shapef_grad = shapefunctions.shapef_grad_stack.at[p_id, w_id].get()
            normal = (shapef_grad * mass).at[: self.dim].get()

            return normal_prev + normal

        normal_stack = grid.vmap_grid_gather_fori(
            vmap_p2g_grid_normals, jnp.zeros(self.dim)
        )

        @partial(
            jax.vmap,
            in_axes=(
                0,
                0,
            ),
        )
        def vmap_selected_nodes(n_id, levelset_vel):
            normal = normal_stack.at[n_id].get()
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

            if self.dim == 2:
                delta_vel_padded = jnp.pad(delta_vel, pad_width=(0, 1))
                norm_padded = jnp.pad(normal, pad_width=(0, 1))

                delta_vel_cross_normal = jnp.cross(
                    delta_vel_padded, norm_padded
                )  # works only for vectors of len 3
                norm_delta_vel_cross_normal = jnp.linalg.vector_norm(
                    delta_vel_cross_normal
                )

                omega = delta_vel_cross_normal / norm_delta_vel_cross_normal
                mu_prime = jnp.minimum(
                    self.mu, norm_delta_vel_cross_normal / delta_vel_dot_normal
                )

                normal_cross_omega = jnp.cross(
                    norm_padded, omega
                )  # works only for vectors of len 3

                tangent = (norm_padded + mu_prime * normal_cross_omega).at[:2].get()
            else:  #  3D
                delta_vel_cross_normal = jnp.cross(
                    delta_vel, normal
                )  # works only for vectors of len 3
                norm_delta_vel_cross_normal = jnp.linalg.vector_norm(
                    delta_vel_cross_normal
                )

                omega = delta_vel_cross_normal / norm_delta_vel_cross_normal
                mu_prime = jnp.minimum(
                    self.mu, norm_delta_vel_cross_normal / delta_vel_dot_normal
                )

                normal_cross_omega = jnp.cross(
                    normal, omega
                )  # works only for vectors of len 3

                tangent = normal + mu_prime * normal_cross_omega

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


@classmethod
def create_domain_box(cls: Self, nodes: Nodes, thickness: jnp.int32 = 2, mu=0.0):
    _, dim = nodes.moment_nt_stack.shape


    velocity_stack = jnp.zeros((id_stack.shape[0], dim))

    return cls(id_stack=id_stack, velocity_stack=velocity_stack, mu=mu)


# def distributed(self: Self, device: Sharding):
#     id_stack = jax.device_put(self.id_stack,device)
#     velocity_stack = jax.device_put(self.velocity_stack,device)
#     return self.replace(
#         id_stack = id_stack,
#         velocity_stack = velocity_stack
#     )

# def debug_plot_2d(
#     self,
#     nodes,
#     particles,
#     breaking =True
# ):
#     import matplotlib.pyplot as plt

#     aspect_x_y = nodes.grid_size.at[0].get()/ nodes.grid_size.at[1].get()
#     fig,ax = plt.subplots(figsize=(4*aspect_x_y,4))
#     nodes_position_stack = nodes.get_coordinate_stack(dim=2)
#     X, Y = nodes_position_stack.T

#     x, y= nodes_position_stack.at[self.id_stack].get().T

#     p_x, p_y = particles.position_stack.T
#     ax.scatter(
#         X,
#         Y,
#         s=1
#     )

#     ax.scatter(
#         x,
#         y,
#         s=2,
#         marker ="s",
#         c="red"
#     )

#     ax.scatter(
#         p_x,
#         p_y,
#         s=2,
#         marker ="s",
#         c="green"
#     )
#     plt.show()
#     if breaking:
#         exit()
