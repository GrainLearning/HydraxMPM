"""Gravity force on Nodes."""

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


# @partial(jax.jit, static_argnames=["boundary_types"])



@struct.dataclass
class DirichletBox:
    """Dirichlet boundary conditions with user defined values.

    Attributes:
        type mask (Array):
            type mask in (X,Y,Z) space on where to apply the Dirichlet boundary conditions.
            - 0 is not applied
            - 1 is fixed
            - 2 max
            - 3 min
            Shape is `(num_nodes, dim)`.
        values (Array):
            values of shape `(num_nodes, dim)` to apply on the nodes.
    """

    boundary_types: jnp.array
    stick_ids: jnp.array
    slip_ids: jnp.array
    width: int

    @classmethod
    def create(cls: Self, nodes, boundary_types: List = None, width: int = 1) -> Self:
        """Register the Dirichlet nodes."""

        dim = nodes.origin.shape[0]

        if boundary_types is None:
            boundary_types = jnp.ones(dim).repeat(2).reshape(dim, 2).astype(jnp.int32)

        node_types = jnp.zeros(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jnp.int32)
        print(node_types.shape, boundary_types.shape)
        if dim == 3:
            node_types = node_types.at[0:width, :, :].set(boundary_types[0, 0])
            node_types = node_types.at[-width:, :, :].set(boundary_types[0, 1])
            node_types = node_types.at[:, 0:width, :].set(boundary_types[1, 0])
            node_types = node_types.at[:, -width:, :].set(boundary_types[1, 1])
            node_types = node_types.at[:, :, 0:width].set(boundary_types[2, 0])
            node_types = node_types.at[:, :, -width:].set(boundary_types[2, 1])
        elif dim == 2:
            node_types = node_types.at[0:width, :].set(boundary_types[0, 0])
            node_types = node_types.at[-width:, :].set(boundary_types[0, 1])
            node_types = node_types.at[:, 0:width].set(boundary_types[1, 0])
            node_types = node_types.at[:, -width:].set(boundary_types[1, 1])
        else:
            raise ValueError("Only 2D and 3D are supported")

        node_ids = jnp.arange(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jax.numpy.int32)
        stick_ids = node_ids.at[jnp.where(node_types == 1)].get()
        slip_ids = node_ids.at[jnp.where(node_types == 2)].get()


        return cls(
            boundary_types.astype(jnp.int32),
            stick_ids=stick_ids,
            slip_ids=slip_ids,
            width=width)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""


        moments_nt = nodes.moments_nt.at[self.stick_ids].set(0.0)
        # moments = moments_nt.at[self.stick_ids].set(0.0)

        return nodes.replace(moments_nt=moments_nt), self
        # moments, moments_nt = apply_boundary_box(nodes.moments, nodes.moments_nt, nodes.ids_grid, self.boundary_types)

        #  get fixed ids
        # fixed_ids = jnp.where(self.boundary_types == 1)
        # dim  = nodes.origin.shape[0]
        # node_types = jnp.zeros(nodes.num_nodes_total).reshape(nodes.grid_size).repeat(dim).astype(jnp.int32)
        # node_ids = jnp.arange(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jax.numpy.int32)

#         def apply_boundary_box(
#     moments: Array, moments_nt: Array, ids_grid: Array, boundary_types: Array
# ) -> Tuple[Array, Array]:
#     """Apply boundary conditions on the nodes, based on the boundary types.

#     Args:
#         moments (Array): nodal moments `(num_nodes, dim)`.
#         moments_nt (Array): nodal moments `(num_nodes, dim)`.
#         ids_grid (Array): grid ids `(num_nodes_x, num_nodes_y,...)`.
#         boundary_types (Array): boundary types `(3, 2)`.

#     Returns:
#         Tuple[Array, Array]: Updated moments and moments_nt
#     """

#     def vmap_nodes_boundary(
#         selected_moments: Array,
#         selected_moments_nt: Array,
#         boundary_type:Array,
#         axis: int,
#         edge: int,
#         ):

#         if boundary_type == 1:
#             moments = selected_moments.at[0, :].set(0.0)
#             moments_nt = selected_moments_nt.at[0, :].set(0.0)
#         return moments, moments_nt

#         moments = jax.lax.cond(
#         boundary_types.at[0, 0].get() == 1,
#         lambda x: x.at[x0, :].set(0.0),
#         lambda x: x,
#         operand=moments,
#         )


#     # def vmap_nodes_apply(
#     #         moments: Array,
#     #         moments_nt: Array,

#     # )
#     # # TODO: Add support for 3D
#     # # TODO: Add support for slip/stick

#     # x0 = ids_grid.at[0, :].get().reshape(-1)
#     # x1 = ids_grid.at[-1, :].get().reshape(-1)

#     # # find ids at y0
#     # y0 = ids_grid.at[:, 0].get().reshape(-1)
#     # y1 = ids_grid.at[:, -1].get().reshape(-1)

#     # moments = jax.lax.cond(
#     #     boundary_types.at[0, 0].get() == 1,
#     #     lambda x: x.at[x0, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments,
#     # )
#     # moments = jax.lax.cond(
#     #     boundary_types.at[0, 1].get() == 1,
#     #     lambda x: x.at[x1, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments,
#     # )
#     # moments = jax.lax.cond(
#     #     boundary_types.at[1, 0].get() == 1,
#     #     lambda x: x.at[y0, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments,
#     # )
#     # moments = jax.lax.cond(
#     #     boundary_types.at[1, 1].get() == 1,
#     #     lambda x: x.at[y1, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments,
#     # )
#     # ##
#     # moments_nt = jax.lax.cond(
#     #     boundary_types.at[0, 0].get() == 1,
#     #     lambda x: x.at[x0, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments_nt,
#     # )
#     # moments_nt = jax.lax.cond(
#     #     boundary_types.at[0, 1].get() == 1,
#     #     lambda x: x.at[x1, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments_nt,
#     # )
#     # moments_nt = jax.lax.cond(
#     #     boundary_types.at[1, 0].get() == 1,
#     #     lambda x: x.at[y0, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments_nt,
#     # )
#     # moments_nt = jax.lax.cond(
#     #     boundary_types.at[1, 1].get() == 1,
#     #     lambda x: x.at[y1, :].set(0.0),
#     #     lambda x: x,
#     #     operand=moments_nt,
#     # )

#     # return moments, moments_nt
#         return nodes.replace(moments=moments, moments_nt=moments_nt), self
