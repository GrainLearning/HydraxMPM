"""Gravity force on Nodes."""

from functools import partial
from typing import List, Tuple
from flax import struct
import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction

from functools import partial


@partial(jax.jit, static_argnames=["boundary_types"])
def apply_boundary_box(
    moments: Array, moments_nt: Array, ids_grid: Array, boundary_types: Array
) -> Tuple[Array, Array]:
    """Apply boundary conditions on the nodes, based on the boundary types.

    Args:
        moments (Array): nodal moments `(num_nodes, dim)`.
        moments_nt (Array): nodal moments `(num_nodes, dim)`.
        ids_grid (Array): grid ids `(num_nodes_x, num_nodes_y,...)`.
        boundary_types (Array): boundary types `(3, 2)`.

    Returns:
        Tuple[Array, Array]: Updated moments and moments_nt
    """
    # TODO: Add support for 3D
    # TODO: Add support for slip/stick

    x0 = ids_grid.at[0, :].get().reshape(-1)
    x1 = ids_grid.at[-1, :].get().reshape(-1)

    # find ids at y0
    y0 = ids_grid.at[:, 0].get().reshape(-1)
    y1 = ids_grid.at[:, -1].get().reshape(-1)
    
    moments = jax.lax.cond(
        boundary_types.at[0, 0].get() == 1,
        lambda x: x.at[x0, :].set(0.0),
        lambda x: x,
        operand=moments,
    )
    moments = jax.lax.cond(
        boundary_types.at[0, 1].get() == 1,
        lambda x: x.at[x1, :].set(0.0),
        lambda x: x,
        operand=moments,
    )
    moments = jax.lax.cond(
        boundary_types.at[1, 0].get() == 1,
        lambda x: x.at[y0, :].set(0.0),
        lambda x: x,
        operand=moments,
    )
    moments = jax.lax.cond(
        boundary_types.at[1, 1].get() == 1,
        lambda x: x.at[y1, :].set(0.0),
        lambda x: x,
        operand=moments,
    )
    ##
    moments_nt = jax.lax.cond(
        boundary_types.at[0, 0].get() == 1,
        lambda x: x.at[x0, :].set(0.0),
        lambda x: x,
        operand=moments_nt,
    )
    moments_nt = jax.lax.cond(
        boundary_types.at[0, 1].get() == 1,
        lambda x: x.at[x1, :].set(0.0),
        lambda x: x,
        operand=moments_nt,
    )
    moments_nt = jax.lax.cond(
        boundary_types.at[1, 0].get() == 1,
        lambda x: x.at[y0, :].set(0.0),
        lambda x: x,
        operand=moments_nt,
    )
    moments_nt = jax.lax.cond(
        boundary_types.at[1, 1].get() == 1,
        lambda x: x.at[y1, :].set(0.0),
        lambda x: x,
        operand=moments_nt,
    )

    return moments, moments_nt



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
    width: int

    @classmethod
    def create(cls: Self, boundary_types: List = None, width: int = 1) -> Self:
        """Register the Dirichlet nodes."""
        # if boundary_types is None:
        boundary_types = jnp.array([[1, 1], [1, 1], [1, 1]])  # fixed

        return cls(boundary_types.astype(jnp.int32), width)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shape_function: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""
        
        moments, moments_nt = apply_boundary_box(nodes.moments, nodes.moments_nt, nodes.ids_grid, self.boundary_types)
        return nodes.replace(moments=moments, moments_nt=moments_nt), self
