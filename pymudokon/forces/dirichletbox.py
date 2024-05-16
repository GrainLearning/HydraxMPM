import dataclasses
from typing import Tuple, Dict, List

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.base import Base
from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction
from functools import partial


@partial(jax.jit, static_argnames=["boundary_types"])
def apply_boundary_box(moments: Array, ids_grid: Array, boundary_types: List):
    # # TODO: Add support for 3D

    x0 = ids_grid.at[0, :].get().reshape(-1)
    x1 = ids_grid.at[-1, :].get().reshape(-1)

    # # # find ids at y0
    y0 = ids_grid.at[:, 0].get().reshape(-1)
    y1 = ids_grid.at[:, -1].get().reshape(-1)

    if boundary_types.at[0, 0] == 1:
        moments = moments.at[x0].set(0)

    if boundary_types.at[0, 1] == 1:
        moments = moments.at[x1].set(0)

    if boundary_types.at[1, 0] == 1:
        moments = moments.at[y0].set(0)

    if boundary_types.at[1, 1] == 1:
        moments = moments.at[y1].set(0)

    return moments

    # return nodes.replace(moments=moments), self


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class DirichletBox(Base):
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
    def register(cls: Self, boundary_types: Dict = None, width: int = 1) -> Self:
        """Register the Dirichlet nodes."""
        if boundary_types is None:
            boundary_types = jnp.array([[0, 0], [0, 0], [0, 0]]).astype(jnp.int32)  # fixed

        return cls(boundary_types, width)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        interactions: Interactions = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""

        moments = apply_boundary_box(nodes.moments, nodes.ids_grid, self.boundary_types)

        return nodes.replace(moments=moments), self
