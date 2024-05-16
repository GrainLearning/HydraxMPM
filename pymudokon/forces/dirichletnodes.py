import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.base import Base
from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class DirichletNodes(Base):
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

    type_mask: jnp.array
    values: jnp.array

    @classmethod
    def register(cls: Self, type_mask: Array, values: Array) -> Nodes:
        """Register the Dirichlet nodes."""
        return cls(values=values, mask=type_mask.astype[jnp.int16])

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        interactions: Interactions,
        dt: jnp.float32,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""
        # TODO
        return nodes, self
