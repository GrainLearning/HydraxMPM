"""Module for containing the base class for the shape functions.

Shape functions are calculated for the particle-node interactions.
"""

import dataclasses

import jax
from jax import Array

from ..core.base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class ShapeFunction(Base):
    """ShapeFunctions state for the particle-node interactions.

    Attributes:
        shapef (Array):
            Shape function array `(num_particles, stencil_size)`.
        shapef_grad (Array):
            Shape function gradient array `(num_particles, stencil_size, dim)`.
        stencil (Array):
            Stencil containing relative displacements of neighboring nodes for particle-node pair interactions.
    """

    shapef: Array
    shapef_grad: Array
    stencil: Array
