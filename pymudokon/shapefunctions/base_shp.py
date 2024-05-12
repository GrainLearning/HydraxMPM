"""Module for containing the baseclass for the shape functions.

Shape functions are calculated for the particle-node interactions.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class BaseShapeFunction(Base):
    """BaseShapeFunctions state for the particle-node interactions.

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
