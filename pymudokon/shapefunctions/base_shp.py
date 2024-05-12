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
        shapef Array:
            Shape function array `(num_particles, stencil_size)`.
        shapef_grad Array:
            Shape function gradient array `(num_particles, stencil_size, dim)`.
    """

    shapef: Array
    shapef_grad: Array

    @classmethod
    def register(cls: Self, num_particles: jnp.int32, stencil_size: jnp.int32, dim: jnp.int16) -> Self:
        """Initializes the shape function container.

        Args:
            cls (Self):
                self type reference
            num_particles (jnp.int32):
                Number of particles
            stencil_size (jnp.int32):
                Size of the stencil
            dim (jnp.int16):
                Dimension of the problem

        Returns:
            BaseShapeFunction:
                Container for shape functions and gradients
        """
        return cls(
            shapef=jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
            shapef_grad=jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
        )
