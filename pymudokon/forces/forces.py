"""Module containing base forces state."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

# from ..core.nodes import Nodes
# from ..core.particles import Particles
# from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass
class Forces:
    """Force state for the material properties."""

    @classmethod
    def create(cls: Self) -> Self:
        """Initialize the force state."""
        return cls()
