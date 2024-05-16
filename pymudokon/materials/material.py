"""Module for containing base class for the material."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.base import Base
from ..core.particles import Particles


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Material(Base):
    """Bass state for the materials."""

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # potentially unused
    ) -> Tuple[Particles, Self]:
        """Base update method for materials."""
        return particles, self
