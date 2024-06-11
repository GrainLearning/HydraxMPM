"""Module for containing base class for the material."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Self

from ..core.particles import Particles


@struct.dataclass
class Material:
    """Bass state for the materials."""

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # potentially unused
    ) -> Tuple[Particles, Self]:
        """Base update method for materials."""
        return particles, self
