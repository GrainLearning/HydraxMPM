"""Base class for materials in the simulation."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.particles import Particles


@chex.dataclass
class Material:
    """Base material class.

    Attributes:
        stress_ref: Reference stress tensor.
    """

    stress_ref: chex.Array

    @classmethod
    def create(cls: Self, stress_ref: chex.Array = None) -> Self:
        """Initialize the base material."""
        return cls(stress_ref=stress_ref)

    @jax.jit
    def update_stress(
        self: Self,
        particles: Particles,
        dt: jnp.float32,  # potentially unused
    ) -> Tuple[Particles, Self]:
        """Placeholder for stress update."""
        return particles, self
