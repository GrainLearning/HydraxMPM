"""Base class for materials in the simulation."""

import chex
import jax.numpy as jnp


@chex.dataclass
class Material:
    """Base material class.

    Attributes:
        absolute_density: Absolute density e.g., particle density.
    """

    absolute_density: jnp.float32
