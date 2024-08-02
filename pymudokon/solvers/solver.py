"""Base class for MPM solver"""

import chex
import jax.numpy as jnp


@chex.dataclass
class Solver:
    """MPM solver base class

    Attributes:
        dt: Time step.
    """

    dt: jnp.float32
