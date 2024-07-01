"""Module containing base forces state."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass
class Forces:
    """Force state for the material properties."""

    @classmethod
    def create(cls: Self) -> Self:
        """Initialize the force state."""
        return cls()

    @jax.jit
    def apply_on_particles(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        dt: jnp.float32,
    ) -> Tuple[Particles, Self]:
        """Placeholder. Apply the force on the particles."""
        return particles, self

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        dt: jnp.float32,
    ) -> Tuple[Nodes, Self]:
        """Placeholder. Apply the force on the nodes."""
        return nodes, self
