"""Module for containing base class for the material."""

from typing import Tuple
from flax import struct
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction

@struct.dataclass
class Forces:
    """Force state for the material properties."""

    @jax.jit
    def apply_on_particles(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        dt: jnp.float32,
    ) -> Tuple[Particles, Self]:
        """Apply the force on the particles."""
        return particles, self

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        dt: jnp.float32,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""
        return nodes, self
