"""Module for containing base class for the material."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.base import Base
from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Forces(Base):
    """Force state for the material properties."""

    @jax.jit
    def apply_on_particles(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: ShapeFunction,
        interactions: Interactions,
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
        interactions: Interactions,
        dt: jnp.float32,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""
        return nodes, self
