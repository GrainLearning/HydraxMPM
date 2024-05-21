import dataclasses
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.base import Base
from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@partial(jax.jit, static_argnames=["gravity"])
def apply_gravity(moments: Array, moments_nt: Array, masses: Array, gravity: Array, dt: jnp.float32):
    # # TODO: Add support for 3D

    moment_gravity = masses.reshape(-1, 1) * gravity * dt

    moments = moments + moment_gravity
    moments_nt = moments_nt + moment_gravity

    return moments, moments_nt


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Gravity(Base):
    """Dirichlet boundary conditions with user defined values.

    Attributes:
        type mask (Array):
            type mask in (X,Y,Z) space on where to apply the Dirichlet boundary conditions.
            - 0 is not applied
            - 1 is fixed
            - 2 max
            - 3 min
            Shape is `(num_nodes, dim)`.
        values (Array):
            values of shape `(num_nodes, dim)` to apply on the nodes.
    """

    gravity: Array

    @classmethod
    def register(cls: Self, gravity: Array) -> Self:
        """Initialize Gravity force on Nodes.

        Args:
            cls (Self): _description_
            gravity (Array): _description_

        Returns:
            Self: _description_
        """
        dim = gravity.shape[0]
        gravity.reshape(-1, dim)
        return cls(gravity=gravity)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        interactions: Interactions = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""

        moments, moments_nt = apply_gravity(nodes.moments, nodes.moments_nt, nodes.masses, self.gravity, dt)
        return nodes.replace(
            moments=moments,
            moments_nt=moments_nt,
        ), self
