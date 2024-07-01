"""Module for the gravity force. Impose gravity on the nodes."""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass(mappable_dataclass=False, frozen=True)
class Gravity:
    """Gravity force enforced on the background grid.

    Example usage:
        >>> import jax.numpy as jnp
        >>> import pymudokon as pm
        >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)
        >>> grav = pm.Gravity.create(jnp.array([0.0, 9.8]))
        >>> # add gravity to solver

    Attributes:
        gravity (Array): Gravity vector `(dim,)`.
    """

    gravity: chex.Array

    @classmethod
    def create(cls: Self, gravity: chex.Array) -> Self:
        """Initialize Gravity force on Nodes."""
        dim = gravity.shape[0]
        gravity.reshape(-1, dim)
        return cls(gravity=gravity)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes."""
        moments, moments_nt = self.apply_gravity(nodes.moments, nodes.moments_nt, nodes.masses, dt)
        return nodes.replace(
            moments_nt=moments_nt,
        ), self

    @partial(jax.jit, static_argnames=["self"])
    def apply_gravity(
        self, moments: chex.Array, moments_nt: chex.Array, masses: chex.Array, dt: jnp.float32
    ) -> Tuple[chex.Array, chex.Array]:
        """Apply gravity on the nodes moments."""
        moment_gravity = masses.reshape(-1, 1) * self.gravity * dt

        moments_nt = moments_nt + moment_gravity

        return moments, moments_nt
