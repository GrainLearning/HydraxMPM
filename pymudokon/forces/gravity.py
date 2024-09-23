"""Module for the gravity force. Impose gravity on the nodes."""

from typing import Tuple
from typing_extensions import Self

import chex
import jax.numpy as jnp
import jax

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction


@chex.dataclass
class Gravity:
    """Gravity force enforced on the background grid.

    Attributes:
        gravity (Array): Gravity vector `(dim,)`.

    Example usage:
        >>> import jax.numpy as jnp
        >>> import pymudokon as pm
        >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]), node_spacing=0.5)
        >>> grav = pm.Gravity.create(jnp.array([0.0, 9.8]))
        >>> # add gravity to solver

    """

    gravity: chex.Array
    increment: chex.Array
    stop_increment: jnp.int32


    @classmethod
    def create(cls: Self, gravity: chex.Array=None, increment : chex.Array = None,stop_increment : jnp.int32 = 0) -> Self:
        """Initialize Gravity force on Nodes."""
        return cls(gravity=gravity, increment= increment, stop_increment=stop_increment)

    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
        step: jnp.int32 = 0
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes."""
        
        if self.increment is not None:
            # jax.debug.print("grav step {} {} {}",step, self.stop_increment, is_done)
            self = jax.lax.cond(
                self.stop_increment >= step,
                lambda:  self.replace(gravity = self.gravity + self.increment),
                lambda: self
                )

        
        moment_stack, moment_nt_stack = self.apply_gravity(
            nodes.moment_stack, nodes.moment_nt_stack, nodes.mass_stack, dt
        )
        return nodes.replace(
            moment_nt_stack=moment_nt_stack,
        ), self

    def apply_gravity(
        self,
        moment: chex.Array,
        moment_nt: chex.Array,
        masses: chex.Array,
        dt: jnp.float32,
    ) -> Tuple[chex.Array, chex.Array]:
        """Apply gravity on the nodes moments."""
        moment_gravity = masses.reshape(-1, 1) * self.gravity * dt

        moment_nt = moment_nt + moment_gravity

        return moment, moment_nt
