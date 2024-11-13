"""Module for the gravity force. Impose gravity on the nodes."""

from typing import Tuple

import chex
import equinox as eqx
import jax.numpy as jnp
from typing_extensions import Self

from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from .forces import Forces


class Gravity(Forces):
    """Gravity force enforced on the background grid."""

    gravity: chex.Array
    increment: chex.Array
    stop_ramp_step: jnp.int32

    def __init__(
        self: Self,
        config: MPMConfig,
        gravity: chex.Array = None,
        increment: chex.Array = None,
        stop_ramp_step: jnp.int32 = 0,
    ) -> Self:
        """Initialize Gravity force on Nodes."""
        self.gravity = gravity
        self.increment = increment
        self.stop_ramp_step = stop_ramp_step
        super().__init__(config)

    def apply_on_nodes(
        self: Self,
        particles: Particles = None,
        nodes: Nodes = None,
        step: int = 0,
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes."""

        if self.increment is not None:
            gravity = self.gravity + self.increment * jnp.minimum(
                step, self.stop_ramp_step
            )
        else:
            gravity = self.gravity

        moment_gravity = nodes.mass_stack.reshape(-1, 1) * gravity * self.config.dt

        new_moment_nt_stack = nodes.moment_nt_stack + moment_gravity

        new_nodes = eqx.tree_at(
            lambda state: state.moment_nt_stack,
            nodes,
            new_moment_nt_stack,
        )

        # self is updated if there is a gravity ramp
        return new_nodes, self
