"""Module for the gravity force. Impose gravity on the nodes."""

from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..config.mpm_config import MPMConfig

from ..nodes.nodes import Nodes
from ..particles.particles import Particles

from .forces import Forces

import equinox as eqx
from functools import partial


class Gravity(Forces):
    """Gravity force enforced on the background grid."""

    gravity: chex.Array
    increment: chex.Array
    stop_increment: jnp.int32

    def __init__(
        self: Self,
        config: MPMConfig,
        gravity: chex.Array = None,
        increment: chex.Array = None,
        stop_increment: jnp.int32 = 0,
    ) -> Self:
        """Initialize Gravity force on Nodes."""
        self.gravity = gravity
        self.increment = increment
        self.stop_increment = stop_increment
        super().__init__(config)

    def apply_on_nodes(
        self: Self,
        particles: Particles = None,
        nodes: Nodes =None,
        step: int = 0,
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes."""

        moment_gravity = nodes.mass_stack.reshape(-1, 1) * self.gravity * self.config.dt

        new_moment_nt_stack = nodes.moment_nt_stack + moment_gravity

        new_nodes = eqx.tree_at(
            lambda state: state.moment_nt_stack,
            nodes,
            new_moment_nt_stack,
        )

        # self is updated if there is a gravity ramp
        return new_nodes, self
