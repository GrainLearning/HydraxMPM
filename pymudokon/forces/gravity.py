"""Module for the gravity force. Impose gravity on the nodes."""

from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from pymudokon.config.mpm_config import MPMConfig

from ..partition.grid_stencil_map import GridStencilMap

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction

import equinox as eqx
from functools import partial


class Gravity(eqx.Module):
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

    dt: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig,
        gravity: chex.Array = None,
        increment: chex.Array = None,
        stop_increment: jnp.int32 = 0,
        dt: float = 0.0,
    ) -> Self:
        """Initialize Gravity force on Nodes."""
        if config:
            dt = config.dt

        self.dt = dt
        self.gravity = gravity
        self.increment = increment
        self.stop_increment = stop_increment

    def __call__(
        self: Self,
        nodes: Nodes,
        grid: GridStencilMap = None,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        step: int = 0,
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes."""

        moment_gravity = nodes.mass_stack.reshape(-1, 1) * self.gravity * self.dt

        new_moment_nt_stack = nodes.moment_nt_stack + moment_gravity

        new_nodes = eqx.tree_at(
            lambda state: state.moment_nt_stack,
            nodes,
            new_moment_nt_stack,
        )

        # self is updated if there is a gravity ramp
        return new_nodes, self
