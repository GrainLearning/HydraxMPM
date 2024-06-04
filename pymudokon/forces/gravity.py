"""Gravity force on Nodes."""

import dataclasses
from functools import partial
from typing import Tuple

from flax import struct
import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles





@struct.dataclass
class Gravity:
    """Dataclass containing the state of the Gravity forces.

    Attributes:
        gravity (Array): Gravity vector `(dim,)`.
    """

    gravity: Array

    @classmethod
    def create(cls: Self, gravity: Array) -> Self:
        """Initialize Gravity force on Nodes.

        Args:
            cls (Gravity): Self reference.
            gravity (Array): Gravity vector `(dim,)`.

        Returns:
            Gravity: Initialized Gravity force.
        """
        dim = gravity.shape[0]
        gravity.reshape(-1, dim)
        return cls(gravity=gravity)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: Interactions = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply gravity on the nodes.

        Args:
            self (Self): Self reference.
            nodes (Nodes): Nodes in the simulation.
            particles (Particles, optional): MPM particles. Defaults to None.
            shapefunctions (Interactions, optional): Shapefunctions. Defaults to None.
            dt (jnp.float32, optional): Time step. Defaults to 0.0.

        Returns:
            Tuple[Nodes, Self]: Updated nodes and gravity force.

        Example:
            >>> import jax.numpy as jnp
            >>> import pymudokon as pm
            >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)
            >>> grav = pm.Gravity.create(jnp.array([0.0, 9.8]))
            >>> nodes, grav = grav.apply_on_nodes_moments(nodes=nodes, dt=0.01)
        """
        moments, moments_nt = self.apply_gravity(nodes.moments, nodes.moments_nt, nodes.masses, self.gravity, dt)
        return nodes.replace(
            moments_nt=moments_nt,
        ), self

    @partial(jax.jit, static_argnames=["self","gravity"])
    def apply_gravity(
        self,moments: Array, moments_nt: Array, masses: Array, gravity: Array, dt: jnp.float32
    ) -> Tuple[Array, Array]:
        """Apply gravity on the nodes.

        Args:
            moments (Array): Nodal moments `(num_nodes, dim)`.
            moments_nt (Array): Nodal moments in the forward step `(num_nodes, dim)`.
            masses (Array): Nodal masses `(num_nodes,)`.
            gravity (Array): Gravity vector `(dim,)`.
            dt (jnp.float32): Time step.

        Returns:
            Tuple[Array, Array]: Updated moments and moments_nt.
        """
        moment_gravity = masses.reshape(-1, 1) * gravity * dt

        moments_nt = moments_nt + moment_gravity

        return moments, moments_nt