"""Module for imposing zero/non-zero non boundaries on a nodes wall."""
# TODO add friction

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction


@chex.dataclass
class NodeWall:
    """Imposing zero/non-zero boundaries on a node wall.

    Attributes:
        wall_type: Type of the wall (0: stick all, 1: stick in inward/outward normal,
            2: slip inward normal, 3: slip outward normal)
        wall_dim: Direction of normal vector (0: x, 1: y, 2: z)
        node_id_stack: Ids of the nodes on the wall

    Example for 2D sticky floor and slip walls:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> nodewall = pm.NodeWall.create(wall_type=0, wall_dim=1,
        node_id_stack=jnp.array([0, 1, 2, 3]))

    """

    wall_type: jnp.int32
    wall_dim: jnp.int32
    node_id_stack: chex.Array

    @classmethod
    def create(
        cls: Self, wall_type: jnp.int32, wall_dim: jnp.int32, node_id_stack: chex.Array
    ) -> Self:
        """Create a node wall."""
        return cls(wall_type=wall_type, wall_dim=wall_dim, node_id_stack=node_id_stack)

    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
        step: jnp.int32 = 0,
    ) -> Tuple[Nodes, Self]:
        """Apply the boundary conditions on the nodes moments."""

        def stick_all(moment_nt):
            """Stick all directions."""
            moment_nt = moment_nt.at[self.node_id_stack].set(0.0)
            return moment_nt

        def slip_positive_normal(moment_nt):
            """Slip in min direction of inward normal."""
            moment_nt = moment_nt.at[self.node_id_stack, self.wall_dim].min(0.0)
            return moment_nt

        def slip_negative_normal(moment_nt):
            """Slip in max direction of outward normal."""
            moment_nt = moment_nt.at[self.node_id_stack, self.wall_dim].max(0.0)
            return moment_nt

        moment_nt_stack = jax.lax.switch(
            self.wall_type,
            (stick_all, slip_positive_normal, slip_negative_normal),
            (nodes.moment_nt_stack),
        )
        return nodes.replace(moment_nt_stack=moment_nt_stack), self
