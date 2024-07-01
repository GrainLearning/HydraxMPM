"""Module for imposing zero/non-zero non boundaries on a nodes wall."""
# TODO add friction

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass
class NodeWall:
    """Imposing zero/non-zero boundaries on a node wall.

    Example for 2D sticky floor and slip walls:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> nodewall = pm.NodeWall.create(wall_type=0, wall_dim=1, node_ids=jnp.array([0, 1, 2, 3]))

    Attributes:
        wall_type: Type of the wall (0: stick all, 1: stick in inward/outward normal,
            2: slip inward normal, 3: slip outward normal)
        wall_dim: Direction of normal vector (0: x, 1: y, 2: z)
        node_ids: Ids of the nodes on the wall
    """

    wall_type: jnp.int32
    wall_dim: jnp.int32
    node_ids: chex.Array

    @classmethod
    def create(cls: Self, wall_type: jnp.int32, wall_dim: jnp.int32, node_ids: chex.Array) -> Self:
        """Create a node wall."""
        return cls(wall_type=wall_type, wall_dim=wall_dim, node_ids=node_ids)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the boundary conditions on the nodes moments."""

        def stick_all(moments_nt):
            """Stick all directions."""
            moments_nt = moments_nt.at[self.node_ids].set(0.0)
            return moments_nt

        def stick_x(moments_nt):
            """Stick in the direction of inward/outward normal."""
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].set(0.0)
            return moments_nt

        def slip_min(moments_nt):
            """Slip in min direction of inward normal."""
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].min(0.0)
            return moments_nt

        def slip_max(moments_nt):
            """Slip in max direction of outward normal."""
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].max(0.0)
            return moments_nt

        moments_nt = jax.lax.switch(
            self.wall_type,
            (stick_all, stick_x, slip_min, slip_max),
            (nodes.moments_nt),
        )
        return nodes.replace(moments_nt=moments_nt), self
