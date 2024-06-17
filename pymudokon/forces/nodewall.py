"""Node walls"""

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..shapefunctions.shapefunction import ShapeFunction


@struct.dataclass
class NodeWall:
    """Walls.

    Type
        type 0: stick all directions
        type 1: stick in the direction of dim
        type 2: slip in min direction of dim
        type 3: slip in max direction of dim

    dim  0:x, 1:y, 2:z

    """

    wall_type: jnp.int32
    wall_dim: jnp.int32
    node_ids: jnp.array

    @classmethod
    def create(cls: Self, wall_type: jnp.int32, wall_dim: jnp.int32, node_ids: jnp.array) -> Self:
        return cls(wall_type=wall_type, wall_dim=wall_dim, node_ids=node_ids)

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        def stick_all(moments_nt):
            moments_nt = moments_nt.at[self.node_ids].set(0.0)
            return moments_nt

        def stick_x(moments_nt):
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].set(0.0)
            return moments_nt

        def slip_min(moments_nt):
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].min(0.0)
            return moments_nt

        def slip_max(moments_nt):
            moments_nt = moments_nt.at[self.node_ids, self.wall_dim].max(0.0)
            return moments_nt

        moments_nt = jax.lax.switch(
            self.wall_type,
            (stick_all, stick_x, slip_min, slip_max),
            (nodes.moments_nt),
        )
        return nodes.replace(moments_nt=moments_nt), self
