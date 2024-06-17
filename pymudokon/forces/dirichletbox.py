"""Gravity force on Nodes."""

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

from .nodewall import NodeWall


@struct.dataclass
class DirichletBox:
    """Dirichlet boundary conditions with user defined values.

    type 0: stick all directions
    type 1: stick in the direction of dim
    type 2: slip in min direction of dim
    type 3: slip in max direction of dim
    """

    wall_x0: NodeWall = None
    wall_x1: NodeWall = None
    wall_y0: NodeWall = None
    wall_y1: NodeWall = None
    wall_z0: NodeWall = None
    wall_z1: NodeWall = None

    @classmethod
    def create(cls: Self, nodes, boundary_types: List = None, width: int = 1) -> Self:
        """Register the Dirichlet nodes."""

        dim = nodes.origin.shape[0]

        if boundary_types is None:
            boundary_types = jnp.zeros(dim).repeat(2).reshape(dim, 2).astype(jnp.int32)

        node_ids = jnp.arange(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jnp.int32).T

        if dim == 3:
            x0_ids = jax.lax.slice(node_ids, (0, 0, 0), (width, nodes.grid_size[1], nodes.grid_size[2]))
            x1_ids = jax.lax.slice(
                node_ids,
                (nodes.grid_size[0] - width, 0, 0),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            y0_ids = jax.lax.slice(node_ids, (0, 0, 0), (nodes.grid_size[0], width, nodes.grid_size[2]))
            y1_ids = jax.lax.slice(
                node_ids,
                (0, nodes.grid_size[1] - width, 0),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            z0_ids = jax.lax.slice(node_ids, (0, 0, 0), (nodes.grid_size[0], nodes.grid_size[1], width))
            z1_ids = jax.lax.slice(
                node_ids,
                (0, 0, nodes.grid_size[2] - width),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            wall_x0 = NodeWall.create(wall_type=boundary_types[0, 0], wall_dim=0, node_ids=x0_ids)
            wall_x1 = NodeWall.create(wall_type=boundary_types[0, 1], wall_dim=0, node_ids=x1_ids)
            wall_y0 = NodeWall.create(wall_type=boundary_types[1, 0], wall_dim=1, node_ids=y0_ids)
            wall_y1 = NodeWall.create(wall_type=boundary_types[1, 1], wall_dim=1, node_ids=y1_ids)
            wall_z0 = NodeWall.create(wall_type=boundary_types[2, 0], wall_dim=2, node_ids=z0_ids)
            wall_z1 = NodeWall.create(wall_type=boundary_types[2, 1], wall_dim=2, node_ids=z1_ids)

        elif dim == 2:
            x0_ids = node_ids.at[0:width, :].get().reshape(-1)
            x1_ids = node_ids.at[nodes.grid_size[0] - width :, :].get().reshape(-1)
            y0_ids = node_ids.at[:, 0:width].get().reshape(-1)
            y1_ids = node_ids.at[:, nodes.grid_size[1] - width :].get().reshape(-1)

            print(x0_ids.shape, x1_ids.shape, y0_ids.shape, y1_ids.shape)
            print(x0_ids[0], x1_ids[0])
            wall_x0 = NodeWall.create(wall_type=boundary_types[0, 0], wall_dim=0, node_ids=x0_ids)
            wall_x1 = NodeWall.create(wall_type=boundary_types[0, 1], wall_dim=0, node_ids=x1_ids)
            wall_y0 = NodeWall.create(wall_type=boundary_types[1, 0], wall_dim=1, node_ids=y0_ids)
            wall_y1 = NodeWall.create(wall_type=boundary_types[1, 1], wall_dim=1, node_ids=y1_ids)
            wall_z0 = None
            wall_z1 = None

        else:
            raise ValueError("Only 2D and 3D are supported")

        return cls(
            wall_x0=wall_x0,
            wall_x1=wall_x1,
            wall_y0=wall_y0,
            wall_y1=wall_y1,
            wall_z0=wall_z0,
            wall_z1=wall_z1,
        )

    @jax.jit
    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the force on the nodes."""

        nodes, _ = self.wall_x0.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_x1.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_y0.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_y1.apply_on_nodes_moments(nodes)
        if self.wall_z0 is not None:
            nodes, _ = self.wall_z0.apply_on_nodes_moments(nodes)
        if self.wall_z1 is not None:
            nodes, _ = self.wall_z1.apply_on_nodes_moments(nodes)

        return nodes, self
