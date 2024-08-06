"""Module for the imposing zero/non-zero boundaries at the edges of the domain."""

from typing import List, Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from .nodewall import NodeWall


@chex.dataclass
class DirichletBox:
    """Imposing zero/non-zero boundaries at the edges of the domain.

    The type of boundary is defined by the boundary_types.

    Supported boundary types:
        type 0: stick all directions
        type 1: stick in the direction of inward/outward normal
        type 2: slip in min direction of inward normal
        type 3: slip in max direction of outward normal

    Attributes:
        wall_x0: Wall at the minimum x direction.
        wall_x1: Wall at the maximum x direction.
        wall_y0: Wall at the minimum y direction.
        wall_y1: Wall at the maximum y direction.
        wall_z0: Wall at the minimum z direction.
        wall_z1: Wall at the maximum z direction.

    Example for 2D sticky floor and slip walls:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]),
        end=jnp.array([5.0, 5.0]), node_spacing=1.0)
    >>> dirichlet_box = pm.DirichletBox.create(
    ... nodes, boundary_types=jnp.array([[2, 3], [0, 3]]), width=1)
    """

    wall_x0: NodeWall = None
    wall_x1: NodeWall = None
    wall_y0: NodeWall = None
    wall_y1: NodeWall = None
    wall_z0: NodeWall = None
    wall_z1: NodeWall = None

    @classmethod
    def create(
        cls: Self,
        nodes,
        boundary_types: Tuple[str] = None,
        width: int = 1,
    ) -> Self:
        """Initialize the Dirichlet box to enfore boundary conditions on domain edge.

        Args:
            cls: Self reference_
            nodes: Nodes state
            boundary_types (optional): list of boundary types
                0 - stick in all direction,
                1 - stick in direction of inward/outward normal
                2 - slip in direction of inward normal
                3 - slip in direction of outward normal
            width (optional): wall width. Defaults to 1.

        Returns:
            Self: Initialized Dirichlet box.
        """
        dim = nodes.origin.shape[0]
        if boundary_types is None:
            if dim == 2:
                boundary_types = (("stick", "stick"), ("stick", "stick"))
            elif dim == 3:
                boundary_types = (
                    ("stick", "stick"),
                    ("stick", "stick"),
                    ("stick", "stick"),
                )

        enum_boundary_types = jnp.zeros((dim, 2), dtype=jnp.int32)
        for ai, axis in enumerate(boundary_types):
            for bi, boundary_type in enumerate(axis):
                if boundary_type == "stick":
                    enum_boundary_types = enum_boundary_types.at[ai, bi].set(0)
                elif boundary_type == "slip_positive_normal":
                    enum_boundary_types = enum_boundary_types.at[ai, bi].set(1)
                elif boundary_type == "slip_negative_normal":
                    enum_boundary_types = enum_boundary_types.at[ai, bi].set(2)
                else:
                    raise ValueError("Invalid boundary type")

        node_id_stack = (
            jnp.arange(nodes.num_nodes_total).reshape(nodes.grid_size).astype(jnp.int32)
        )

        if dim == 3:
            x0_ids = jax.lax.slice(
                node_id_stack,
                (0, 0, 0),
                (width, nodes.grid_size[1], nodes.grid_size[2]),
            )
            x1_ids = jax.lax.slice(
                node_id_stack,
                (nodes.grid_size[0] - width, 0, 0),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            y0_ids = jax.lax.slice(
                node_id_stack,
                (0, 0, 0),
                (nodes.grid_size[0], width, nodes.grid_size[2]),
            )
            y1_ids = jax.lax.slice(
                node_id_stack,
                (0, nodes.grid_size[1] - width, 0),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            z0_ids = jax.lax.slice(
                node_id_stack,
                (0, 0, 0),
                (nodes.grid_size[0], nodes.grid_size[1], width),
            )
            z1_ids = jax.lax.slice(
                node_id_stack,
                (0, 0, nodes.grid_size[2] - width),
                (nodes.grid_size[0], nodes.grid_size[1], nodes.grid_size[2]),
            )
            wall_x0 = NodeWall.create(
                wall_type=enum_boundary_types[0, 0], wall_dim=0, node_id_stack=x0_ids
            )
            wall_x1 = NodeWall.create(
                wall_type=enum_boundary_types[0, 1], wall_dim=0, node_id_stack=x1_ids
            )
            wall_y0 = NodeWall.create(
                wall_type=enum_boundary_types[1, 0], wall_dim=1, node_id_stack=y0_ids
            )
            wall_y1 = NodeWall.create(
                wall_type=enum_boundary_types[1, 1], wall_dim=1, node_id_stack=y1_ids
            )
            wall_z0 = NodeWall.create(
                wall_type=enum_boundary_types[2, 0], wall_dim=2, node_id_stack=z0_ids
            )
            wall_z1 = NodeWall.create(
                wall_type=enum_boundary_types[2, 1], wall_dim=2, node_id_stack=z1_ids
            )

        elif dim == 2:
            x0_ids = node_id_stack.at[0:width, :].get().reshape(-1)
            x1_ids = node_id_stack.at[nodes.grid_size[0] - width :, :].get().reshape(-1)
            y0_ids = node_id_stack.at[:, 0:width].get().reshape(-1)
            y1_ids = node_id_stack.at[:, nodes.grid_size[1] - width :].get().reshape(-1)

            wall_x0 = NodeWall.create(
                wall_type=enum_boundary_types[0, 0], wall_dim=0, node_id_stack=x0_ids
            )
            wall_x1 = NodeWall.create(
                wall_type=enum_boundary_types[0, 1], wall_dim=0, node_id_stack=x1_ids
            )
            wall_y0 = NodeWall.create(
                wall_type=enum_boundary_types[1, 0], wall_dim=1, node_id_stack=y0_ids
            )
            wall_y1 = NodeWall.create(
                wall_type=enum_boundary_types[1, 1], wall_dim=1, node_id_stack=y1_ids
            )

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

    def apply_on_nodes_moments(
        self: Self,
        nodes: Nodes,
        particles: Particles = None,
        shapefunctions: ShapeFunction = None,
        dt: jnp.float32 = 0.0,
    ) -> Tuple[Nodes, Self]:
        """Apply the boundary conditions on the nodes moments."""
        nodes, _ = self.wall_x0.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_x1.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_y0.apply_on_nodes_moments(nodes)
        nodes, _ = self.wall_y1.apply_on_nodes_moments(nodes)
        if self.wall_z0 is not None:
            nodes, _ = self.wall_z0.apply_on_nodes_moments(nodes)
        if self.wall_z1 is not None:
            nodes, _ = self.wall_z1.apply_on_nodes_moments(nodes)

        return nodes, self
