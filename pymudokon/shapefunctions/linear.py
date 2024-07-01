"""Most basic linear shape functions."""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.nodes import Nodes
from .shapefunction import ShapeFunction


@chex.dataclass(mappable_dataclass=False, frozen=True)
class LinearShapeFunction(ShapeFunction):
    """Most basic and fast shape functions, yet unstable for traditional solvers.

    Requires at least 2 particles per cell (in 2D) for stability.

    C0 continuous and suffers cell crossing instability.

    The shapefunction forms part of the solver class. However, it can be used as a standalone module.

    Example:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> positions = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    >>> nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([5.0, 5.0]), node_spacing=1.0
    >>> linear_shp = pm.LinearShapeFunction.create(num_particles=len(positions), dim=2)
    >>> linear_shp, intr_dist = linear_shp.calculate_shapefunction(nodes, positions)
    """

    @classmethod
    def create(cls: Self, num_particles: jnp.int32, dim: jnp.int16) -> Self:
        """Initializes the state of the linear shape functions.

        Args:
            cls: Self type reference
            num_particles: Number of particles
            dim: Dimension of the problem

        Returns:
            ShapeFunction: linear shape function state
        """
        # Generate stencil by dimension.
        if dim == 1:
            stencil = jnp.array([[0.0], [1.0]])
        if dim == 2:
            stencil = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        if dim == 3:
            stencil = jnp.array(
                [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
            )
        stencil_size = stencil.shape[0]

        intr_ids = jnp.arange(num_particles * stencil_size).astype(jnp.int32)

        return cls(
            intr_ids=intr_ids,
            intr_hashes=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
            intr_shapef=jnp.zeros((num_particles * stencil_size), dtype=jnp.float32),
            intr_shapef_grad=jnp.zeros((num_particles * stencil_size, 3), dtype=jnp.float32),  # 3 is for plane strain
            stencil=stencil,
        )

    @jax.jit
    def calculate_shapefunction(self: Self, nodes: Nodes, positions: chex.Array) -> Tuple[Self, chex.Array]:
        """Calculate shape functions and its gradients.

        Args:
            self: Shape function at previous state
            nodes: Nodes state containing grid size and inv_node_spacing
            positions: coordinates on the grid

        Returns:
            Tuple:
                - Updated shape function state
                - Interaction distances
        """
        _, dim = self.stencil.shape

        # Get relative interaction distances and hashes, see `ShapeFunction class` for more details
        intr_dist, intr_hashes = self.vmap_intr(
            self.intr_ids, positions, nodes.origin, nodes.inv_node_spacing, nodes.grid_size
        )

        # Get shape functions and gradients. Batched over intr_dist.
        intr_shapef, intr_shapef_grad = self.vmap_intr_shp(intr_dist, nodes.inv_node_spacing)

        # Return updated state, and distances.
        return self.replace(
            intr_shapef=intr_shapef, intr_shapef_grad=intr_shapef_grad, intr_hashes=intr_hashes
        ), intr_dist

    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=(0))
    def vmap_intr_shp(
        self: Self,
        intr_dist: chex.ArrayBatched,
        inv_node_spacing: jnp.float32,
    ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched]:
        """Vectorized map to compute shapefunctions and gradients.

        Args:
            intr_dist: Batched particle-node pair interactions distance.
            inv_node_spacing: Inverse node spacing.

        Returns:
            Tuple: Shape function and its gradient.
        """
        abs_intr_dist = jnp.abs(intr_dist)
        basis = jnp.where(abs_intr_dist < 1.0, 1.0 - abs_intr_dist, 0.0)
        dbasis = jnp.where(abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_node_spacing, 0.0)

        intr_shapef = jnp.prod(basis)

        dim = basis.shape[0]
        if dim == 2:
            intr_shapef_grad = jnp.array(
                [
                    dbasis[0] * basis[1],
                    dbasis[1] * basis[0],
                    0.0,
                ]
            )
        elif dim == 3:
            intr_shapef_grad = jnp.array(
                [dbasis[0] * basis[1] * basis[2], dbasis[1] * basis[0] * basis[2], dbasis[2] * basis[0] * basis[1]]
            )
        else:
            intr_shapef_grad = jnp.array([dbasis, 0.0, 0.0])

        return intr_shapef, intr_shapef_grad
