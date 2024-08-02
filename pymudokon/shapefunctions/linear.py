"""Most basic linear shape functions."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from .shapefunctions import ShapeFunction


@chex.dataclass(mappable_dataclass=False, frozen=True)
class LinearShapeFunction(ShapeFunction):
    """Most basic and fast shape functions, yet unstable for traditional solvers.

    Requires at least 2 particles per cell (in 2D) for stability.

    C0 continuous and suffers cell crossing instability.

    The shapefunction forms part of the solver class. However, it can be used as a
    standalone module.

    Example:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> positions = jnp.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    >>> nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]), end=jnp.array([5.0, 5.0]), node_spacing=1.0)
    >>> linear_shp = pm.LinearShapeFunction.create(num_particles=len(positions), dim=2)
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
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            )

        stencil_size = stencil.shape[0]

        intr_id_stack = jnp.arange(num_particles * stencil_size).astype(jnp.int32)

        return cls(
            intr_id_stack=intr_id_stack,
            intr_hash_stack=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
            intr_shapef_stack=jnp.zeros(
                (num_particles * stencil_size), dtype=jnp.float32
            ),
            intr_shapef_grad_stack=jnp.zeros(
                (num_particles * stencil_size, 3), dtype=jnp.float32
            ),
            stencil=stencil,
        )

    def calculate_shapefunction(
        self: Self,
        origin: chex.Array,
        inv_node_spacing: jnp.float32,
        grid_size: chex.Array,
        position_stack: chex.Array,
    ) -> Tuple[Self, chex.Array]:
        """Calculate shape functions and its gradients.

        Args:
            self: Shape function at previous state
            origin: start coordinates of the grid
            inv_node_spacing: 1/node_spacing (inverse node spacing/ grid spacing)
            grid_size: Number of nodes in each axis
            position_stack: All coordinates on the grid

        Returns:
            Tuple:
                - Updated shape function state
                - Interaction distances
        """
        stencil_size, dim = self.stencil.shape

        num_particles = position_stack.shape[0]

        intr_id_stack = jnp.arange(num_particles * stencil_size).astype(jnp.int32)

        # Get relative interaction distances and hashes, see `ShapeFunction class`
        intr_dist_stack, intr_hash_stack = self.vmap_intr(
            intr_id_stack, position_stack, origin, inv_node_spacing, grid_size
        )

        # Get shape functions and gradients. Batched over intr_dist.
        intr_shapef_stack, intr_shapef_grad_stack = self.vmap_intr_shp(
            intr_dist_stack, inv_node_spacing
        )

        intr_dist_3d_stack = jnp.pad(
            intr_dist_stack,
            [(0, 0), (0, 3 - dim)],
            mode="constant",
            constant_values=0,
        )

        return self.replace(
            intr_shapef_stack=intr_shapef_stack,
            intr_shapef_grad_stack=intr_shapef_grad_stack,
            intr_id_stack=intr_id_stack,
            intr_hash_stack=intr_hash_stack,
        ), intr_dist_3d_stack

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
        dbasis = jnp.where(
            abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_node_spacing, 0.0
        )

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
                [
                    dbasis[0] * basis[1] * basis[2],
                    dbasis[1] * basis[0] * basis[2],
                    dbasis[2] * basis[0] * basis[1],
                ]
            )
        else:
            intr_shapef_grad = jnp.array([dbasis, 0.0, 0.0])

        return intr_shapef, intr_shapef_grad
