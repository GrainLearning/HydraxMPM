"""Module for containing the linear shape functions."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from .shapefunction import ShapeFunction


@struct.dataclass
class LinearShapeFunction(ShapeFunction):
    """Linear shape functions for the particle-node interactions."""

    @classmethod
    def create(cls: Self, num_particles: jnp.int32, dim: jnp.int16) -> Self:
        """Initializes the state of the linear shape functions.

        Args:
            cls (Self): Self type reference
            num_particles (jnp.int32): Number of particles
            dim (jnp.int16): Dimension of the problem

        Returns: ShapeFunction: Initialized shape function and interactions state.
        """
        # Generate the stencil based on the dimension
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
            intr_shapef_grad=jnp.zeros((num_particles * stencil_size, dim), dtype=jnp.float32),
            stencil=stencil,
        )

    @jax.jit
    def calculate_shapefunction(self: Self, nodes: Nodes, positions: Array) -> Self:
        """Top level function to calculate the shape functions. Assumes `get_interactions` has been called.

        Args:
            self (LinearShapeFunction):
                Shape function at previous state
            nodes (Nodes):
                Nodes state containing grid size and inv_node_spacing
            particles (Particles):
                Particles state containing particle positions

        Returns:>
            LinearShapeFunction: Updated shape function state for the particle and node pairs.
        """
        # Solution procedure:
        _, dim = self.stencil.shape

        # 1. Calculate the particle-node pair interactions
        # see `ShapeFunction class` for more details
        intr_dist, intr_hashes = self.vmap_intr(
            self.intr_ids, positions, nodes.origin, nodes.inv_node_spacing, nodes.grid_size, dim
        )

        # 3. Calculate the shape functions
        intr_shapef, intr_shapef_grad = self.vmap_intr_shp(intr_dist, nodes.inv_node_spacing)

        # return nodal distancesm if needed
        return self.replace(
            intr_shapef=intr_shapef, intr_shapef_grad=intr_shapef_grad, intr_hashes=intr_hashes
        ), intr_dist

    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=(0))
    def vmap_intr_shp(
        self: Self,
        intr_dist: Array,
        inv_node_spacing: jnp.float32,
    ) -> Tuple[Array, Array]:
        """Vectorized linear shape function calculation.

        Calculate the shape function, and then its gradient

        Args:
            intr_dist (Array):
                Particle-node pair interactions distance.
            inv_node_spacing (jnp.float32):
                Inverse node spacing.

        Returns:
            Tuple[Array, Array]:
                Shape function and its gradient.
        """
        abs_intr_dist = jnp.abs(intr_dist)
        basis = jnp.where(abs_intr_dist < 1.0, 1.0 - abs_intr_dist, 0.0)
        dbasis = jnp.where(abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_node_spacing, 0.0)

        intr_shapef = jnp.prod(basis)
        # intr_shapef_grad = dbasis * jnp.roll(basis, shift=-1)
        dim = basis.shape[0]
        if dim == 2:
            intr_shapef_grad = jnp.array(
                [
                    dbasis[0] * basis[1],
                    dbasis[1] * basis[0],
                ]
            )
        elif dim == 3:
            intr_shapef_grad = jnp.array(
                [dbasis[0] * basis[1] * basis[2], dbasis[1] * basis[0] * basis[2], dbasis[2] * basis[0] * basis[1]]
            )
        else:
            intr_shapef_grad = dbasis

        return intr_shapef, intr_shapef_grad
