"""Module for containing the linear shape functions."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.interactions import Interactions
from ..core.nodes import Nodes
from .base_shp import BaseShapeFunction


def vmap_linear_shapefunction(
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
    
    N0 = basis[0, :]
    N1 = basis[1, :]
    dN0 = dbasis[0, :]
    dN1 = dbasis[1, :]

    shapef = jnp.expand_dims(N0 * N1, axis=1)

    shapef_grad = jnp.array([dN0 * N1, N0 * dN1])

    # shapes of returned array are
    # (num_particles*stencil_size, 1, 1)
    # and (num_particles*stencil_size, dim,1)
    # respectively
    return shapef, shapef_grad,basis


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class LinearShapeFunction(BaseShapeFunction):
    """Linear shape functions for the particle-node interactions."""

    @jax.jit
    def calculate_shapefunction(
        self: Self,
        nodes: Nodes,
        interactions: Interactions,
    ) -> Self:
        """Top level function to calculate the shape functions.

        Args:
            self (LinearShapeFunction):
                Shape function at previous state
            nodes (Nodes):
                Nodes state containing grid size and inv_node_spacing
            interactions (Interactions):
                Interactions state containing particle-node interactions' distances.

        Returns:
            ShapeFunction:
                Updated shape function state for the particle and node pairs.
        """
        shapef, shapef_grad, basis = jax.vmap(vmap_linear_shapefunction, in_axes=(0, None), out_axes=(0, 0,0))(
            interactions.intr_dist, nodes.inv_node_spacing
        )
        return self.replace(shapef=shapef, shapef_grad=shapef_grad)