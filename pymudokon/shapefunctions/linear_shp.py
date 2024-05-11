"""Module for containing the linear shape functions.

Shape functions are calculated for the particle-node interactions.

The module contains the following main components:

- ShapeFunctionContainer:
    A JAX pytree for shapefunction state
- init:
    Initialize the state for the shape functions.
- vmap_linear_shapefunction:
    Vectorized linear shape function calculation.
- calculate_shapefunction:
    Top level function to calculate the shape functions.
"""
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp

from ..core.interactions import InteractionsContainer
from ..core.nodes import NodesContainer


class ShapeFunctionContainer(NamedTuple):
    """Shapefunctions state for the particle-node interactions.

    The arrays are of shape (num_particles, stencil_size),
    and (num_particles, stencil_size, dim) respectively.

    Attributes:
        shapef_array Union[jnp.array, jnp.float32]:
            Shape function array.
        shapef_grad_array Union[jnp.array, jnp.float32]:
            Shape function gradient array.
    """
    shapef_array: Union[jnp.array, jnp.float32]
    shapef_grad_array: Union[jnp.array, jnp.float32]


def init(num_particles: jnp.int32, stencil_size: jnp.int32, dim: jnp.int16) -> ShapeFunctionContainer:
    """Initializes the shape function container.

    Args:
        num_particles (jnp.int32):
            number of particles
        stencil_size (jnp.int32):
            size of the stencil
        dim (jnp.int16):
            dimension of the problem

    Returns:
        ShapeFunctionContainer:
            container for shape functions and gradients
    """
    return ShapeFunctionContainer(
        shapef_array=jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
        shapef_grad_array=jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
    )


def vmap_linear_shapefunction(
    intr_dist: jnp.array,
    inv_node_spacing: jnp.float32,
) -> ShapeFunctionContainer:
    """Vectorized linear shape function calculation.

    Calculate the shape function, and then its gradient

    Args:
        intr_dist (jnp.array):
            particle-node pair interactions distance.
        inv_node_spacing (jnp.float32):
            inverse node spacing.

    Returns:
        ShapeFunctionContainer:
        Updated shape function state for the particle and node pairs.
    """
    abs_intr_dist = jnp.abs(intr_dist)
    basis = jnp.where(abs_intr_dist < 1.0, 1.0 - abs_intr_dist, 0.0)
    dbasis = jnp.where(abs_intr_dist < 1.0, -jnp.sign(intr_dist) * inv_node_spacing, 0.0)
    N0 = basis[0,:]
    N1 = basis[1,:]
    dN0 = dbasis[0,:]
    dN1 = dbasis[1,:]

    shapef_array = jnp.expand_dims(N0 * N1, axis=1)

    shapef_grad_array = jnp.array([dN0 * N1, N0 * dN1])

    # shapes of returned array are
    # (num_particles*stencil_size, 1, 1)
    # and (num_particles*stencil_size, dim,1)
    # respectively
    return shapef_array, shapef_grad_array


def calculate_shapefunction(
    shapefunctions_state: ShapeFunctionContainer,
    nodes_state: NodesContainer,
    interactions_state: InteractionsContainer,
) -> ShapeFunctionContainer:
    """Top level function to calculate the shape functions.

    Args:
        shapefunction_state (ShapeFunctionContainer):
            shape function at previous state
        nodes_state (NodesContainer):
            nodes state containing grid size and inv_node_spacing
        interactions_state (InteractionsContainer):
            interactions state containing particle-node
            interactions distances

    Returns:
        ShapeFunctionContainer: _description_
    """

    shapef_array, shapef_grad_array = jax.vmap(vmap_linear_shapefunction, in_axes=(0, None), out_axes=(0, 0))(
        interactions_state.intr_dist_array, nodes_state.inv_node_spacing
    )

    return shapefunctions_state._replace(shapef_array=shapef_array, shapef_grad_array=shapef_grad_array)
