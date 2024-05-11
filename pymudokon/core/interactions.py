"""Module for particle-node pair interactions.

Interactions between node-particles are determined by a stencil.
The stencil is a window/box around each particle with the
relative position of the particle to the node.

Interactions are an intermediate representation before
transfering information from particles to nodes (or vice versa)

The module contains the following main components:

- InteractionsContainer:
A JAX pytree (NamedTuple) with interaction state
for the particle and node pairs.
- init:
Initialize the state for the particle and node pairs.
- vmap_interactions:
Vectorized mapping of particle-node pair interactions.
- get_interactions:
Get the particle-node pair interactions.
"""

from typing import NamedTuple, Union

import jax
import jax.numpy as jnp

from .nodes import NodesContainer
from .particles import ParticlesContainer


class InteractionsContainer(NamedTuple):
    """Interaction state for the particle and node pairs.

    A stencil is used to efficiently get particle-node interactions.
    Stencil size is the number of nodes surrounding each particle.
    Number of interactions is the number of particles times stencil size.
    Typically, stencil size depends on the shapefunction/element used.
    For example, for a 2D linear element, stencil size 4 is used,
    3D linear element stencil size 8, and 2D quadratic element stencil size of 9.

    Attributes:
        intr_dist_array (Union[jnp.array, jnp.float32]:
            Distance between particle-node pair interactions.
            in the node's local coordinate system.
            Shape is `(number of particless*stencil_size,dim,1)`.
        intr_bins_array (Union[jnp.array, jnp.int32]):
            Node-grid bins of the particle-node pair interactions.
            Shape is `(number of particles*stencil_size,dim,1)`.
        intr_hash_array (Union[jnp.array, jnp.int32]):
            Cartesian hash of particle-node pair interactions.
            Shape is `(number of particles*stencil_size)`.
        stencil_array (Union[jnp.array, jnp.float32]):
            Stencil containing relative displacements of
            neighboring nodes for particle-node pair interactions.
    """

    # arrays
    intr_dist_array: Union[jnp.array, jnp.float32]
    intr_bins_array: Union[jnp.array, jnp.int32]
    intr_hashes_array: Union[jnp.array, jnp.int32]
    stencil_array: Union[jnp.array, jnp.float32]


def init(
    stencil_array: Union[jnp.array, jnp.float32], num_particles: jnp.int16
) -> InteractionsContainer:
    """Initialize the node-particle pair interactions.

    Args:
        num_particles (jnp.int16):
            Number of particles.
        stencil_array (Union[jnp.array, jnp.float32]):
            Stencil array used to calculate particle-node pair interactions.
            It is a window/box around each particle with the relative position of the particle to the node.
            Stencil array is normally, the same for each particle/node pair.
            Expected shape is `(stencil_size,dim)`.

    Returns:
        InteractionsContainer:
            Updated interactions state for the particle and node pairs.
    """
    stencil_size, dim = stencil_array.shape
    return InteractionsContainer(
        intr_dist_array=jnp.zeros(
            (num_particles*stencil_size, dim,1), dtype=jnp.float32
        ),
        intr_bins_array=jnp.zeros((num_particles*stencil_size, dim,1), dtype=jnp.int32),
        intr_hashes_array=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        stencil_array=stencil_array,
    )


# @partial(jax.vmap, in_axes=(None,0, None, None))
def vmap_interactions(
    position: Union[jnp.array, jnp.float32],
    stencil_array: Union[jnp.array, jnp.float32],
    origin: Union[jnp.array, jnp.float32],
    inv_node_spacing: jnp.float32,
    grid_size: jnp.int32,
) -> InteractionsContainer:
    """Vectorized mapping of particle-node pair interactions.

    Position array is mapped per particle (num_particles, dim) -> (dim,).

    Performs the following operations:
    - Calculate the relative position of the particle to the node.
    - Calculate the particle-node pair interactions (by repeating particle for each stencil point).
    - Calculate particle-node pair interaction distances.
    - Calculate particle-node pair grid ids.
    - Calculate particle-node pair hash ids.

    Args:
        position (Union[jnp.array, jnp.float32]):
            Position of the particle (vmap per particle).
            Expects shape `(num_particles,dim)` before vectorize.
        stencil_array (Union[jnp.array, jnp.float32]):
            Stencil array, displacement of neighboring nodes.
            Expects shape `(stencil_size,dim)` static.
        origin (Union[jnp.array, jnp.float32]):
            Grid origin. Expected shape `(dim,)`.
        inv_node_spacing (jnp.float32):
            Inverse of the node spacing.
        grid_size (jnp.int32):
            Grid size/ total number of nodes about each axis.
            Expects shape `(dim,)`.

    """
    # TODO see if intr_bins actually needs to be stored

    # window_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    stencil_size, dim = stencil_array.shape

    # current shape per particle (dim,)
    # relative position of the particle to the background grid
    rel_pos = (position - origin) * inv_node_spacing

    # transforms shape per particle (stencil_size, dim)
    repeat_rel_pos = jnp.tile(rel_pos, (stencil_size, 1))

    # hereafter, all operations are per particle-per interactions
    # shape per particle (stencil_size, dim), except hashes
    intr_node_pos = jnp.floor(repeat_rel_pos + stencil_array)

    intr_dist = repeat_rel_pos - intr_node_pos

    intr_bins = intr_node_pos.astype(jnp.int32)

    intr_hashes = (intr_node_pos[:, 0] + intr_node_pos[:, 1] * grid_size[0]).astype(
        jnp.int32
    )

    # shape for all particle (after vmap) is
    # (num_particles, stencil_size, dim) for distances and bins
    # (num_particles, stencil_size) for hashes
    return intr_dist, intr_bins, intr_hashes


@jax.jit
def get_interactions(
    interactions_state: InteractionsContainer,
    particles_state: ParticlesContainer,
    nodes_state: NodesContainer,
) -> InteractionsContainer:
    """Get the particle-node pair interactions.

    Top level function to get the particle-node pair interactions.

    Arrays are reshaped to `(num_particles*stencil_size,dim,1)` and `(num_particles*stencil_size,1,1)`
    to be consistent during the transfer of information from particles to nodes.
    
    Args:
        interactions_state (InteractionsContainer):
            interactions state for the particle and node pairs.
        particles_state (ParticlesContainer):
            particles state.
        nodes_state (NodesContainer):
            nodes state.

    Returns:
        InteractionsContainer:
            Updated interactions state for the particle and node pairs.
    """
    stencil_size, dim = interactions_state.stencil_array.shape
    
    intr_dist_array, intr_bins_array, intr_hashes_array = jax.vmap(
        vmap_interactions,
        in_axes=(0, None, None, None, None),
        out_axes=(0, 0, 0),
    )(
        particles_state.positions_array,
        interactions_state.stencil_array,
        nodes_state.origin,
        nodes_state.inv_node_spacing,
        nodes_state.grid_size,
    )

    return interactions_state._replace(
        intr_dist_array=intr_dist_array.reshape(-1, dim, 1),
        intr_bins_array=intr_bins_array.reshape(-1, dim, 1),
        intr_hashes_array=intr_hashes_array.reshape(-1),
    )
