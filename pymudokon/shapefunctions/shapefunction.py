"""Module for shapefunction base class.

The interaction between node-particles are determined by a stencil.
The stencil is a window/box around each particle with the relative position of the particle to the node.

Shapefunctions use interactions which are an intermediate representation before transferring information
from particles to nodes (or vice versa). These arrays are typically denoted as `intr_...` (interaction arrays).
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Self

from ..core.nodes import Nodes


@struct.dataclass
class ShapeFunction:
    """Interaction state for the particle and node pairs.

    Each shapefunction inherits this class.

    Attributes:
        intr_hash (jax.Array): Cartesian hash of particle-node pair interactions
            `(num_particles*stencil_size)`, type int32.
        intr_shapef (jax.Array): Shape functions for the particle-node pair interactions
            `(num_particles, stencil_size,1)`
        intr_shapef_grad (jax.Array): Shape function gradients for the particle-node pair interactions
            `(num_particles, stencil_size, dim)`
    """

    intr_shapef: jax.Array
    intr_shapef_grad: jax.Array
    intr_hashes: jax.Array
    intr_ids: jax.Array
    stencil: jax.Array

    def set_boundary_nodes(self: Self, nodes: Nodes) -> Nodes:
        return nodes

    @partial(jax.vmap, in_axes=(None, 0, None, None, None, None, None), out_axes=(0, 0))
    def vmap_intr(
        self: Self,
        intr_id: jax.Array,
        position: jax.Array,
        origin: jax.Array,
        inv_node_spacing: jnp.float32,
        grid_size: jnp.int32,
        dim: jnp.int32,
    ) -> Tuple[jax.Array, jax.Array]:
        """Vectorized mapping of particle-node pair interactions.

        Position array is batched per particle via vmap shape (num_particles, dim) -> (dim,).

        Args:
            self: ShapeFunction class.
            position (jax.Array):
                Spatial coordinates of particle. Expects shape `(num_particles,dim)` vectorized to `(dim,)`.
            origin (jax.Array): Grid origin. Expected shape `(dim,)` static.
            inv_node_spacing (jnp.float32): Inverse of the node spacing, static.
            grid_size (jnp.int32): Grid size/ total number of nodes about each axis. Expects shape `(dim,)`, static.

        Returns:
            (Tuple[jax.Array, jax.Array, jax.Array]):
                Tuple of particle-node pair interactions.
                - intr_dist: Particle-node pair interaction distances.
                - intr_hashes: Particle-node pair hash ids.
        """
        stencil_size, dim = self.stencil.shape

        # Solution procedure:

        # 0. Get particle and stencil ids
        particle_id = (intr_id / stencil_size).astype(jnp.int32)
        stencil_id = (intr_id % stencil_size).astype(jnp.int16)

        # 1. Calculate the relative position of the particle to the node.
        particle_pos = position.at[particle_id].get(indices_are_sorted=True)
        rel_pos = (particle_pos - origin) * inv_node_spacing

        # 2. Calculate particle-node pair interaction distances.
        stencil_pos = self.stencil.at[stencil_id].get(indices_are_sorted=True, unique_indices=True)
        intr_n_pos = jnp.floor(rel_pos) + stencil_pos

        intr_dist = rel_pos - intr_n_pos

        if dim == 1:
            intr_hashes = intr_n_pos.astype(jnp.int32)
        elif dim == 2:
            intr_hashes = (intr_n_pos[0] + intr_n_pos[1] * grid_size[0]).astype(jnp.int32)
        else:
            intr_hashes = (
                intr_n_pos[0] + intr_n_pos[1] * grid_size[0] + intr_n_pos[2] * grid_size[0] * grid_size[1]
            ).astype(jnp.int32)

        return intr_dist, intr_hashes
