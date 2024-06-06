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
    stencil: jax.Array

    @partial(jax.vmap, in_axes=(None,0, None, None, None), out_axes=(0, 0, 0))
    def vmap_get_interactions(
        self: Self,
        position: jax.Array,
        origin: jax.Array,
        inv_node_spacing: jnp.float32,
        grid_size: jnp.int32,
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
        # 1. Calculate the relative position of the particle to the node.
        rel_pos = (position - origin) * inv_node_spacing

        # 2. Calculate the particle-node pair interactions (by repeating particle for each stencil point).
        repeat_rel_pos = jnp.tile(rel_pos, (stencil_size, 1))

        # 3. Calculate particle-node pair interaction distances.
        intr_n_pos = jnp.floor(repeat_rel_pos) + self.stencil
        intr_dist = repeat_rel_pos - intr_n_pos

        # 4. Calculate particle-node pair hash ids.
        intr_hashes = jax.lax.switch(
            dim-1,
            (lambda pos,grid: pos.astype(jnp.int32),
            lambda pos,grid: (pos[:, 0] + pos[:, 1] * grid[0]).astype(jnp.int32),
            lambda pos,grid: (pos[:, 0] + pos[:, 1] * grid[0] + pos[:, 2] * grid[0] * grid[1]).astype(jnp.int32)
            ),
            (intr_n_pos,grid_size)
        )

        # 5. Return the particle-node pair interactions.
        # shape for intr_dist (after vmap) is (num_particles, stencil_size, dim),
        # (num_particles, stencil_size) for hashes
        return intr_dist, intr_hashes
