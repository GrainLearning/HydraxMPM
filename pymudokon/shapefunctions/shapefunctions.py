"""Module containing the shapefunction base class."""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

# from ..core.nodes import Nodes


@chex.dataclass(mappable_dataclass=False, frozen=True)
class ShapeFunction:
    """Shapefunction base class. Contains base method to calculate relative distances between particles and nodes.

    Stores particle-node pair interaction arrays. Not to be used directly, but to be inherited by specific
    shape functions.

    Attributes:
        intr_hash: Cartesian spatial hash of particle-node pair interactions `(num_particles*stencil_size)`
        intr_shapef: Shape functions for the particle-node pair interactions `(num_particles*stencil_size,1)`
        intr_shapef_grad: Shape function gradients for the particle-node pair interactions
            `(num_particles, stencil_size, dim)`
        intr_ids: Particle-node pair interaction ids `(num_particles*stencil_size)`
    """

    intr_hashes: chex.Array
    intr_shapef: chex.Array
    intr_shapef_grad: chex.Array
    intr_ids: chex.Array
    stencil: chex.Array

    # def set_boundary_nodes(self: Self, nodes: Nodes) -> Nodes:
    #     """Placeholder method to set boundary nodes."""
    #     return nodes

    @partial(jax.vmap, in_axes=(None, 0, None, None, None, None), out_axes=(0, 0))
    def vmap_intr(
        self: Self,
        intr_id: chex.ArrayBatched,
        position: chex.Array,
        origin: chex.Array,
        inv_node_spacing: jnp.float32,
        grid_size: jnp.int32,
    ) -> Tuple[chex.Array, chex.Array]:
        """Calculate particle-node pair interaction distances and hashes. Only interaction ids are vectorized.

        Args:
            self: ShapeFunction class.
            intr_id: Particle-node pair interaction ids.
            position: Particle coordinates.
            origin: Grid origin. Expected shape `(dim,)`.
            inv_node_spacing: Inverse of the node spacing.
            grid_size: Grid size/ total number of nodes about each axis. Expects shape `(dim,)`.

        Returns:
            Tuple containing:
                - intr_dist: Particle-node pair interaction distances.
                - intr_hashes: Particle-node pair hash ids.
        """
        stencil_size, dim = self.stencil.shape

        particle_id = (intr_id / stencil_size).astype(jnp.int32)
        stencil_id = (intr_id % stencil_size).astype(jnp.int16)

        # Relative position of the particle to the node.
        particle_pos = position.at[particle_id].get()
        rel_pos = (particle_pos - origin) * inv_node_spacing

        # article-node pair interaction distances.
        stencil_pos = self.stencil.at[stencil_id].get()
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
