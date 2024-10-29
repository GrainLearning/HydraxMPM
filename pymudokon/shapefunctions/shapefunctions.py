"""Module containing the shapefunction base class."""

from functools import partial
from typing import Tuple
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

from jax.sharding import Sharding
import jax


@chex.dataclass(mappable_dataclass=False, frozen=True)
class ShapeFunction:
    """Contains base method to calculate relative distances between particles and nodes.

    Stores particle-node pair interaction arrays. Not to be used directly, but to be
    inherited by specific shape functions.

    Stencil size depends on the dimension of the problem and shape function used.

    Attributes:
        intr_hash_stack: Cartesian spatial hash of particle-node pair interactions
            `(num_particles*stencil_size)`
        intr_shapef_stack: Shape functions for the particle-node pair interactions
            `(num_particles*stencil_size,1)`
        intr_shapef_grad_stack: Shape function gradients for the particle-node
            interactions `(num_particles, stencil_size, dim)`
        intr_id_stack: Particle-node pair interaction ids `(num_particles*stencil_size)`
    """

    intr_hash_stack: chex.Array
    intr_shapef_stack: chex.Array
    intr_shapef_grad_stack: chex.Array
    intr_id_stack: chex.Array
    stencil: chex.Array

    @partial(jax.vmap, in_axes=(None, 0, None, None, None, None), out_axes=(0, 0))
    def vmap_intr(
        self: Self,
        intr_id: chex.ArrayBatched,
        position_stack: chex.Array,
        origin: chex.Array,
        inv_node_spacing: jnp.float32,
        grid_size: jnp.int32,
    ) -> Tuple[chex.Array, chex.Array]:
        """Calculate particle-node pair interaction distances and hashes.

        Only interaction ids are vectorized.

        Args:
            self: ShapeFunction class.
            intr_id: Particle-node pair interaction ids.
            position_stack: Particle coordinates array.
            origin: Grid origin. Expected shape `(dim,)`.
            inv_node_spacing: Inverse of the node spacing.
            grid_size: Grid size/ total number of nodes about each axis.
                Expects shape `(dim,)`.

        Returns:
            Tuple containing:
                - intr_dist: Particle-node pair interaction distances.
                - intr_hashes: Particle-node pair hash ids.
        """
        stencil_size, dim = self.stencil.shape

        particle_id = (intr_id / stencil_size).astype(jnp.int32)
        stencil_id = (intr_id % stencil_size).astype(jnp.int16)

        # Relative position of the particle to the node.
        particle_pos = position_stack.at[particle_id].get()
        rel_pos = (particle_pos - origin) * inv_node_spacing

        # article-node pair interaction distances.
        stencil_pos = self.stencil.at[stencil_id].get()
        intr_n_pos = jnp.floor(rel_pos) + stencil_pos

        intr_dist = rel_pos - intr_n_pos

        if dim == 1:
            intr_hashes = intr_n_pos.astype(jnp.int32)
        elif dim == 2:
            intr_hashes = (intr_n_pos[1] + intr_n_pos[0] * grid_size[1]).astype(
                jnp.int32
            )
        else:
            intr_hashes = (
                intr_n_pos[2]
                + intr_n_pos[0] * grid_size[2]
                + intr_n_pos[1] * grid_size[2] * grid_size[0]
            ).astype(jnp.int32)

        return intr_dist, intr_hashes

    def distributed(self: Self, device: Sharding):
        intr_hash_stack = jax.device_put(self.intr_hash_stack, device)
        intr_shapef_stack = jax.device_put(self.intr_shapef_stack, device)
        intr_shapef_grad_stack = jax.device_put(self.intr_shapef_grad_stack, device)
        intr_id_stack = jax.device_put(self.intr_id_stack, device)
        stencil = jax.device_put(self.stencil, device)

        return self.replace(
            intr_hash_stack=intr_hash_stack,
            intr_shapef_stack=intr_shapef_stack,
            intr_shapef_grad_stack=intr_shapef_grad_stack,
            intr_id_stack=intr_id_stack,
            stencil=stencil,
        )
