"""Module for particle-node pair interactions.

Interactions between node-particles are determined by a stencil.
The stencil is a window/box around each particle with the
relative position of the particle to the node.

Interactions are an intermediate representation before
transfering information from particles to nodes (or vice versa).
These arrays are typically denoted as `intr_...` (interaction arrays).
They have a shape size of 3.
"""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from .base import Base
from .nodes import Nodes
from .particles import Particles
from ..shapefunctions.base_shp import BaseShapeFunction


def vmap_interactions(
    position: Array,
    stencil: Array,
    origin: Array,
    inv_node_spacing: jnp.float32,
    grid_size: jnp.int32,
) -> Tuple[Array, Array, Array]:
    """Vectorized mapping of particle-node pair interactions.

    Position array is mapped per particle via vmap (num_particles, dim) -> (dim,).

    Performs the following operations:
    - Calculate the relative position of the particle to the node.
    - Calculate the particle-node pair interactions (by repeating particle for each stencil point).
    - Calculate particle-node pair interaction distances.
    - Calculate particle-node pair grid ids.
    - Calculate particle-node pair hash ids.

    Args:
        position (Array):
            Position of the particle (vmap per particle).
            Expects shape `(num_particles,dim)` before vectorize.
        stencil (Array):
            Stencil array, displacement of neighboring nodes.
            Expects shape `(stencil_size,dim)` static.
        origin (Array):
            Grid origin. Expected shape `(dim,)`.
        inv_node_spacing (jnp.float32):
            Inverse of the node spacing.
        grid_size (jnp.int32):
            Grid size/ total number of nodes about each axis.
            Expects shape `(dim,)`.

    Returns:
        (Tuple[Array, Array, Array]):
            Tuple of particle-node pair interactions.
            - intr_dist: Particle-node pair interaction distances.
            - intr_bins: Particle-node pair grid ids.
            - intr_hashes: Particle-node pair hash ids.
    """
    stencil_size, dim = stencil.shape

    # current shape per particle (dim,)
    # relative position of the particle to the background grid
    rel_pos = (position - origin) * inv_node_spacing

    # transforms shape per particle (stencil_size, dim)
    repeat_rel_pos = jnp.tile(rel_pos, (stencil_size, 1))

    # hereafter, all operations are per particle-per interactions
    # shape per particle (stencil_size, dim), except hashes
    intr_node_pos = jnp.floor(repeat_rel_pos) + stencil

    intr_dist = repeat_rel_pos - intr_node_pos

    intr_bins = intr_node_pos.astype(jnp.int32)

    intr_hashes = (intr_node_pos[:, 0] + intr_node_pos[:, 1] * grid_size[0]).astype(jnp.int32)

    # shape for all particle (after vmap) is
    # (num_particles, stencil_size, dim) for distances and bins
    # (num_particles, stencil_size) for hashes
    return intr_dist, intr_bins, intr_hashes


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Interactions(Base):
    """Interaction state for the particle and node pairs.

    A stencil is used to efficiently get particle-node interactions.
    Stencil size is the number of nodes surrounding each particle.
    Number of interactions is the number of particles times stencil size.
    Typically, stencil size depends on the shapefunction/element used.
    For example, for a 2D linear element, stencil size 4 is used,
    3D linear element stencil size 8, and 2D quadratic element stencil size of 9.

    Attributes:
        intr_dist (Array:
            particle-node distance interactions in the nodes' coordinate system.`(num_particles*stencil_size,dim,1)`.
        intr_bins (Array):
            Node-grid bins of the particle-node  interactions `(num_particles*stencil_size,dim,1)`, type int32.
        intr_hash (Array):
            Cartesian hash of particle-node pair interactions `(num_particles*stencil_size)`, type int32.
    """

    # arrays
    intr_dist: Array
    intr_bins: Array
    intr_hashes: Array

    @classmethod
    def register(cls: Self, stencil_size: jnp.int16, num_particles: jnp.int16, dim: jnp.int16) -> Self:
        """Initialize the node-particle pair interactions.

        Args:
            cls (Self):
                Reference to the self type.
            num_particles (jnp.int16):
                Number of particles.
            stencil (Array):
                Stencil array used to calculate particle-node pair interactions.
                It is a window/box around each particle with the relative position of the particle to the node.
                Stencil array is normally, the same for each particle/node pair.
                Expected shape is `(stencil_size,dim)`.
            dim (jnp.int16):
                Dimension of the problem

        Returns:
            Interactions:
                Updated interactions state for the particle and node pairs.
        """
        return cls(
            intr_dist=jnp.zeros((num_particles * stencil_size, dim, 1), dtype=jnp.float32),
            intr_bins=jnp.zeros((num_particles * stencil_size, dim, 1), dtype=jnp.int32),
            intr_hashes=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        )

    @jax.jit
    def get_interactions(
        self: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: BaseShapeFunction,
    ) -> Self:
        """Get the particle-node pair interactions.

        Top level function to get the particle-node pair interactions.

        Arrays are reshaped to `(num_particles*stencil_size,dim,1)` and `(num_particles*stencil_size,1,1)`
        to be consistent during the transfer of information from particles to nodes.

        Args:
            interactions (Interactions):
                Interactions state for the particle and node pairs.
            particles (Particles):
                Particles state.
            nodes (Nodes):
                Nodes state.

        Returns:
            Interactions:
                Updated interactions state for the particle and node pairs.
        """
        stencil_size, dim = shapefunctions.stencil.shape

        intr_dist, intr_bins, intr_hashes = jax.vmap(
            vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(
            particles.positions,
            shapefunctions.stencil,
            nodes.origin,
            nodes.inv_node_spacing,
            nodes.grid_size,
        )

        return self.replace(
            intr_dist=intr_dist.reshape(-1, dim, 1),
            intr_bins=intr_bins.reshape(-1, dim, 1),
            intr_hashes=intr_hashes.reshape(-1),
        )
