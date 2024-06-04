"""Module for particle-node pair interactions.

Interactions dataclass is the base for all shape functions. The interaction between node-particles are determined by a
stencil. The stencil is a window/box around each particle with the relative position of the particle to the node.

Interactions are an intermediate representation before transferring information from particles to nodes (or vice versa).
These arrays are typically denoted as `intr_...` (interaction arrays).
"""

from typing import Tuple
from typing_extensions import Self

from flax import struct

import jax

from .nodes import Nodes
from .particles import Particles

from functools import partial

@struct.dataclass
class Interactions:
    """Interaction state for the particle and node pairs.

    Each shapefunction inherits this class.

    A stencil is used to efficiently get particle-node interactions. Stencil size is the number of nodes surrounding
    each particle. Number of interactions is the number of particles times stencil size. Typically, stencil size depends
    on the shapefunction/element used. For example, for a 2D linear element, stencil size 4 is used, 3D linear element
    stencil size 8, and 2D quadratic element stencil size of 9.

    Attributes:
        intr_dist (jax.Array: Particle-node interactions distance in the nodes' coordinate system.
            `(num_particles*stencil_size,dim,1)`.
        intr_bins (jax.Array): Node-grid bins of the particle-node  interactions
            `(num_particles*stencil_size,dim,1)`, type int32.
        intr_hash (jax.Array): Cartesian hash of particle-node pair interactions
            `(num_particles*stencil_size)`, type int32.
        intr_shapef (jax.Array): Shape functions for the particle-node pair interactions
            `(num_particles, stencil_size,1)`
        intr_shapef_grad (jax.Array): Shape function gradients for the particle-node pair interactions
            `(num_particles, stencil_size, dim)`
    """

    intr_dist: jax.Array
    intr_bins: jax.Array
    intr_hashes: jax.Array

    intr_shapef: jax.Array
    intr_shapef_grad: jax.Array
    stencil: jax.Array

    @classmethod
    def create(cls: Self, stencil: jax.Array, num_particles: jax.numpy.int16) -> Self:
        """Initialize the empty state or node-particle pair interactions.

        This function is typically not called directly, for testing purposes only.

        Args:
            cls (Self): Reference to the self type.
            stencil_size (jax.numpy.int16): Size of the stencil.
            num_particles (jax.numpy.int16): Number of particles.
            stencil (jax.Array):
                Stencil array used to calculate particle-node pair interactions.
                It is a window/box around each particle with the relative position of the particle to the node.
                Stencil array is normally, the same for each particle/node pair.
                Expected shape is `(stencil_size,dim)`.
            dim (jax.numpy.int16):
                Dimension of the problem

        Returns:
            Interactions:
                Updated interactions state for the particle and node pairs.
        """
        stencil_size, dim = stencil.shape

        return cls(
        intr_dist = jax.numpy.zeros((num_particles * stencil_size, dim, 1), dtype=jax.numpy.float32),
        intr_bins = jax.numpy.zeros((num_particles * stencil_size, dim, 1), dtype=jax.numpy.int32),
        intr_hashes = jax.numpy.zeros((num_particles * stencil_size), dtype=jax.numpy.int32),
        intr_shapef = jax.numpy.zeros((num_particles, stencil_size), dtype=jax.numpy.float32),
        intr_shapef_grad = jax.numpy.zeros((num_particles, stencil_size, dim), dtype=jax.numpy.float32),
        stencil = stencil
        )

    @jax.jit
    def get_interactions(
        self: Self,
        particles: Particles,
        nodes: Nodes,
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
            shapefunctions (ShapeFunction):
                Shape functions state.
            nodes (Nodes):
                Nodes state.

        Returns:
            Interactions:
                Updated interactions state for the particle and node pairs.
        """
        stencil_size, dim = self.stencil.shape

        intr_dist, intr_bins, intr_hashes = self.vmap_interactions(
            particles.positions,
            self.stencil,
            nodes.origin,
            nodes.inv_node_spacing,
            nodes.grid_size,
        )

        return self.replace(
            intr_dist=intr_dist.reshape(-1, dim, 1),
            intr_bins=intr_bins.reshape(-1, dim, 1),
            intr_hashes=intr_hashes.reshape(-1),
        )

    def set_boundary_nodes(self: Self, nodes: Nodes) -> Nodes:
        """Set node species at boundaries (See `cubic` shape function)."""
        return nodes

    
    @partial(jax.vmap, in_axes=(None,0, None, None, None, None), out_axes=(0, 0, 0))
    def vmap_interactions(
        self: Self,
        position: jax.Array,
        stencil: jax.Array,
        origin: jax.Array,
        inv_node_spacing: jax.numpy.float32,
        grid_size: jax.numpy.int32,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Vectorized mapping of particle-node pair interactions.

        Position array is mapped per particle via vmap (num_particles, dim) -> (dim,).

        Performs the following operations:
        - Calculate the relative position of the particle to the node.
        - Calculate the particle-node pair interactions (by repeating particle for each stencil point).
        - Calculate particle-node pair interaction distances.
        - Calculate particle-node pair grid ids.
        - Calculate particle-node pair hash ids.

        Args:
            position (jax.Array): Spatial coordinates of particle. Expects shape `(num_particles,dim)` vectorized to `(dim,)`.
            stencil (jax.Array): Stencil array, displacement of neighboring nodes. Expects shape `(stencil_size,dim)` static.
            origin (jax.Array): Grid origin. Expected shape `(dim,)` static.
            inv_node_spacing (jax.numpy.float32): Inverse of the node spacing, static.
            grid_size (jax.numpy.int32): Grid size/ total number of nodes about each axis. Expects shape `(dim,)`, static.

        Returns:
            (Tuple[jax.Array, jax.Array, jax.Array]):
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
        repeat_rel_pos = jax.numpy.tile(rel_pos, (stencil_size, 1))

        # hereafter, all operations are per particle-per interactions
        # shape per particle (stencil_size, dim), except hashes
        intr_node_pos = jax.numpy.floor(repeat_rel_pos) + stencil

        intr_dist = repeat_rel_pos - intr_node_pos

        intr_bins = intr_node_pos.astype(jax.numpy.int32)

        intr_hashes = (intr_node_pos[:, 0] + intr_node_pos[:, 1] * grid_size[0]).astype(
            jax.numpy.int32
        )

        # shape for all particle (after vmap) is
        # (num_particles, stencil_size, dim) for distances and bins
        # (num_particles, stencil_size) for hashes
        return intr_dist, intr_bins, intr_hashes
