# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module handles the mapping of physical quantities between Material Points (Lagrangian)
    and the Grid (Eulerian) in the Material Point Method (MPM).

    The `InteractionCache` class stores precomputed interaction data for efficient mapping to and from material points


    The `ShapeFunctionMapping` class is responsible for:
    1.  **Connectivity**: Determining which grid nodes interact with which particles
        based on the shape function support (stencil).
    2.  **Shape Function Evaluation**: Calculating shape function values ($N_I(x_p)$) and
        gradients ($\nabla N_I(x_p)$).
    3.  **Scatter (P2G)**: Transferring mass and momentum from particles to grid nodes.
    4.  **Gather (G2P)**: Interpolating grid velocities or forces back to particles.

"""

# Check Quadratic shape functions

from typing import Callable, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import UInt, Float, Array, Int

from .cubic import vmap_cubic_shapefunction
from .quadratic import vmap_quadratic_shapefunction
from .linear import vmap_linear_shapefunction

# Kernel registry (shape function and stencil offsets)

kernels = {
    "linear": (vmap_linear_shapefunction, jnp.arange(2)),
    "quadratic": (vmap_quadratic_shapefunction, jnp.arange(3)),
    "cubic": (vmap_cubic_shapefunction, jnp.arange(4) - 1),
}


class InteractionCache(eqx.Module):
    """

    Interaction between particles and nodes

    Stores interaction data for a single time-step.

    Re-compute every timestep, and should not be saved to disk.


    Attributes:
        node_hashes: spatial hashes of interaction
        shape_vals: shape function values at interactions
        shape_grads: shape function gradients at interactions
        rel_dist: (normalized by cell size) (xp - xg) / h
        relative distance from particle to node at interactions (used by APIC/AFLIP)

        point_ids: material point indices for each interaction
        stencil_ids: stencil indices for each interaction
        _stencil_offsets: precomputed stencil offsets for the shape function

    """

    node_hashes: UInt[Array, "n_interactions"]
    shape_vals: Float[Array, "n_interactions"]
    cpic_mask: Float[Array, "n_interactions"]

    shape_grads: Float[Array, "n_interactions 3"]
    rel_dist: Float[Array, "n_interactions 3"]

    point_ids: UInt[Array, "n_interactions"]
    stencil_ids: UInt[Array, "n_interactions"]

    @property
    def dim(self: Self) -> int:
        """Get dimension"""
        return self.shape_grads.shape[1]

    @property
    def num_interactions(self: Self) -> int:
        """Get number of interactions"""
        return self.node_hashes.shape[0]



class ShapeFunctionMapping(eqx.Module):
    """
    Stateless logic operator for mapping particles to grid, and back.
    """

    dim: int = eqx.field(static=True)
    _padding: tuple = eqx.field(static=True)
    _shapefunction_call: Callable = eqx.field(static=True)

    shapefunction: str = eqx.field(static=True)

    _stencil_offsets: Int[Array, "window dim"]

    @property
    def window_size(self: Self) -> int:
        """Get stencil window size"""
        return self._stencil_offsets.shape[0]

    def __init__(
        self,
        shapefunction: str,
        dim: int,
    ):
        self.dim = dim
        self._padding = (0, 3 - dim)

        # select Kernel and 1D window support based on name
        self._shapefunction_call = kernels[shapefunction][0]
        self.shapefunction = shapefunction


        window_1d = kernels[self.shapefunction][1]

        # build stencil n-dimensional stencil offsets
        mesh = jnp.meshgrid(*[window_1d] * dim, indexing="ij")

        # set data fields for stencils
        self._stencil_offsets = jnp.stack(mesh, axis=-1).reshape(-1, dim)


    # def create_cache(
    #     self,
    #     num_points: int,
    #     dim: int,
    # ):
    #     """
    #     Creates an empty InteractionCache for given number of points and dimension. Used within MPM solver.Re
    #     """


    #     window_size = _stencil_offsets.shape[0]

    #     # pre-compute indices
    #     num_interactions = num_points * window_size
    #     flat_indices = jnp.arange(num_interactions)

    #     # set data fields for point and stencil ids
    #     point_ids = (flat_indices // window_size).astype(jnp.uint32)
    #     stencil_ids = (flat_indices % window_size).astype(jnp.uint32)

    #     return InteractionCache(
    #         node_hashes=jnp.zeros((num_interactions,), dtype=jnp.uint32),
    #         shape_vals=jnp.zeros((num_interactions,)),
    #         shape_grads=jnp.zeros((num_interactions, 3)),
    #         rel_dist=jnp.zeros((num_interactions, 3)),
    #         cpic_mask=jnp.ones((num_interactions,)),
    #         point_ids=point_ids,
    #         stencil_ids=stencil_ids,
    #         _stencil_offsets=_stencil_offsets,
    #     )

    def compute(
        self,
        position_stack: Array,
        origin: Array | tuple,
        grid_size: Array | tuple,
        inv_cell_size: float,
    ) -> InteractionCache:
        """
        Calculates hashes, shape values, and gradients from current positions.
        Returns a generic InteractionCache.
        """
        origin = jnp.array(origin)

        num_points = position_stack.shape[0]
        
        window_size = self._stencil_offsets.shape[0]
        
        num_interactions = num_points * window_size


        # 1. Generate Indices (Fast integer math)
        flat_indices = jnp.arange(num_interactions)
        point_ids = (flat_indices // window_size).astype(jnp.uint32)
        stencil_ids = (flat_indices % window_size).astype(jnp.uint32)

        # Quadratic splines require a -0.5 cell shift to align the 3-node stencil
        needs_shift = self.shapefunction == "quadratic"


        def compute_single_interaction(p_id, s_id):

            # material point position
            xp = position_stack[p_id]

            # interaction offset
            offset = self._stencil_offsets[s_id]

            # normalized grid coordinate to float
            base_pos = (xp - origin) * inv_cell_size
            
            # apply shift for quadratic splines
            if needs_shift:
                base_id_float = jnp.floor(base_pos - 0.5)
            else:
                base_id_float = jnp.floor(base_pos)



            # compute node index
            node_idx = base_id_float.astype(jnp.int32) + offset

            # check if node_idx is strictly inside [0, grid_size)
            # if not, set shapefunctions to zero
            is_valid = jnp.all((node_idx >= 0) & (node_idx < jnp.array(grid_size)))

            # clip/wrap mode handles BCs implicitly or prevents crash
            node_hash = jnp.ravel_multi_index(node_idx, grid_size, mode="clip").astype(
                jnp.uint32
            )

            # compute shape function
            # dist_vec = (pos material point - pos grid) / cell size
            dist_vec = base_pos - node_idx

            # TODO interaction type, for closed nodes in cubic splines?
            val, grad = self._shapefunction_call(
                dist_vec, inv_cell_size, self.dim, self._padding, intr_node_type=0
            )

            # ignore ghost nodes outside grid by clipping the indices
            val = jnp.where(is_valid, val, 0.0)
            grad = jnp.where(is_valid, grad, 0.0)

            # pad gradients/dist to 3D if needed, in case of 2D simulation
            if self.dim == 2:
                # grad is padded inside _shapefunction_call
                # TODO move outside?
                dist_vec = jnp.concatenate([dist_vec, jnp.array([0.0])])

            return node_hash, val, grad, dist_vec

        # vectorized over all interactions
        hashes, vals, grads, dists = jax.vmap(compute_single_interaction)(
            point_ids, stencil_ids
        )

        return InteractionCache(
            node_hashes=hashes,
            shape_vals=vals,
            shape_grads=grads,
            rel_dist=dists,
            cpic_mask=jnp.ones((num_interactions,)), # Default mask
            point_ids=point_ids,
            stencil_ids=stencil_ids,
            # _stencil_offsets=self._stencil_offsets # this needed later?
        )

    # Particle to Grid transfer (P2G, Scatter)
    def scatter_to_grid(
        self,
        cache: InteractionCache,
        particle_data_stack: Float[Array, "num_points *shape"],
        particle_mass_stack: Float[Array, "num_points"],
        num_cells: int | UInt[Array, ""],
        small_mass_cutoff: float = 1e-9,
        normalize: bool = True,
    ) -> Array:
        """
        Generic P2G (Particle to Grid) transfer method for `particle_data_stack` given `particle_mass_stack`

        Mainly used for post-processing,

        Arbitrary particle_data_stack`  data of shape 'num_points shape' allows to handle:
            - Scalars: (num_points, ) -> (num_cells,)
            - Vectors: (num_points, dim) -> (num_cells, dim)
            - Tensors: (num_points, dim, dim) -> (num_cells, dim, dim)
        """
        particle_data_shape = particle_data_stack.shape[1:]

        # node hashes
        point_ids = cache.point_ids
        node_hashes = cache.node_hashes

        # map data to interaction list
        intr_data_stack = particle_data_stack[point_ids]
        intr_mass_stack = particle_mass_stack[point_ids]

        # compute shape function weights (N_I(xp) * m_p)
        intr_weight_stack = cache.shape_vals * intr_mass_stack

        # Apply Weights to data by broadcast weight_itr against the data
        # if data is scalar (N_intr,), result is (N_intr,)
        # if data is vector (N_intr, dim), result is (N_intr, dim)

        # Ensure weight (N_intr,) can multiply data (N_intr, D1, D2...)
        # Ensure weight (N_intr, 1) can multiply data (N_intr, D1, D2...)
        # We want shape: (N_intr, 1, 1, ...) matching the rank of data features

        # Use shape[0] (N_intr) explicitly, rather than the full shape tuple which includes the '1'
        target_weight_shape = (intr_weight_stack.shape[0],) + (1,) * (
            intr_data_stack.ndim - 1
        )

        weight_shaped_stack = intr_weight_stack.reshape(target_weight_shape)

        weighted_data_stack = intr_data_stack * weight_shaped_stack

        flat_data_stack = weighted_data_stack.reshape(weighted_data_stack.shape[0], -1)

        grid_accum = jnp.zeros((num_cells, flat_data_stack.shape[1]))
        grid_accum = grid_accum.at[node_hashes].add(flat_data_stack)

        # 6. Normalize (Optional)
        if normalize:
            # A. Sum weights (N*m)
            # Initialize as 1D (num_cells,) because intr_weight_stack is 1D
            weight_accum = (
                jnp.zeros((num_cells,)).at[node_hashes].add(intr_weight_stack)
            )

            # B. Create Mask & Safe Weight (1D)
            mask = weight_accum > small_mass_cutoff
            weight_safe = jnp.where(mask, weight_accum, 1.0)

            # This creates shape (N, 1)
            mask_expanded = mask[:, None]
            weight_expanded = weight_safe[:, None]

            grid_accum = jnp.where(mask_expanded, grid_accum / weight_expanded, 0.0)

        return grid_accum.reshape(num_cells, *particle_data_shape)

    def gather_from_grid(
        self, cache: InteractionCache, grid_data_stack: Array
    ) -> Array:
        """
        Generic G2P interpolation using cached interactions.
        """
        # align grid data to interaction list, reindex to (N,)
        node_hashes = cache.node_hashes
        data_on_nodes = jnp.take(grid_data_stack, node_hashes, axis=0)

        # add dimensions so that broadcasting works correctly
        data_dims = data_on_nodes.ndim - 1

        target_weight_shape = (cache.shape_vals.shape[0],) + (1,) * data_dims
        weight_expanded = cache.shape_vals.reshape(target_weight_shape)

        # apply weights to interactions
        weighted_data = weight_expanded * data_on_nodes

        # flatten
        data_shape = grid_data_stack.shape[1:]
        flat_data = weighted_data.reshape(weighted_data.shape[0], -1)

        point_ids = cache.point_ids

        # accumulate (gather)
        particle_accum = jnp.zeros(
            (point_ids.shape[0] // cache.window_size, flat_data.shape[1])
        )
        particle_accum = particle_accum.at[point_ids].add(flat_data)

        # reshape
        return particle_accum.reshape(particle_accum.shape[0], *data_shape)
