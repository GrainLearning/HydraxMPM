# # Copyright (c) 2024, Retiefasuarus
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# # -*- coding: utf-8 -*-
"""
Explanation:
    This module contains the `GridState` class which contains the state of the computational grid.
"""

import jax.numpy as jnp

from jaxtyping import Array, Float, UInt

from typing import Self

import equinox as eqx


class GridState(eqx.Module):
    """
    The grid state in MPM simulations

    Attributes:
        mass_stack: Mass assigned to each grid node
        moment_stack: Momentum (velocity * mass) stored at each
        moment_nt_stack: Momentum at the next time step
        force_stack: Force accumulated at each grid node
        origin: Origin coordinates of the grid
        end: End coordinates of the grid
        cell_size: Size of each grid cell
        _inv_cell_size: Precomputed inverse of cell size for efficiency
        grid_size: Number of grid nodes along each dimension
    """

    mass_stack: Float[Array, "num_nodes"]
    moment_stack: Float[Array, "num_nodes dim"]
    moment_nt_stack: Float[Array, "num_nodes dim"]
    force_stack: Float[Array, "num_nodes dim"]

    # store static info to resolve dependency injection problems
    origin: tuple = eqx.field(static=True)
    end: tuple = eqx.field(static=True)
    cell_size: float | Float[Array, "..."] = eqx.field(static=True)

    _inv_cell_size: float | Float[Array, "..."] = eqx.field(static=True)

    grid_size: tuple = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        origin: Float[Array, "dim"] | tuple,
        end: Float[Array, "dim"] | tuple,
        cell_size: float | Float[Array, "..."],
        **kwargs,
    ) -> Self:
        """
        Creates a GridState with zero-initialized stacks.

        kwargs:
            padding: Optional padding by length of cell_size to extend the grid boundaries. `1` by default.
        """

        # Pad the origin and end
        padded_origin = tuple(o - kwargs.get("padding", 0) * cell_size for o in origin)
        padded_end = tuple(e + kwargs.get("padding", 0) * cell_size for e in end)

        # Update the origin and end with padding
        origin = jnp.array(padded_origin)
        end = jnp.array(padded_end)

        # number of cells in each dimens (M_x,M_y) for 2D, (M_x,M_y,M_z) for 3D
        grid_size = ((jnp.array(end) - jnp.array(origin)) / cell_size + 1).astype(
            jnp.uint32
        )

        _inv_cell_size = 1.0 / cell_size

        dim = len(origin)
        num_cells = jnp.prod(grid_size).astype(jnp.uint32)

        # Create empty array
        mass_stack = jnp.zeros(num_cells)
        moment_stack = jnp.zeros((num_cells, dim))
        moment_nt_stack = jnp.zeros((num_cells, dim))
        force_stack = jnp.zeros((num_cells, dim))

        return GridState(
            mass_stack=mass_stack,
            moment_stack=moment_stack,
            moment_nt_stack=moment_nt_stack,
            force_stack=force_stack,
            origin=tuple(origin.tolist()),
            end=tuple(end.tolist()),
            cell_size=float(cell_size),
            _inv_cell_size=float(_inv_cell_size),
            grid_size=tuple(grid_size.tolist()),
        )

    @property
    def dim(self) -> int:
        """Give dimension of the grid"""
        return len(self.origin)

    @property
    def num_cells(self) -> UInt[Array, ""]:
        """Give total number of cells/ nodes in the grid"""
        return jnp.prod(jnp.array(self.grid_size)).astype(jnp.uint32)

    @property
    def position_mesh(self):
        """Create mesh of node coordinates compatible with C-Contiguous (Row-Major) layout.
        
        (M_x, M_y, M_z, dim) shaped array for 3D, (M_x, M_y, dim) for 2D.
        
        """
        indices = jnp.indices(self.grid_size, dtype=jnp.float32)
        return jnp.moveaxis(indices, 0, -1) * self.cell_size + jnp.array(self.origin)

    @property
    def position_stack(self):
        """Flattened stack of grid node positions"""
        return self.position_mesh.reshape(-1, self.dim)
