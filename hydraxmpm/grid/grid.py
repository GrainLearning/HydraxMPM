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

class GridDomain(eqx.Module):
    """
    Defines static grid topology or domain information.

    Attributes:
        origin: Origin coordinates of the grid
        end: End coordinates of the grid
        cell_size: Size of each grid cell
        grid_size: Number of grid nodes along each dimension

    """
    origin: tuple = eqx.field(static=True)
    end: tuple = eqx.field(static=True)
    cell_size: float | Float[Array, "..."] = eqx.field(static=True)
    grid_size: tuple = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    periodic_axes: tuple = eqx.field(static=True)

    _inv_cell_size: float | Float[Array, "..."] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        origin: Float[Array, "dim"] | tuple,
        end: Float[Array, "dim"] | tuple,
        cell_size: float | Float[Array, "..."],
        padding: int = 2,
        periodic_axes: tuple[bool, ...] | None = None,
    ) -> Self:
        """
        Creates a GridState with zero-initialized stacks.

        kwargs:
            padding: Optional padding by length of cell_size to extend the grid boundaries. `1` by default.
        """

        if not isinstance(origin, tuple):
            origin = tuple(origin.tolist())
        if not isinstance(end, tuple):
            end = tuple(end.tolist())

        dim = len(origin)
        if periodic_axes is None:
            periodic_axes = (False,) * dim
        else:
            periodic_axes = tuple(bool(axis) for axis in periodic_axes)
            if len(periodic_axes) != dim:
                raise ValueError(
                    "periodic_axes must have one boolean entry per grid dimension"
                )

        # Periodic axes contain only the unique physical nodes. Non-periodic axes
        # retain the existing ghost-node padding.
        padded_origin = tuple(
            o if is_periodic else o - padding * cell_size
            for o, is_periodic in zip(origin, periodic_axes)
        )
        padded_end = tuple(
            e if is_periodic else e + padding * cell_size
            for e, is_periodic in zip(end, periodic_axes)
        )

        # number of cells in each dimens (M_x,M_y) for 2D, (M_x,M_y,M_z) for 3D
        raw_size = [(e - o) / cell_size for e, o in zip(padded_end, padded_origin)]
        grid_size = tuple(
            int(round(s)) if is_periodic else int(round(s)) + 1
            for s, is_periodic in zip(raw_size, periodic_axes)
        )


        return cls(
            origin=padded_origin,
            end=padded_end,
            cell_size=float(cell_size),
            _inv_cell_size=1.0 / float(cell_size),
            grid_size=grid_size,
            padding=padding,
            periodic_axes=periodic_axes,
        )


    @property
    def dim(self) -> int:
        """Give dimension of the grid"""
        return len(self.origin)

    @property
    def num_cells(self) -> int:
        """Total number of nodes in the grid."""
        n = 1
        for s in self.grid_size:
            n *= s
        return n

    @property
    def position_mesh(self):
        """Create mesh of node coordinates compatible with C-Contiguous (Row-Major) layout.

        (M_x, M_y, M_z, dim) shaped array for 3D, (M_x, M_y, dim) for 2D.

        """
        indices = jnp.indices(self.grid_size, dtype=jnp.float32)
        # Move the dimension axis to the last position
        return jnp.moveaxis(indices, 0, -1) * self.cell_size + jnp.array(self.origin)

    @property
    def position_stack(self):
        """Flattened stack of grid node positions"""
        return self.position_mesh.reshape(-1, self.dim)


class GridArrays(eqx.Module):
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

    @classmethod
    def new(cls, grid_topology: GridDomain) -> Self:
        """
        Allocates fresh zero-filled buffers based on the GridDomain topology.
        """
        num = grid_topology.num_cells
        dim = grid_topology.dim

        return cls(
            mass_stack=jnp.zeros((num,)),
            moment_stack=jnp.zeros((num, dim)),
            moment_nt_stack=jnp.zeros((num, dim)),
            force_stack=jnp.zeros((num, dim))
        )

    @property
    def dim(self) -> int:
        """Dimension of the grid"""
        return self.moment_stack.shape[1]
