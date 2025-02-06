import copy

import equinox as eqx
import jax.numpy as jnp
from typing_extensions import Self

from ..common.types import (
    TypeFloat,
    TypeFloat3,
    TypeFloatScalarNStack,
    TypeFloatVectorAStack,
    TypeFloatVectorNStack,
)


class Grid(eqx.Module):
    """Background grid of the MPM simulation.

    Attributes:
        origin: start point of the domain box
        end: end point of the domain box
        cell_size: cell size of the background grid
        mass_stack: Mass assigned to each grid node
            (shape: `(num_nodes)`).
        moment_stack: Momentum (velocity * mass) stored at each
            grid node (shape: `(num_nodes, dim)`).
        moment_nt_stack (jnp.ndarray): Momentum at the next time step, used for
            integration schemes like FLIP (shape: `(num_nodes, dim)`).
        normal_stack (jnp.ndarray): Normal vectors associated with each node
            (shape: `(num_nodes, dim)`).
            This might represent surface normals if the grid represents a boundary.
        small_mass_cutoff (float): Threshold for small mass values.
            Nodes with mass below this cutoff may be treated specially to
            avoid numerical instabilities.
    """

    origin: tuple = eqx.field(static=True)
    end: tuple = eqx.field(static=True)
    cell_size: float = eqx.field(static=True)
    num_cells: int = eqx.field(init=False, static=True, converter=lambda x: int(x))
    grid_size: tuple = eqx.field(init=False, static=True)
    dim: int = eqx.field(static=True, init=False)

    small_mass_cutoff: float = eqx.field(static=True, converter=lambda x: float(x))

    mass_stack: TypeFloatScalarNStack
    moment_stack: TypeFloatVectorNStack
    moment_nt_stack: TypeFloatVectorNStack
    normal_stack: TypeFloatVectorNStack

    _is_padded: bool = eqx.field(static=True)
    _inv_cell_size: float = eqx.field(init=False, static=True)

    def __init__(
        self,
        origin: TypeFloat3 | tuple,
        end: TypeFloat3 | tuple,
        cell_size: TypeFloat,
        small_mass_cutoff: TypeFloat = 1e-10,
        **kwargs,
    ) -> Self:
        self.origin = jnp.array(origin)
        self.end = jnp.array(end)
        self.cell_size = cell_size
        self.dim = len(origin)

        self._inv_cell_size = 1.0 / self.cell_size

        # requires jnp.array for calculations
        self.grid_size = (
            (jnp.array(self.end) - jnp.array(self.origin)) / self.cell_size + 1
        ).astype(jnp.uint32)

        self.num_cells = jnp.prod(self.grid_size).astype(jnp.uint32)

        # convert to tuple after calculation
        self.grid_size = tuple(self.grid_size.tolist())
        self.origin = tuple(self.origin.tolist())
        self.end = tuple(self.end.tolist())

        self.mass_stack = jnp.zeros(self.num_cells)
        self.moment_stack = jnp.zeros((self.num_cells, self.dim))

        self.moment_nt_stack = jnp.zeros((self.num_cells, self.dim))

        self.normal_stack = jnp.zeros((self.num_cells, self.dim))

        self.small_mass_cutoff = small_mass_cutoff

        # flag if the outside domain is padded
        self._is_padded = kwargs.get("_is_padded", False)

        # super().__init__(**kwargs)

    def refresh(self: Self) -> Self:
        """Reset background MPM node states."""

        return eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
            ),
            self,
            (
                jnp.zeros_like(self.mass),
                jnp.zeros_like(self.momentum),
                jnp.zeros_like(self.momentum_next),
            ),
        )

    def init_padding(self, shapefunction) -> Self:
        # pad outside of the domain
        if self._is_padded:
            return self
        else:
            if shapefunction == "linear":
                pad = 1
            elif shapefunction == "cubic":
                pad = 2

            new_origin = (
                jnp.array(self.origin) - jnp.ones(self.dim) * self.cell_size * pad
            )
            new_end = jnp.array(self.end) + jnp.ones(self.dim) * self.cell_size * pad

            # returns a copy of the object with the new domain
            return copy.copy(
                Grid(
                    new_origin,
                    new_end,
                    self.cell_size,
                    self.small_mass_cutoff,
                    _is_padded=True,
                )
            )

    @property
    def position_mesh(self) -> TypeFloatVectorAStack:
        x = jnp.linspace(self.origin[0], self.end[0], self.grid_size[0])
        y = jnp.linspace(self.origin[1], self.end[1], self.grid_size[1])

        if self.dim == 3:
            z = jnp.linspace(self.origin[2], self.end[2], self.grid_size[2])
            X, Y, Z = jnp.meshgrid(x, y, z)
            return jnp.array([X, Y, Z]).T
        else:
            X, Y = jnp.meshgrid(x, y)
            return jnp.array([X, Y]).T

    @property
    def position_stack(self) -> TypeFloatVectorNStack:
        return self.position_mesh.reshape(-1, self.dim)
