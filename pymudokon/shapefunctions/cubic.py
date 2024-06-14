"""Module for containing the cubic shape functions.

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.'
    - [Gaussian Quadrature - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_quadrature)
"""

# TODO add test for cubic shape function
# TODO add docstring for cubic shape function

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from .shapefunction import ShapeFunction


@struct.dataclass
class CubicShapeFunction(ShapeFunction):
    """Cubic B-spline shape functions for the particle-node interactions."""

    @classmethod
    def create(cls: Self, num_particles: jax.numpy.int32, dim: jax.numpy.int16) -> Self:
        """Initializes Cubic B-splines.

        It is recommended that each background cell is populated by
        2 (1D), 4 (2D), 8 (3D) material points. The optimal integration points are
        at 0.2113, 0.7887 determined by Gauss quadrature rule.

        Args:
            cls (Self):
                self type reference
            num_particles (jax.numpy.int32):
                Number of particles
            dim (jax.numpy.int16):
                Dimension of the problem

        Returns:
            ShapeFunction:
                Container for shape functions and gradients
        """
        if dim == 1:
            stencil = jax.numpy.array([[-1], [0], [1], [2]])
        if dim == 2:
            stencil = jax.numpy.array(
                [
                    [-1, -1],
                    [0, -1],
                    [1, -1],
                    [2, -1],
                    [-1, 0],
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [-1, 1],
                    [0, 1],
                    [1, 1],
                    [2, 1],
                    [-1, 2],
                    [0, 2],
                    [1, 2],
                    [2, 2],
                ]
            )

        if dim == 3:
            stencil = jax.numpy.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 0],
                    [-1, -1, 1],
                    [-1, -1, 2],
                    [0, -1, -1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [0, -1, 2],
                    [1, -1, -1],
                    [1, -1, 0],
                    [1, -1, 1],
                    [1, -1, 2],
                    [2, -1, -1],
                    [2, -1, 0],
                    [2, -1, 1],
                    [2, -1, 2],
                    [-1, 0, -1],
                    [-1, 0, 0],
                    [-1, 0, 1],
                    [-1, 0, 2],
                    [0, 0, -1],
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 0, -1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 2],
                    [2, 0, -1],
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [-1, 1, -1],
                    [-1, 1, 0],
                    [-1, 1, 1],
                    [-1, 1, 2],
                    [0, 1, -1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 2],
                    [1, 1, -1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [2, 1, -1],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [-1, 2, -1],
                    [-1, 2, 0],
                    [-1, 2, 1],
                    [-1, 2, 2],
                    [0, 2, -1],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [1, 2, -1],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [2, 2, -1],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                ]
            )

        stencil_size = stencil.shape[0]

        intr_ids = jnp.arange(num_particles * stencil_size).astype(jnp.int32)

        return cls(
            intr_ids=intr_ids,
            intr_hashes=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
            intr_shapef=jnp.zeros((num_particles * stencil_size), dtype=jnp.float32),
            intr_shapef_grad=jax.numpy.zeros((num_particles * stencil_size, dim), dtype=jnp.float32),
            stencil=stencil,
        )

    def set_boundary_nodes(self: Self, nodes: Nodes) -> Nodes:
        """Set the node species to ensure connectivity is correct.

        The background grid is divided into 4 regions. Each region is stored within
        the `nodes.species` array. Each region corresponds to different elements.
        Boundary nodes are open to ensure Dirichlet boundary conditions.

        The regions are:
        - Boundary nodes (flag 3)
        - Middle nodes (flag 0)
        - Right boundary nodes (flag 2)
        - Left boundary nodes (flag 1)

        Args:
            self (Self):
                self reference
            nodes (Nodes):
                Background grid nodes

        Returns:
            Nodes:
                Updated shape function state

        Example:
            >>> import pymudokon as pm
            >>> nodes = pm.Nodes.create(grid_size=4, node_spacing=0.5)
            >>> shapefunctions = pm.CubicShapeFunction.create(2, 2)
            >>> nodes = shapefunctions.set_node_species(nodes)
        """
        # TODO generalize
        # middle nodes
        species = jax.numpy.zeros(nodes.grid_size).astype(jax.numpy.int32)

        # # # boundary nodes 0 + h
        # species = species.at[1, 1:-1].set(1)
        # species = species.at[1:-1, 1].set(1)

        # # # boundary nodes L - h
        # species = species.at[1:-1, -2].set(2)
        # species = species.at[-2, 1:-1].set(2)

        # # # boundary nodes
        # species = species.at[0, :].set(3)  # xstart
        # species = species.at[-1, :].set(3)  # xend
        # species = species.at[:, 0].set(3)  # ystart
        # species = species.at[:, -1].set(3)  # yend

        species = species.reshape(-1)
        return nodes.replace(species=species)

    @jax.jit
    def calculate_shapefunction(
        self: Self,
        nodes: Nodes,
        positions: Array,
    ) -> Self:
        """Top level function to calculate the shape functions.

        Args:
            self (CubicShapeFunction):
                Shape function at previous state
            nodes (Nodes):
                Nodes state containing grid size and inv_node_spacing
        Returns:
            CubicShapeFunction:
                Updated shape function state for the particle and node pairs.
        """
        dim = self.stencil.shape[1]

        # repeat for each dimension
        intr_species = nodes.species.take(self.intr_hashes, axis=0).reshape(-1)

        # 1. Calculate the particle-node pair interactions
        # see `ShapeFunction class` for more details
        intr_dist, intr_hashes = self.vmap_intr(
            self.intr_ids, positions, nodes.origin, nodes.inv_node_spacing, nodes.grid_size, dim
        )

        intr_shapef, intr_shapef_grad = self.vmap_intr_shp(intr_dist, intr_species, nodes.inv_node_spacing)
        return self.replace(intr_shapef=intr_shapef, intr_shapef_grad=intr_shapef_grad, intr_hashes=intr_hashes)
     
    def validate(self: Self, solver) -> Self:
        """Verify the shape functions and gradients.

        Args:
            self (Self):
                Self reference
            solver (BaseSolver):
                Solver class
        Returns:
            Self:
                Updated solver class
        """
        # TODO check if 2,4,
        raise NotImplementedError("This method is not yet implemented.")

    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def vmap_intr_shp(
        self,
        intr_dist: Array,
        intr_species: Array,
        inv_node_spacing: jax.numpy.float32,
    ) -> Tuple[Array, Array]:
        """Vectorized cubic shape function calculation.

        Calculate the shape function, and then its gradients.

        Args:
            intr_dist (Array):
                Particle-node pair interactions distance.
            intr_species (Array):
                Node type of the background grid. See
                :meth:`pymudokon.core.nodes.Nodes.set_species` for details.
            inv_node_spacing (jax.numpy.float32):
                Inverse node spacing.

        Returns:
            Tuple[Array, Array]:
                Shape function and its gradient.
        """
        spline_branches = [
            middle_splines,  # species 0
            boundary_padding_start_splines,  # species 1
            boundary_padding_end_splines,  # species 2
            boundary_splines,  # species 3
        ]

        basis, dbasis = jax.lax.switch(intr_species, spline_branches, (intr_dist, inv_node_spacing))
        intr_shapef = jnp.prod(basis)
        # intr_shapef_grad = dbasis * jnp.roll(basis, shift=-1)
        dim = basis.shape[0]
        if dim ==2:
            intr_shapef_grad = jnp.array(
                [
                dbasis[0]*basis[1],
                dbasis[1]*basis[0],
                ]
            )
        elif dim ==3:
            intr_shapef_grad = jnp.array(
                [
                dbasis[0]*basis[1]*basis[2],
                dbasis[1]*basis[0]*basis[2],
                dbasis[2]*basis[0]*basis[1]]
            )
        else:
            intr_shapef_grad = dbasis

        return intr_shapef, intr_shapef_grad


def middle_splines(package) -> Tuple[Array, Array]:
    intr_dist, inv_node_spacing = package
    conditions = [
        (intr_dist >= 1.0) & (intr_dist < 2.0),
        (intr_dist >= 0.0) & (intr_dist < 1.0),
        (intr_dist >= -1.0) & (intr_dist < 0.0),
        (intr_dist >= -2.0) & (intr_dist < -1.0),
    ]
    # Arrays are evaluated for each condition, is there a better way to do this?
    basis_functions = [
        (lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0)(intr_dist),
        (lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0)(intr_dist),
        (lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0)(intr_dist),
        (lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0)(intr_dist),
    ]

    dbasis_functions = [
        (lambda x, h: h * ((-0.5 * x + 2) * x - 2.0))(intr_dist, inv_node_spacing),
        (lambda x, h: h * (3.0 / 2.0 * x - 2.0) * x)(intr_dist, inv_node_spacing),
        (lambda x, h: h * (-3.0 / 2.0 * x - 2.0) * x)(intr_dist, inv_node_spacing),
        (lambda x, h: h * ((0.5 * x + 2) * x + 2.0))(intr_dist, inv_node_spacing),
    ]
    basis = jax.numpy.select(conditions, basis_functions)
    dbasis = jax.numpy.select(conditions, dbasis_functions)
    return basis, dbasis


def boundary_padding_end_splines(package) -> Tuple[Array, Array]:
    #     #     # left side of the boundary L - h

    intr_dist, inv_node_spacing = package
    conditions = [
        (intr_dist >= 0.0) & (intr_dist < 1.0),
        (intr_dist >= -1.0) & (intr_dist < 0.0),
        (intr_dist >= -2.0) & (intr_dist < -1.0),
    ]

    basis_functions = [
        (lambda x: (1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0)(intr_dist),
        (lambda x: (-0.5 * x - 1) * x * x + 2.0 / 3.0)(intr_dist),
        (lambda x: ((1.0 / 6.0 * x + 1) * x + 2) * x + 4.0 / 3.0)(intr_dist),
    ]

    dbasis_functions = [
        (lambda x, h: h * x * (x - 2))(intr_dist, inv_node_spacing),
        (lambda x, h: h * (-3.0 / 2.0 * x - 2.0) * x)(intr_dist, inv_node_spacing),
        (lambda x, h: h * ((0.5 * x + 2.0) * x + 2.0))(intr_dist, inv_node_spacing),
    ]
    basis = jax.numpy.select(conditions, basis_functions)
    dbasis = jax.numpy.select(conditions, dbasis_functions)
    return basis, dbasis


def boundary_padding_start_splines(package) -> Tuple[Array, Array]:
    intr_dist, inv_node_spacing = package
    conditions = [
        (intr_dist >= 1.0) & (intr_dist < 2.0),
        (intr_dist >= 0.0) & (intr_dist < 1.0),
        (intr_dist >= -1.0) & (intr_dist < 0.0),
    ]

    basis_functions = [
        (lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0)(intr_dist),
        (lambda x: (0.5 * x - 1) * x * x + 2.0 / 3.0)(intr_dist),
        (lambda x: (-1.0 / 3.0 * x - 1.0) * x * x + 2.0 / 3.0)(intr_dist),
    ]

    dbasis_functions = [
        (lambda x, h: h * ((-0.5 * x + 2) * x - 2.0))(intr_dist, inv_node_spacing),
        (lambda x, h: h * (3.0 / 2.0 * x - 2.0) * x)(intr_dist, inv_node_spacing),
        (lambda x, h: h * (-x - 2) * x)(intr_dist, inv_node_spacing),
    ]
    basis = jax.numpy.select(conditions, basis_functions)
    dbasis = jax.numpy.select(conditions, dbasis_functions)
    return basis, dbasis


def boundary_splines(package) -> Tuple[Array, Array]:
    intr_dist, inv_node_spacing = package
    conditions = [
        (intr_dist >= 1.0) & (intr_dist < 2.0),
        (intr_dist >= 0.0) & (intr_dist < 1.0),
        (intr_dist >= -1.0) & (intr_dist < 0.0),
        (intr_dist >= -2.0) & (intr_dist < -1.0),
    ]

    basis_functions = [
        (lambda x: ((-1.0 / 6.0 * x + 1.0) * x - 2.0) * x + 4.0 / 3.0)(intr_dist),
        (lambda x: (1.0 / 6.0 * x * x - 1.0) * x + 1.0)(intr_dist),
        (lambda x: (-1.0 / 6.0 * x * x + 1.0) * x + 1.0)(intr_dist),
        (lambda x: ((1.0 / 6.0 * x + 1.0) * x + 2.0) * x + 4.0 / 3.0)(intr_dist),
    ]

    dbasis_functions = [
        (lambda x, h: h * ((-0.5 * x + 2) * x - 2.0))(intr_dist, inv_node_spacing),
        (lambda x, h: h * (0.5 * x * x - 1.0))(intr_dist, inv_node_spacing),
        (lambda x, h: h * (-0.5 * x * x + 1.0))(intr_dist, inv_node_spacing),
        (lambda x, h: h * ((0.5 * x + 2) * x + 2.0))(intr_dist, inv_node_spacing),
    ]
    basis = jax.numpy.select(conditions, basis_functions)
    dbasis = jax.numpy.select(conditions, dbasis_functions)
    return basis, dbasis
