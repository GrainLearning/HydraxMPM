"""Module for containing the cubic shape functions.

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.'
    - [Gaussian Quadrature - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_quadrature)
"""

# TODO add test for cubic shape function
# TODO add docstring for cubic shape function
import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from ..core.interactions import Interactions
from ..core.nodes import Nodes
from .shapefunction import ShapeFunction


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
    basis = jnp.select(conditions, basis_functions)
    dbasis = jnp.select(conditions, dbasis_functions)
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
    basis = jnp.select(conditions, basis_functions)
    dbasis = jnp.select(conditions, dbasis_functions)
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
    basis = jnp.select(conditions, basis_functions)
    dbasis = jnp.select(conditions, dbasis_functions)
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
    basis = jnp.select(conditions, basis_functions)
    dbasis = jnp.select(conditions, dbasis_functions)
    return basis, dbasis


def vmap_cubic_shapefunction(
    intr_dist: Array,
    intr_species: Array,
    inv_node_spacing: jnp.float32,
) -> Tuple[Array, Array]:
    """Vectorized cubic shape function calculation.

    Calculate the shape function, and then its gradients.

    Args:
        intr_dist (Array):
            Particle-node pair interactions distance.
        intr_species (Array):
            Node type of the background grid. See
            :meth:`pymudokon.core.nodes.Nodes.set_species` for details.
        inv_node_spacing (jnp.float32):
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
    return basis, dbasis


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class CubicShapeFunction(ShapeFunction):
    """Cubic B-spline shape functions for the particle-node interactions."""

    @classmethod
    def register(cls: Self, num_particles: jnp.int32, dim: jnp.int16) -> Self:
        """Initializes Cubic B-splines.

        It is recommended that each background cell is populated by
        2 (1D), 4 (2D), 8 (3D) material points. The optimal integration points are
        at 0.2113, 0.7887 determined by Gauss quadrature rule.

        Args:
            cls (Self):
                self type reference
            num_particles (jnp.int32):
                Number of particles
            dim (jnp.int16):
                Dimension of the problem

        Returns:
            ShapeFunction:
                Container for shape functions and gradients
        """
        if dim == 1:
            stencil = jnp.array([[-1], [0], [1], [2]])
        if dim == 2:
            stencil = jnp.array(
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
            stencil = jnp.array(
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

        return cls(
            shapef=jnp.zeros((num_particles * stencil_size, 1, 1), dtype=jnp.float32),
            shapef_grad=jnp.zeros((num_particles * stencil_size, dim, 1), dtype=jnp.float32),
            stencil=stencil,
        )

    def set_nodes_species(self: Self, nodes: Nodes) -> Nodes:
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
            >>> nodes = pm.Nodes.register(grid_size=4, node_spacing=0.5)
            >>> shapefunctions = pm.CubicShapeFunction.register(2, 2)
            >>> nodes = shapefunctions.set_node_species(nodes)
        """

        # middle nodes
        species = jnp.zeros(nodes.grid_size).astype(jnp.int32)

        # # # TODO generalize for 3D
        # # # TODO document

        # # boundary nodes 0 + h
        species = species.at[1, 1:-1].set(1)
        species = species.at[1:-1, 1].set(1)

        # # boundary nodes L - h
        species = species.at[1:-1, -2].set(2)
        species = species.at[-2, 1:-1].set(2)

        # # boundary nodes
        species = species.at[0, :].set(3)  # xstart
        species = species.at[-1, :].set(3)  # xend
        species = species.at[:, 0].set(3)  # ystart
        species = species.at[:, -1].set(3)  # yend

        species = species.reshape(-1)
        return nodes.replace(species=species)

    @jax.jit
    def calculate_shapefunction(
        self: Self,
        nodes: Nodes,
        interactions: Interactions,
    ) -> Self:
        """Top level function to calculate the shape functions.

        Args:
            self (CubicShapeFunction):
                Shape function at previous state
            nodes (Nodes):
                Nodes state containing grid size and inv_node_spacing
            interactions (Interactions):
                Interactions state containing particle-node interactions' distances.

        Returns:
            CubicShapeFunction:
                Updated shape function state for the particle and node pairs.
        """
        dim = self.stencil.shape[1]

        # repeat for each dimension
        intr_species = nodes.species.take(interactions.intr_hashes, axis=0).reshape(-1).repeat(dim)

        basis, dbasis = jax.vmap(vmap_cubic_shapefunction, in_axes=(0, 0, None))(
            interactions.intr_dist.reshape(-1), intr_species, nodes.inv_node_spacing
        )
        basis = basis.reshape(-1, dim)
        dbasis = dbasis.reshape(-1, dim)

        N0 = basis[:, 0]
        N1 = basis[:, 1]
        dN0 = dbasis[:, 0]
        dN1 = dbasis[:, 1]

        shapef = (N0 * N1).reshape(-1, 1, 1)

        shapef_grad = jnp.array([dN0 * N1, N0 * dN1]).T.reshape(-1, dim, 1)

        return self.replace(
            shapef=shapef,
            shapef_grad=shapef_grad,
        )

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
