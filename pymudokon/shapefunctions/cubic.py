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
    get_m2_lt_m1 = (intr_dist > -2.0) & (intr_dist < -1.0)
    get_m1_lt_0 = (intr_dist >= -1.0) & (intr_dist < 0.0)
    get_0_lt_1 = (intr_dist >= 0.0) & (intr_dist < 1.0)
    get_1_lt_2 = (intr_dist >= 1.0) & (intr_dist < 2.0)

    middle = intr_species == 0
    boundary = intr_species == 3
    boundary_right = intr_species == 2
    boundary_left = intr_species == 1

    # boundary
    basis = jnp.where(
        get_1_lt_2 & boundary, ((-1.0 / 6.0 * intr_dist + 1.0) * intr_dist - 2.0) * intr_dist + 4.0 / 3.0, 0.0
    )
    basis = jnp.where(get_0_lt_1 & boundary, (1.0 / 6.0 * intr_dist * intr_dist - 1.0) * intr_dist + 1.0, basis)
    basis = jnp.where(get_m1_lt_0 & boundary, (-1.0 / 6.0 * intr_dist * intr_dist + 1.0) * intr_dist + 1.0, basis)
    basis = jnp.where(
        get_m2_lt_m1 & boundary, ((1.0 / 6.0 * intr_dist + 1.0) * intr_dist + 2.0) * intr_dist + 4.0 / 3.0, basis
    )
    dbasis = jnp.where(get_1_lt_2 & boundary, inv_node_spacing * ((-0.5 * intr_dist + 2.0) * intr_dist - 2.0), 0.0)
    dbasis = jnp.where(get_0_lt_1 & boundary, inv_node_spacing * (0.5 * intr_dist * intr_dist - 1.0), dbasis)
    dbasis = jnp.where(get_m1_lt_0 & boundary, inv_node_spacing * (-0.5 * intr_dist * intr_dist + 1.0), dbasis)
    dbasis = jnp.where(get_m2_lt_m1 & boundary, inv_node_spacing * ((0.5 * intr_dist + 2.0) * intr_dist + 2.0), dbasis)

    # middle
    basis = jnp.where(
        get_1_lt_2 & middle, ((-1.0 / 6.0 * intr_dist + 1.0) * intr_dist - 2.0) * intr_dist + 4.0 / 3.0, 0.0
    )
    basis = jnp.where(get_0_lt_1 & middle, (0.5 * intr_dist - 1) * intr_dist * intr_dist + 2.0 / 3.0, basis)
    basis = jnp.where(get_m1_lt_0 & middle, (-0.5 * intr_dist - 1) * intr_dist * intr_dist + 2.0 / 3.0, basis)
    basis = jnp.where(
        get_m2_lt_m1 & middle, ((1.0 / 6.0 * intr_dist + 1.0) * intr_dist + 2.0) * intr_dist + 4.0 / 3.0, basis
    )
    dbasis = jnp.where(get_1_lt_2 & middle, inv_node_spacing * ((-0.5 * intr_dist + 2) * intr_dist - 2.0), 0.0)
    dbasis = jnp.where(get_0_lt_1 & middle, inv_node_spacing * (3.0 / 2.0 * intr_dist - 2.0) * intr_dist, dbasis)
    dbasis = jnp.where(get_m1_lt_0 & middle, inv_node_spacing * (-3.0 / 2.0 * intr_dist - 2.0) * intr_dist, dbasis)
    dbasis = jnp.where(get_m2_lt_m1 & middle, inv_node_spacing * ((0.5 * intr_dist + 2.0) * intr_dist + 2.0), dbasis)

    # right side of the boundary 0 + h
    basis = jnp.where(
        get_1_lt_2 & boundary_right, ((-1.0 / 6.0 * intr_dist + 1) * intr_dist - 2) * intr_dist + 4.0 / 3.0, 0.0
    )
    basis = jnp.where(get_0_lt_1 & boundary_right, (0.5 * intr_dist - 1.0) * intr_dist * intr_dist + 2.0 / 3.0, basis)
    basis = jnp.where(
        get_m1_lt_0 & boundary_right, (-1.0 / 3.0 * intr_dist - 1.0) * intr_dist * intr_dist + 2.0 / 3.0, basis
    )
    dbasis = jnp.where(get_1_lt_2 & boundary_right, inv_node_spacing * ((-0.5 * intr_dist + 2) * intr_dist - 2.0), 0.0)
    dbasis = jnp.where(
        get_0_lt_1 & boundary_right, inv_node_spacing * (3.0 / 2.0 * intr_dist - 2.0) * intr_dist, dbasis
    )
    dbasis = jnp.where(get_m1_lt_0 & boundary_right, inv_node_spacing * (-intr_dist - 2) * intr_dist, dbasis)

    # left side of the boundary L - h
    basis = jnp.where(
        get_0_lt_1 & boundary_left, (1.0 / 3.0 * intr_dist - 1.0) * intr_dist * intr_dist + 2.0 / 3.0, 0.0
    )
    basis = jnp.where(get_m1_lt_0 & boundary_left, (-0.5 * intr_dist - 1) * intr_dist * intr_dist + 2.0 / 3.0, basis)
    basis = jnp.where(
        get_m2_lt_m1 & boundary_left, ((1.0 / 6.0 * intr_dist + 1) * intr_dist + 2) * intr_dist + 4.0 / 3.0, basis
    )
    dbasis = jnp.where(get_0_lt_1 & boundary_left, inv_node_spacing * intr_dist * (intr_dist - 2), 0.0)
    dbasis = jnp.where(
        get_m1_lt_0 & boundary_left, inv_node_spacing * (-3.0 / 2.0 * intr_dist - 2.0) * intr_dist, dbasis
    )
    dbasis = jnp.where(
        get_m2_lt_m1 & boundary_left, inv_node_spacing * ((0.5 * intr_dist + 2.0) * intr_dist + 2.0), dbasis
    )

    N0 = basis[0, :]
    N1 = basis[1, :]
    dN0 = dbasis[0, :]
    dN1 = dbasis[1, :]

    shapef = jnp.expand_dims(N0 * N1, axis=1)

    shapef_grad = jnp.array([dN0 * N1, N0 * dN1])

    # shapes of returned array are
    # (num_particles*stencil_size, 1, 1)
    # and (num_particles*stencil_size, dim,1)
    # respectively
    return shapef, shapef_grad


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
            shapef=jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
            shapef_grad=jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
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
        species = jnp.ones(nodes.grid_size).astype(jnp.int32)
        species = species.at[:].set(0)

        # TODO generalize for 3D
        # TODO document
        species = species.at[:, 0].set(1)
        species = species.at[0, :].set(1)
        species = species.at[-1, :].set(1)
        species = species.at[:, -1].set(1)

        # boundary nodes +- 1
        species = species.at[:, 1].set(2)
        species = species.at[1, :].set(2)

        species = species.at[:, -2].set(4)
        species = species.at[-2, :].set(4)

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
        # get node species at state
        intr_species = nodes.species.take(interactions.intr_hashes, axis=0).reshape(-1, 1, 1)

        shapef, shapef_grad = jax.vmap(vmap_cubic_shapefunction, in_axes=(0, 0, None))(
            interactions.intr_dist, intr_species, nodes.inv_node_spacing
        )
        return self.replace(shapef=shapef, shapef_grad=shapef_grad)

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
