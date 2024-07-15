"""State and functions for managing the Material Point Method (MPM) background grid nodes."""
# TODO: Add support for Sparse grid. This feature is currently experimental in JAX.

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self


@chex.dataclass
class Nodes:
    """Background grid nodes of MPM solver.

    Cartesian grid nodes currently supported. To be used with the MPM solver.

    Example:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> origin = jnp.array([0.0, 0.0, 0.0])
    >>> end = jnp.array([1.0, 1.0, 1.0])
    >>> node_spacing = 0.5
    >>> small_mass_cutoff = 1e-10
    >>> nodes = pm.Nodes.create(origin, end, node_spacing, small_mass_cutoff)
    >>> # ... use nodes in MPM solver

    Attributes:
        origin: Start coordinates of domain `(dim,)`.
        end: End coordinates of domain `(dim,)`.
        node_spacing: Spacing between each node in the grid.
        small_mass_cutoff: Cut-off value for small masses to avoid unphysical large velocities, defaults to 1e-12.
        num_nodes_total: Total number of nodes in the grid (derived attribute).
        grid_size: Size of the grid (derived attribute).
        inv_node_spacing: Inverse of the node spacing (derived attribute).
        masses: Nodal masses `(num_nodes_total,)`.
        moments: Nodal moments `(num_nodes_total, dim)`.
        moments_nt: Nodal moments in forward step `(num_nodes_total, dim)`.
        species: Node types. i.e., type of nodes, etc. cubic shape functions
    """

    origin: chex.Array
    end: chex.Array
    node_spacing: jnp.float32
    small_mass_cutoff: jnp.float32
    num_nodes_total: jnp.int32
    grid_size: chex.Array
    inv_node_spacing: jnp.float32

    masses: chex.Array
    moments: chex.Array
    moments_nt: chex.Array
    species: chex.Array

    @classmethod
    def create(
        cls: Self,
        origin: chex.Array,
        end: chex.Array,
        node_spacing: jnp.float32,
        small_mass_cutoff: jnp.float32 = 1e-12,
    ) -> Self:
        """Initialize the state for the background MPM nodes.

        Args:
            cls: Self type reference
            origin: Start coordinates of domain box `(dim,)`.
            end: End coordinates of domain box `(dim,)`.
            node_spacing: Spacing between each node in the grid.
            small_mass_cutoff (optional):
                Small masses threshold to avoid unphysical large velocities, defaults to 1e-10.

        Returns:
            Nodes: Updated node state.

        Example:
            >>> import pymudokon as pm
            >>> origin = jnp.array([0.0, 0.0, 0.0])
            >>> end = jnp.array([1.0, 1.0, 1.0])
            >>> node_spacing = 0.5
            >>> small_mass_cutoff = 1e-10
            >>> nodes = pm.Nodes.create(origin, end, node_spacing, small_mass_cutoff)
        """
        inv_node_spacing = 1.0 / node_spacing

        grid_size = ((end - origin) / node_spacing + 1).astype(jnp.int32)

        num_nodes_total = jnp.prod(grid_size).astype(jnp.int32)

        dim = origin.shape[0]

        return cls(
            origin=origin,
            end=end,
            node_spacing=node_spacing,
            small_mass_cutoff=small_mass_cutoff,
            num_nodes_total=num_nodes_total,
            grid_size=grid_size,
            inv_node_spacing=inv_node_spacing,
            masses=jnp.zeros((num_nodes_total)).astype(jnp.float32),
            moments=jnp.zeros((num_nodes_total, dim)).astype(jnp.float32),
            moments_nt=jnp.zeros((num_nodes_total, dim)).astype(jnp.float32),
            species=jnp.zeros(num_nodes_total).astype(jnp.int32),
        )

    def refresh(self: Self) -> Self:
        """Reset background MPM node states.

        Args:
            self: Nodes state.

        Returns:
            Nodes: Updated node state.
        """
        return self.replace(
            masses=self.masses.at[:].set(0.0),
            moments=self.moments.at[:].set(0.0),
            moments_nt=self.moments_nt.at[:].set(0.0),
        )
