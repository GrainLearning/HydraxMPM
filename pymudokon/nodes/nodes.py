"""State and functions for the background MPM grid nodes."""
# TODO: Add support for Sparse grid. This feature is currently experimental in JAX.

from typing_extensions import Self

import chex
import jax.numpy as jnp


@chex.dataclass
class Nodes:
    """Background grid nodes of MPM solver.

    Cartesian grid nodes currently supported. To be used with the MPM solver.

    Attributes:
        origin: Start coordinates of domain `(dim,)`.
        end: End coordinates of domain `(dim,)`.
        node_spacing: Spacing between each node in the grid.
        small_mass_cutoff: Cut-off value for small masses to avoid unphysical
            large velocities,
        defaults to 1e-12.
        num_nodes_total: Total number of nodes in the grid (derived attribute).
        grid_size: Size of the grid (derived attribute).
        inv_node_spacing: Inverse of the node spacing (derived attribute).
        mass_stack: Nodal masses `(num_nodes_total,)`.
        moment_stack: Nodal moments `(num_nodes_total, dim)`.
        moment_nt_stack: Nodal moments in forward step `(num_nodes_total, dim)`.
        species_stack: Node types. i.e., type of nodes, etc. cubic shape functions

    Example:
    >>> import pymudokon as pm
    >>> import jax.numpy as jnp
    >>> origin = jnp.array([0.0, 0.0, 0.0])
    >>> end = jnp.array([1.0, 1.0, 1.0])
    >>> node_spacing = 0.5
    >>> small_mass_cutoff = 1e-10
    >>> nodes = pm.Nodes.create(origin, end, node_spacing, small_mass_cutoff)
    >>> # ... use nodes in MPM solver
    """

    origin: chex.Array
    end: chex.Array
    node_spacing: jnp.float32
    small_mass_cutoff: jnp.float32
    num_nodes_total: jnp.int32
    grid_size: chex.Array
    inv_node_spacing: jnp.float32

    mass_stack: chex.Array
    moment_stack: chex.Array
    moment_nt_stack: chex.Array
    species_stack: chex.Array

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
                Small masses threshold to avoid unphysical large velocities,
                defaults to 1e-10.

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
            mass_stack=jnp.zeros((num_nodes_total)).astype(jnp.float32),
            moment_stack=jnp.zeros((num_nodes_total, dim)).astype(jnp.float32),
            moment_nt_stack=jnp.zeros((num_nodes_total, dim)).astype(jnp.float32),
            species_stack=jnp.zeros(num_nodes_total).astype(jnp.int32),
        )

    def refresh(self: Self) -> Self:
        """Reset background MPM node states.

        Args:
            self: Nodes state.

        Returns:
            Nodes: Updated node state.
        """
        return self.replace(
            mass_stack=self.mass_stack.at[:].set(0.0),
            moment_stack=self.moment_stack.at[:].set(0.0),
            moment_nt_stack=self.moment_nt_stack.at[:].set(0.0),
        )
