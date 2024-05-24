"""State and functions for managing the Material Point Method (MPM) background grid nodes."""

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from .base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Nodes(Base):
    """State for the background MPM nodes.

    Store information of the nodes as grided system.

    Attributes:
        origin (Array): Start coordinates of domain `(dim,)`.
        end (Array): End coordinates of domain `(dim,)`.
        node_spacing (jnp.float32): Spacing between each node in the grid.
        small_mass_cutoff (jnp.float32): Cut-off value for small masses to avoid unphysical large velocities.
        num_nodes_total (jnp.int32): Total number of nodes in the grid (derived attribute).
        grid_size (Array): Size of the grid (derived attribute).
        inv_node_spacing (jnp.float32): Inverse of the node spacing (derived attribute).
        masses (Array): Nodal masses `(num_nodes_total,)`.
        moments (Array): Nodal moments `(num_nodes_total, dim)`.
        moments_nt (Array): Nodal moments in forward step `(num_nodes_total, dim)`.
        species (Array): Node types. i.e., type of nodes, etc. cubic shape functions
    """

    origin: Array
    end: Array
    node_spacing: jnp.float32
    small_mass_cutoff: jnp.float32
    num_nodes_total: jnp.int32
    grid_size: Array
    inv_node_spacing: jnp.float32

    # arrays
    masses: Array
    moments: Array
    moments_nt: Array
    species: Array
    ids_grid: Array

    @classmethod
    def register(
        cls: Self,
        origin: Array,
        end: Array,
        node_spacing: jnp.float32,
        small_mass_cutoff: jnp.float32 = 1e-12,
    ) -> Self:
        """Initialize the state for the background MPM nodes.

        Args:
            cls (Nodes): Self type reference
            origin (Array): Start coordinates of domain `(dim,)`.
            end (Array): End coordinates of domain `(dim,)`.
            node_spacing (jnp.float32): Spacing between each node in the grid.
            small_mass_cutoff (jnp.float32, optional): Small masses threshold to avoid unphysical large velocities,
                defaults to 1e-10.

        Returns:
            Nodes: Updated state for the background MPM nodes.

        Example:
            >>> import pymudokon as pm
            >>> origin = jnp.array([0.0, 0.0, 0.0])
            >>> end = jnp.array([1.0, 1.0, 1.0])
            >>> node_spacing = 0.5
            >>> small_mass_cutoff = 1e-10
            >>> nodes = pm.Nodes.register(origin, end, node_spacing, small_mass_cutoff)
        """
        inv_node_spacing = 1.0 / node_spacing

        grid_size = ((end - origin) / node_spacing + 1).astype(jnp.int32)
        num_nodes_total = jnp.prod(grid_size).astype(jnp.int32)

        _dim = origin.shape[0]

        return cls(
            origin=origin,
            end=end,
            node_spacing=node_spacing,
            small_mass_cutoff=small_mass_cutoff,
            num_nodes_total=num_nodes_total,
            grid_size=grid_size,
            inv_node_spacing=inv_node_spacing,
            masses=jnp.zeros((num_nodes_total)).astype(jnp.float32),
            moments=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32),
            moments_nt=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32),
            species=jnp.zeros(num_nodes_total).astype(jnp.int32),
            ids_grid=jnp.arange(num_nodes_total).reshape(grid_size).astype(jnp.int32),
        )

    @jax.jit
    def refresh(self: Self) -> Self:
        """Refresh the state for the background MPM nodes.

        Args:
            self (Nodes): State for the nodes.

        Returns:
            Nodes: Updated state for the background MPM nodes.

        Example:
            >>> import pymudokon as pm
            >>> # ... initialize nodes
            >>> nodes = nodes.refresh()
        """
        return self.replace(
            masses=self.masses.at[:].set(0.0),
            moments=self.moments.at[:].set(0.0),
            moments_nt=self.moments_nt.at[:].set(0.0),
        )
