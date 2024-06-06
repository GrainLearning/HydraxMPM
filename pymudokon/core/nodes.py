"""State and functions for managing the Material Point Method (MPM) background grid nodes."""

import jax
from flax import struct
from typing_extensions import Self


@struct.dataclass
class Nodes:
    """State for the background MPM nodes.

    Store information of the nodes as grided system.

    Attributes:
        origin (jax.Array): Start coordinates of domain `(dim,)`.
        end (jax.Array): End coordinates of domain `(dim,)`.
        node_spacing (jax.numpy.float32): Spacing between each node in the grid.
        small_mass_cutoff (jax.numpy.float32): Cut-off value for small masses to avoid unphysical large velocities.
        num_nodes_total (jax.numpy.int32): Total number of nodes in the grid (derived attribute).
        grid_size (jax.Array): Size of the grid (derived attribute).
        inv_node_spacing (jax.numpy.float32): Inverse of the node spacing (derived attribute).
        masses (jax.Array): Nodal masses `(num_nodes_total,)`.
        moments (jax.Array): Nodal moments `(num_nodes_total, dim)`.
        moments_nt (jax.Array): Nodal moments in forward step `(num_nodes_total, dim)`.
        species (jax.Array): Node types. i.e., type of nodes, etc. cubic shape functions
    """

    origin: jax.Array
    end: jax.Array
    node_spacing: jax.numpy.float32
    small_mass_cutoff: jax.numpy.float32
    num_nodes_total: jax.numpy.int32
    grid_size: jax.Array
    inv_node_spacing: jax.numpy.float32

    # arrays
    masses: jax.Array
    moments: jax.Array
    moments_nt: jax.Array
    species: jax.Array
    ids_grid: jax.Array

    @classmethod
    def create(
        cls: Self,
        origin: jax.Array,
        end: jax.Array,
        node_spacing: jax.numpy.float32,
        small_mass_cutoff: jax.numpy.float32 = 1e-12,
    ) -> Self:
        """Initialize the state for the background MPM nodes.

        Args:
            cls (Nodes): Self type reference
            origin (jax.Array): Start coordinates of domain `(dim,)`.
            end (jax.Array): End coordinates of domain `(dim,)`.
            node_spacing (jax.numpy.float32): Spacing between each node in the grid.
            small_mass_cutoff (jax.numpy.float32, optional):
                Small masses threshold to avoid unphysical large velocities, defaults to 1e-10.

        Returns:
            Nodes: Updated state for the background MPM nodes.

        Example:
            >>> import pymudokon as pm
            >>> origin = jax.numpy.array([0.0, 0.0, 0.0])
            >>> end = jax.numpy.array([1.0, 1.0, 1.0])
            >>> node_spacing = 0.5
            >>> small_mass_cutoff = 1e-10
            >>> nodes = pm.Nodes.create(origin, end, node_spacing, small_mass_cutoff)
        """
        inv_node_spacing = 1.0 / node_spacing

        grid_size = ((end - origin) / node_spacing + 1).astype(jax.numpy.int32)
        num_nodes_total = jax.numpy.prod(grid_size).astype(jax.numpy.int32)

        dim = origin.shape[0]

        return cls(
            origin=origin,
            end=end,
            node_spacing=node_spacing,
            small_mass_cutoff=small_mass_cutoff,
            num_nodes_total=num_nodes_total,
            grid_size=grid_size,
            inv_node_spacing=inv_node_spacing,
            masses=jax.numpy.zeros((num_nodes_total)).astype(jax.numpy.float32),
            moments=jax.numpy.zeros((num_nodes_total, dim)).astype(jax.numpy.float32),
            moments_nt=jax.numpy.zeros((num_nodes_total, dim)).astype(jax.numpy.float32),
            species=jax.numpy.zeros(num_nodes_total).astype(jax.numpy.int32),
            ids_grid=jax.numpy.arange(num_nodes_total).reshape(grid_size).astype(jax.numpy.int32),
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
