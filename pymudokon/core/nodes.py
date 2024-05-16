"""State and functions for managing the Material Point Method (MPM) background grid nodes."""

# TODO add test for node species
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
        origin (Array):
            Origin of the grid system `(dim,)`.
        end (Array):
            End point of the grid system `(dim,)`.
        node_spacing (jnp.float32):
            Spacing between each node in the grid.
        particles_per_cell (jnp.int32):
            Number of particles in each grid cell.
        small_mass_cutoff (jnp.float32):
            Cut-off value for small masses to avoid unphysical large velocities.
        num_nodes_total (jnp.int32):
            Total number of nodes in the grid (derived attribute).
        grid_size (Array):
            Size of the grid (derived attribute).
        inv_node_spacing (jnp.float32):
            Inverse of the node spacing (derived attribute).
        masses (Array):
            Array of the masses of the nodes.
        moments (Array):
            Array of the moments of the nodes.
        moments_nt (Array):
            Array of the forward step moments of the nodes.
        species (Array):
            Node types.
            e.g, for cubic shape functions there are 4 possibilities:
            1 is boundary, 2 is left side boundary + 1, 3 middle boundary, 4 right side boundary + 1.
    """

    origin: Array
    end: Array
    node_spacing: jnp.float32
    particles_per_cell: jnp.int32
    small_mass_cutoff: jnp.float32
    num_nodes_total: jnp.int32
    grid_size: Array
    inv_node_spacing: jnp.float32

    # arrays
    masses: Array
    moments: Array
    moments_nt: Array
    species: Array

    @classmethod
    def register(
        cls: Self,
        origin: Array,
        end: Array,
        node_spacing: jnp.float32,
        particles_per_cell: jnp.int32,
        small_mass_cutoff: jnp.float32 = 1e-10,
    ) -> Self:
        """Initialize the state for the background MPM nodes.

        Args:
            cls (Nodes):
                self type reference
            origin (Array):
                Origin of the grid system `(dim,)`.
            end (Array):
                End point of the grid system `(dim,)`.
            node_spacing (jnp.float32):
                Spacing between each node in the grid.
            particles_per_cell (jnp.int32):
                Number of particles in each cell.
            small_mass_cutoff (jnp.float32, optional):
                Cut-off value for small masses to avoid unphysical large velocities, defaults to 1e-10.

        Returns:
            Nodes: Updated state for the background MPM nodes.

        Example:
            >>> import pymudokon as pm
            >>> origin = jnp.array([0.0, 0.0, 0.0])
            >>> end = jnp.array([1.0, 1.0, 1.0])
            >>> node_spacing = 0.5
            >>> particles_per_cell = 2
            >>> small_mass_cutoff = 1e-10
            >>> nodes = pm.Nodes.register(origin, end, node_spacing, particles_per_cell, small_mass_cutoff)
        """
        inv_node_spacing = 1.0 / node_spacing

        grid_size = ((end - origin) / node_spacing + 1).astype(jnp.int32)
        num_nodes_total = jnp.prod(grid_size).astype(jnp.int32)

        _dim = origin.shape[0]

        return cls(
            origin=origin,
            end=end,
            node_spacing=node_spacing,
            particles_per_cell=particles_per_cell,
            small_mass_cutoff=small_mass_cutoff,
            num_nodes_total=num_nodes_total,
            grid_size=grid_size,
            inv_node_spacing=inv_node_spacing,
            masses=jnp.zeros((num_nodes_total)).astype(jnp.float32),
            moments=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32),
            moments_nt=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32),
            species=jnp.zeros(num_nodes_total).astype(jnp.int32),
        )

    @jax.jit
    def refresh(self: Self) -> Self:
        """Refresh the state for the background MPM nodes.

        Args:
            self (Nodes):
                State for the nodes.

        Returns:
            Nodes:
                Updated state for the background MPM nodes.

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
