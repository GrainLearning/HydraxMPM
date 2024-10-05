"""State and functions for the background MPM grid nodes."""
# TODO: Add support for Sparse grid. This feature is currently experimental in JAX.

from typing_extensions import Self

import chex
import jax.numpy as jnp
import jax

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

        species_stack = jnp.zeros(num_nodes_total).reshape(grid_size).astype(jnp.int16)
        
        if dim ==2:
            # boundary layers
            species_stack = species_stack.at[0, :].set(1) #x0
            species_stack = species_stack.at[:, 0].set(1) #y0
            species_stack = species_stack.at[grid_size[0]-1, :].set(1) #x1
            species_stack = species_stack.at[:,grid_size[1] -1].set(1) #y0
            
            # boundary layers 0 + h
            species_stack = species_stack.at[1, :].set(2) #x0
            species_stack = species_stack.at[:, 1].set(2) #y0
            
            # boundary layer N-h
            species_stack = species_stack.at[grid_size[0]-2, :].set(3) #x1
            species_stack = species_stack.at[:,grid_size[1] -2].set(3) #y0

        species_stack = species_stack.reshape(-1)
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
            species_stack=species_stack,
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
    
    def get_coordinate_stack(self,dim=3):
        
        nx, ny = self.grid_size

        x = jnp.linspace(self.origin[0], self.end[0], nx)

        y = jnp.linspace(self.origin[1], self.end[1], ny)

        xv, yv = jnp.meshgrid(x, y)


        node_coordinate_stack = jnp.array(list(zip(xv.flatten(), yv.flatten()))).astype(jnp.float32)

        # is there a way to avoid this? i.e., generate sorted nodes
        node_hash_stack = self.get_hash_stack(node_coordinate_stack,dim)
        sorted_id_stack = jnp.argsort(node_hash_stack)
        node_coordinate_stack = node_coordinate_stack.at[sorted_id_stack].get()
        return node_coordinate_stack
        
        
    def get_hash_stack(self,position_stack: chex.Array, dim: int=3):
        
        def calculate_hash(position):
            rel_pos = (position - self.origin)*self.inv_node_spacing

            if dim == 2:
                return (rel_pos[1] + rel_pos[0] * self.grid_size[1]).astype(
                    jnp.int32
                )
            else:
                return (
                    rel_pos[0]
                    + rel_pos[1] * self.grid_size[0]
                    + rel_pos[2] * self.grid_size[0] * self.grid_size[1]
                ).astype(jnp.int32)
            
                
        hash_stack = jax.vmap(calculate_hash)(position_stack)

        return hash_stack
