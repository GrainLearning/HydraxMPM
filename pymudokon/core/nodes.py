"""State and functions for managing the Material Point Method (MPM) background grid nodes.

The module contains the following main components:

- NodesContainer:
    A JAX pytree (NamedTuple) that stores the state of the MPM nodes.
- init:
    Initialize the state for the background MPM nodes.
- refresh:
    Refresh/reset the state for the background MPM nodes.
"""

from typing import NamedTuple, Union

import jax
import jax.numpy as jnp


class NodesContainer(NamedTuple):
    """State for the background MPM nodes.

    Store information of the background MPM nodes as grided system.

    Attributes:
        origin (Union[jnp.array, jnp.float32]):
            Origin of the grid system.
        end (Union[jnp.array, jnp.float32]):
            End point of the grid system.
        node_spacing (jnp.float32):
            Spacing between each node in the grid.
        particles_per_cell (jnp.int32):
            Number of particles in each cell.
        small_mass_cutoff (jnp.float32):
            Cut-off value for small masses to avoid unphysical large velocities.
        num_nodes_total (jnp.int32):
            Total number of nodes in the grid (derived attribute).
        grid_size (Union[jnp.array, jnp.float32]):
            Size of the grid (derived attribute).
        inv_node_spacing (jnp.float32):
            Inverse of the node spacing (derived attribute).
        masses_array (Union[jnp.array, jnp.float32]):
            Array of the masses of the nodes.
        moments_array (Union[jnp.array, jnp.float32]):
            Array of the moments of the nodes.
        moments_nt_array (Union[jnp.array, jnp.float32]):
            Array of the forward step moments of the nodes.
    """

    origin: Union[jnp.array, jnp.float32]
    end: Union[jnp.array, jnp.float32]
    node_spacing: jnp.float32
    particles_per_cell: jnp.int32
    small_mass_cutoff: jnp.float32
    num_nodes_total: jnp.int32
    grid_size: Union[jnp.array, jnp.float32]
    inv_node_spacing: jnp.float32

    # arrays
    masses_array: Union[jnp.array, jnp.float32]
    moments_array: Union[jnp.array, jnp.float32]
    moments_nt_array: Union[jnp.array, jnp.float32]


def init(
    origin: Union[jnp.array, jnp.float32],
    end: Union[jnp.array, jnp.float32],
    node_spacing: jnp.float32,
    particles_per_cell: jnp.int32,
    small_mass_cutoff: jnp.float32 = 1e-10,
) -> NodesContainer:
    """Initialize the state for the background MPM nodes.

    Args:
        origin (Union[jnp.array, jnp.float32]):
            Origin of the grid system.
            Should be a 1D array with the same dimension as the grid.
        end (Union[jnp.array, jnp.float32]):
            End point of the grid system.
            Should be a 1D array with the same dimension as the grid.
        node_spacing (jnp.float32):
            Spacing between each node in the grid.
        particles_per_cell (jnp.int32):
            Number of particles in each cell.
        small_mass_cutoff (jnp.float32, optional):
            Cut-off value for small masses to avoid unphysical large velocities.
            Defaults to 1e-10.

    Returns:
        NodesContainer:
            Updated state for the background MPM nodes.

    Example:
        >>> import pymudokon as pm
        >>> origin = jnp.array([0.0, 0.0, 0.0])
        >>> end = jnp.array([1.0, 1.0, 1.0])
        >>> node_spacing = 0.5
        >>> particles_per_cell = 2
        >>> small_mass_cutoff = 1e-10
        >>> nodes_state = pm.nodes.init(origin, end, node_spacing, particles_per_cell, small_mass_cutoff)
    """
    # store the inverse for efficiency
    inv_node_spacing = 1.0 / node_spacing

    grid_size = (end - origin) / node_spacing + 1
    num_nodes_total = jnp.prod(grid_size).astype(jnp.int32)

    _dim = origin.shape[0]

    return NodesContainer(
        origin=origin,
        end=end,
        node_spacing=node_spacing,
        particles_per_cell=particles_per_cell,
        small_mass_cutoff=small_mass_cutoff,
        num_nodes_total=num_nodes_total,
        grid_size=grid_size,
        inv_node_spacing=inv_node_spacing,
        masses_array=jnp.zeros((num_nodes_total)).astype(jnp.float32),
        moments_array=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32),
        moments_nt_array=jnp.zeros((num_nodes_total, _dim)).astype(jnp.float32)
    )


@jax.jit
def refresh(nodes_state: NodesContainer) -> NodesContainer:
    """Refresh the state for the background MPM nodes.

    Refresh (or reset state) is called internally before each time step to reset the masses
    and moments of the nodes (e.g. in :func:`~usl.update`).

    Args:
        nodes_state (NodesContainer): State for the background MPM nodes.

    Returns:
        NodesContainer: Updated state for the background MPM nodes.

    Example:
        >>> import pymudokon as pm
        >>> # ... initialize nodes_state
        >>> nodes_state = pm.nodes.refresh(nodes_state)
    """
    return nodes_state._replace(
        masses_array=nodes_state.masses_array.at[:].set(0.0),
        moments_array=nodes_state.moments_array.at[:].set(0.0),
        moments_nt_array=nodes_state.moments_nt_array.at[:].set(0.0),
    )
