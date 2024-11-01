"""Helper functions to discretize the domain."""

from typing import Tuple

import jax
import jax.numpy as jnp

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..nodes.nodes import Nodes
from ..config.mpm_config import MPMConfig

import equinox as eqx


def discretize(
    config: MPMConfig,
    particles: Particles,
    nodes: Nodes,
    density_ref: float = 1000,
) -> Tuple[Particles, Nodes]:
    """Discretize the domain.

    Args:
        particles (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunction (Interactions): Shape functions in the simulation.
        ppc (int, optional): Particles per cell. Defaults to 2.
        density_ref (float, optional): Reference density. Defaults to 1000.

    Returns:
        Tuple[Particles, Nodes, ShapeFunction]: Discretized particles,
        nodes and shapefunctions.
    """

    new_volume_stack = particles.calculate_volume()

    new_mass_stack = density_ref * new_volume_stack

    new_particles = eqx.tree_at(
        lambda state: (state.mass_stack, state.volume_stack, state.volume0_stack),
        particles,
        (new_mass_stack, new_volume_stack, new_volume_stack),
    )

    return new_particles, nodes


def fill_domain_with_particles(nodes: Nodes, dim=3):
    """Fill the background grid with 2x2 (or 2x2x2 in 3D) particles.

    Args:
        nodes (Nodes): Nodes class
    """

    node_coordinate_stack = nodes.get_coordinate_stack(dim=dim)

    node_coords = node_coordinate_stack.reshape(*nodes.grid_size, dim)

    if dim == 2:
        node_coords = node_coords.at[3:, :].get()
        node_coords = node_coords.at[:, 3:].get()
        node_coords = node_coords.at[:-4, :].get()
        node_coords = node_coords.at[:, :-4].get()

        pnt_opt = jnp.array(
            [[0.2113, 0.2113], [0.2113, 0.7887], [0.7887, 0.2113], [0.7887, 0.7887]]
        )
    else:
        node_coords = node_coords.at[3:, :, :].get()
        node_coords = node_coords.at[:-4, :, :].get()
        node_coords = node_coords.at[:, 3:, :].get()
        node_coords = node_coords.at[:, :-4, :].get()
        node_coords = node_coords.at[:, :, 3:].get()
        node_coords = node_coords.at[:, :, :-4].get()
        pnt_opt = jnp.array(
            [
                [0.2113, 0.2113, 0.2113],
                [0.2113, 0.7887, 0.2113],
                [0.7887, 0.2113, 0.2113],
                [0.7887, 0.7887, 0.2113],
                [0.2113, 0.2113, 0.7887],
                [0.2113, 0.7887, 0.7887],
                [0.7887, 0.2113, 0.7887],
                [0.7887, 0.7887, 0.7887],
            ]
        )
    node_coords = node_coords.reshape(-1, dim)

    def get_opt(node_coords, pnt_opt):
        return pnt_opt * nodes.node_spacing + node_coords

    pnt_stack = jax.vmap(get_opt, in_axes=(0, None))(node_coords, pnt_opt).reshape(
        -1, dim
    )
    return pnt_stack, node_coordinate_stack
