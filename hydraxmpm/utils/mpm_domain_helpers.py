"""Helper functions to discretize the domain."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles


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


def generate_mesh(config: MPMConfig):
    x = jnp.linspace(config.origin[0], config.end[0], config.grid_size[0])
    y = jnp.linspace(config.origin[1], config.end[1], config.grid_size[1])

    if config.dim == 3:
        z = jnp.linspace(config.origin[2], config.end[2], config.grid_size[2])
        X, Y, Z = jnp.meshgrid(x, y, z)
        return jnp.array([X, Y, Z]).T
    else:
        X, Y = jnp.meshgrid(x, y)
        return jnp.array([X, Y]).T


def fill_domain_with_particles(config: MPMConfig, thickness=3, ppc =4):
    """Fill the background grid with 2x2 (or 2x2x2 in 3D) particles.
    Args:
        nodes (Nodes): Nodes class
    """

    node_mesh = generate_mesh(config)

    if config.dim == 2:
        node_mesh = node_mesh.at[thickness:, :].get()
        node_mesh = node_mesh.at[:, thickness:].get()
        node_mesh = node_mesh.at[: -1 - thickness, :].get()
        node_mesh = node_mesh.at[:, : -1 - thickness].get()
        pnt_opt = jnp.array(
            [[0.2113, 0.2113], [0.2113, 0.7887], [0.7887, 0.2113], [0.7887, 0.7887]]
        )
    else:
        node_mesh = node_mesh.at[thickness:, :, :].get()
        node_mesh = node_mesh.at[: -1 - thickness, :, :].get()
        node_mesh = node_mesh.at[:, thickness:, :].get()
        node_mesh = node_mesh.at[:, : -1 - thickness, :].get()
        node_mesh = node_mesh.at[:, :, thickness:].get()
        node_mesh = node_mesh.at[:, :, : -1 - thickness].get()
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
        if ppc ==1:
            pnt_opt = jnp.array([[0.5, 0.5, 0.5]])

        
        
    node_coords_stack = node_mesh.reshape(-1, config.dim)

    def get_opt(node_coords, pnt_opt):
        return pnt_opt * config.cell_size + node_coords

    pnt_stack = jax.vmap(get_opt, in_axes=(0, None))(
        node_coords_stack, pnt_opt
    ).reshape(-1, config.dim)
    return pnt_stack, node_coords_stack
