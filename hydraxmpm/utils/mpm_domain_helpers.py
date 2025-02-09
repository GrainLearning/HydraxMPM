"""Helper functions to discretize the domain."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


from ..grid.grid import Grid


def fill_domain_with_points(grid: Grid, thickness=3, ppc=4, dim=3):
    """Fill the background grid with 2x2 (or 2x2x2 in 3D) particles.
    Args:
        nodes (Nodes): Nodes class
    """

    if dim == 2:
        pnt_opt = jnp.array(
            [[0.2113, 0.2113], [0.2113, 0.7887], [0.7887, 0.2113], [0.7887, 0.7887]]
        )
    else:
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
        if ppc == 1:
            pnt_opt = jnp.array([[0.5, 0.5, 0.5]])

    def get_opt(grid_coords, pnt_opt):
        return pnt_opt * grid.cell_size + grid_coords

    pnt_stack = jax.vmap(get_opt, in_axes=(0, None))(
        grid.position_stack, pnt_opt
    ).reshape(-1, dim)
    return pnt_stack, grid.position_stack
