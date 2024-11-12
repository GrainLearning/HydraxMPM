"""Unit tests for the grid"""

import jax.numpy as jnp
import numpy as np
import pytest

import hydraxmpm as hdx
import equinox as eqx


def test_create():
    """Unit test to create grid nodes over multiple dimensions."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=1,
    )
    grid = hdx.Grid(config)

    assert isinstance(grid, hdx.Grid)

    assert grid.config.num_cells == 121


def test_get_interactions():
    position_stack = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.5,
        num_points=position_stack.shape[0],
    )
    grid = hdx.Grid(config)
    import jax
  
    @jax.jit
    def get_interactions_from_grid(grid):
        return grid.get_interactions(position_stack)  

    # exp = jax.make_jaxpr(get_interactions_from_grid)(grid)
    grid = get_interactions_from_grid(grid)
    np.testing.assert_allclose(
        grid.intr_dist_stack,
        jnp.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0],
                [0.6, 0.8, 0.0],
                [0.6, -0.2, 0.0],
                [-0.4, 0.8, 0.0],
                [-0.4, -0.2, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(grid.intr_hash_stack.shape, (12))
    np.testing.assert_allclose(
        grid.intr_hash_stack, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 3, 4, 6, 7])
    )
    print(grid.intr_shapef_stack)
