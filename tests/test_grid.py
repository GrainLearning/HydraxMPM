"""Unit tests for the grid"""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


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

    grid = grid.get_interactions(position_stack)

    print(grid.intr_dist_stack)
    np.testing.assert_allclose(
        grid.intr_dist_stack,  # normalized at grid
        jnp.array(
            [
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.3, -0.4, -0.0],
                [-0.3, 0.1, -0.0],
                [0.2, -0.4, -0.0],
                [0.2, 0.1, -0.0],
            ]
        ),
    )

    np.testing.assert_allclose(grid.intr_hash_stack.shape, (12))
    np.testing.assert_allclose(
        grid.intr_hash_stack, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 3, 4, 6, 7])
    )
