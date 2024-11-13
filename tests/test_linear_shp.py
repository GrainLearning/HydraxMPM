"""Unit tests for the linear shape functions."""

import chex
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_calc_shp_2d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21], [0.8, 0.4]])

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=len(position_stack),
        shapefunction="linear",
    )

    grid = hdx.Grid(config)

    grid = grid.get_interactions(position_stack)

    expected_shapef_stack = jnp.array(
        [0.45000005, 0.04999995, 0.45000005, 0.04999995, 1.0, 0.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(expected_shapef_stack, grid.intr_shapef_stack)

    np.testing.assert_allclose(jnp.prod(grid.intr_shapef_stack.flatten(), axis=0), 0)

    expected_shapef_grad_stack = jnp.array(
        [
            [-9.000001, -5.0, 0.0],
            [-0.99999905, 5.0, 0.0],
            [9.000001, -5.0, 0.0],
            [0.99999905, 5.0, 0.0],
            [-0.0, -0.0, 0.0],
            [-0.0, 0.0, 0.0],
            [0.0, -0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(expected_shapef_grad_stack, grid.intr_shapef_grad_stack)


def test_calc_shp_3d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21, 0.1], [0.8, 0.4, 0.1]])

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=0.1,
        num_points=len(position_stack),
        shapefunction="linear",
    )

    grid = hdx.Grid(config)

    grid = grid.get_interactions(position_stack)

    expected_shapef_stack = jnp.array(
        [
            0.45,
            0.05,
            0.45,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(expected_shapef_stack, grid.intr_shapef_stack, rtol=1e-5)
    np.testing.assert_allclose(jnp.prod(grid.intr_shapef_stack.flatten(), axis=0), 0)

    expected_shapef_grad_stack = jnp.array(
        [
            [
                -9.0,
                -5.0,
                0.0,
            ],
            [
                -1.0,
                5.0,
                0.0,
            ],
            [
                9.0,
                -5.0,
                0.0,
            ],
            [
                1.0,
                5.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    np.testing.assert_allclose(
        expected_shapef_grad_stack, grid.intr_shapef_grad_stack, rtol=1e-4
    )
