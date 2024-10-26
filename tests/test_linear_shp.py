"""Unit tests for the linear shape functions."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm
import chex


def test_create():
    """Unit test to test initialization."""

    # 2D linear element stencil size of 4
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=2,
        shapefunction_type="linear",
    )
    shapefunction = pm.LinearShapeFunction(config)

    chex.assert_shape(shapefunction.shapef_stack, (2 * 4,))
    chex.assert_shape(shapefunction.shapef_grad_stack, (2 * 4, 3))


def test_calc_shp_2d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21], [0.8, 0.4]])

    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=len(position_stack),
        shapefunction_type="linear",
    )
    grid = pm.GridStencilMap(config)

    grid = grid.partition(position_stack)

    particles = pm.Particles(config, position_stack=position_stack)
    shapefunction = pm.LinearShapeFunction(config)

    shapefunction = shapefunction.get_shapefunctions(grid, particles)

    expected_shapef_stack = jnp.array(
        [
            [0.45000005, 0.04999995, 0.45000005, 0.04999995],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(expected_shapef_stack, shapefunction.shapef_stack)

    np.testing.assert_allclose(
        jnp.prod(shapefunction.shapef_stack.flatten(), axis=0), 0
    )

    expected_shapef_grad_stack = jnp.array(
        [
            [
                [-9.000001, -5.0, 0.0],
                [-0.99999905, 5.0, 0.0],
                [9.000001, -5.0, 0.0],
                [0.99999905, 5.0, 0.0],
            ],
            [[-0.0, -0.0, 0.0], [-0.0, 0.0, 0.0], [0.0, -0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    np.testing.assert_allclose(
        expected_shapef_grad_stack, shapefunction.shapef_grad_stack
    )


def test_calc_shp_3d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21, 0.1], [0.8, 0.4, 0.1]])

    config = pm.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=0.1,
        num_points=len(position_stack),
        shapefunction_type="linear",
    )
    grid = pm.GridStencilMap(config)

    grid = grid.partition(position_stack)

    particles = pm.Particles(config, position_stack=position_stack)
    shapefunction = pm.LinearShapeFunction(config)

    shapefunction = shapefunction.get_shapefunctions(grid, particles)

    expected_shapef_stack = jnp.array(
        [
            [0.45, 0.05, 0.45, 0.05, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        expected_shapef_stack, shapefunction.shapef_stack, rtol=1e-5
    )
    np.testing.assert_allclose(
        jnp.prod(shapefunction.shapef_stack.flatten(), axis=0), 0
    )

    expected_shapef_grad_stack = jnp.array(
        [
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
            ],
            [
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
            ],
        ]
    )
    np.testing.assert_allclose(
        expected_shapef_grad_stack, shapefunction.shapef_grad_stack, rtol=1e-4
    )

