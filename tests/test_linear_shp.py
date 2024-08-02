"""Unit tests for the linear shape functions."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test to test initialization."""
    # 2D linear element stencil size of 4
    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=2)

    assert isinstance(shapefunction, pm.LinearShapeFunction)

    np.testing.assert_allclose(
        shapefunction.intr_shapef_stack, jnp.zeros((8), dtype=jnp.float32)
    )

    np.testing.assert_allclose(
        shapefunction.intr_shapef_grad_stack,
        jnp.zeros((8, 3), dtype=jnp.float32),
    )


def test_calc_shp_2d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21], [0.8, 0.4]])

    origin = jnp.array([0.0, 0.0])
    end = jnp.array([1.0, 1.0])
    node_spacing = 0.1

    grid_size = ((end - origin) / node_spacing + 1).astype(jnp.int32)

    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=2)

    shapefunction, _ = shapefunction.calculate_shapefunction(
        origin=origin,
        inv_node_spacing=10,
        grid_size=grid_size,
        position_stack=position_stack,
    )
    np.testing.assert_allclose(shapefunction.intr_shapef_stack.shape, (8,))

    expected_shapef_stack = jnp.array(
        [0.45000005, 0.04999995, 0.45000005, 0.04999995, 1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(expected_shapef_stack, shapefunction.intr_shapef_stack)
    np.testing.assert_allclose(jnp.prod(shapefunction.intr_shapef_stack, axis=0), 0)

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

    np.testing.assert_allclose(
        expected_shapef_grad_stack, shapefunction.intr_shapef_grad_stack
    )


def test_calc_shp_3d():
    """Test the linear shape function for top level container input."""
    position_stack = jnp.array([[0.45, 0.21, 0.1], [0.8, 0.4, 0.1]])

    origin = jnp.array([0.0, 0.0, 0.0])
    end = jnp.array([1.0, 1.0, 1.0])
    node_spacing = 0.1
    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=3)
    grid_size = ((end - origin) / node_spacing + 1).astype(jnp.int32)
    shapefunction, _ = shapefunction.calculate_shapefunction(
        origin=origin,
        inv_node_spacing=10,
        grid_size=grid_size,
        position_stack=position_stack,
    )
    np.testing.assert_allclose(shapefunction.intr_shapef_stack.shape, (16,))
    expected_shapef_stack = jnp.array(
        [
            0.45,
            0.0,
            0.45,
            0.0,
            0.05,
            0.0,
            0.05,
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

    np.testing.assert_allclose(
        expected_shapef_stack, shapefunction.intr_shapef_stack, rtol=1e-5
    )
    np.testing.assert_allclose(jnp.prod(shapefunction.intr_shapef_stack, axis=0), 0)

    expected_shapef_grad_stack = jnp.array(
        [
            [
                -9.0,
                -5.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                9.0,
                -5.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                -1.0,
                5.0,
                0.0,
            ],
            [
                0.0,
                0.0,
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
        ]
    )

    np.testing.assert_allclose(
        expected_shapef_grad_stack, shapefunction.intr_shapef_grad_stack, rtol=1e-4
    )
