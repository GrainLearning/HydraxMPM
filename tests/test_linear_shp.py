"""Unit tests for the linear shape functions."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test to test initialization."""
    # 2D linear element stencil size of 4
    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=2)

    assert isinstance(shapefunction, pm.LinearShapeFunction)

    np.testing.assert_allclose(shapefunction.intr_shapef, jnp.zeros((8), dtype=jnp.float32))

    np.testing.assert_allclose(
        shapefunction.intr_shapef_grad,
        jnp.zeros((8, 3), dtype=jnp.float32),
    )


def test_calc_shp_2d():
    """Test the linear shape function for top level container input."""
    positions = jnp.array([[0.45, 0.21], [0.8, 0.4]])

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.1)

    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=2)

    shapefunction, _ = shapefunction.calculate_shapefunction(nodes, positions)
    np.testing.assert_allclose(shapefunction.intr_shapef.shape, (8,))

    expected_shapef = jnp.array([0.45000005, 0.04999995, 0.45000005, 0.04999995, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(expected_shapef, shapefunction.intr_shapef)
    np.testing.assert_allclose(jnp.prod(shapefunction.intr_shapef, axis=0), 0)

    expected_shapef_grad = jnp.array(
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

    np.testing.assert_allclose(expected_shapef_grad, shapefunction.intr_shapef_grad)


def test_calc_shp_3d():
    """Test the linear shape function for top level container input."""
    positions = jnp.array([[0.45, 0.21, 0.1], [0.8, 0.4, 0.1]])

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0, 1.0]), end=jnp.array([1.0, 1.0, 1.0]), node_spacing=0.1)

    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=3)

    shapefunction, _ = shapefunction.calculate_shapefunction(nodes, positions)
    np.testing.assert_allclose(shapefunction.intr_shapef.shape, (16,))
    expected_shapef = jnp.array([0.45, 0.0, 0.45, 0.0, 0.05, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(expected_shapef, shapefunction.intr_shapef, rtol=1e-5)
    np.testing.assert_allclose(jnp.prod(shapefunction.intr_shapef, axis=0), 0)

    expected_shapef_grad = jnp.array(
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

    np.testing.assert_allclose(expected_shapef_grad, shapefunction.intr_shapef_grad, rtol=1e-4)


test_calc_shp_2d()
