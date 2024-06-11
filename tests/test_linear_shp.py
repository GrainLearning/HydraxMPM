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
        jnp.zeros((8, 2), dtype=jnp.float32),
    )


def test_calculate_shapefunction():
    """Test the linear shape function for top level container input."""
    particles = pm.Particles.create(positions=jnp.array([[0.25, 0.25], [0.8, 0.4]]))

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

    shapefunction = pm.LinearShapeFunction.create(num_particles=2, dim=2)

    shapefunction = shapefunction.calculate_shapefunction(nodes, particles)

    np.testing.assert_allclose(shapefunction.intr_shapef.shape, (8,))

    np.testing.assert_allclose(
        shapefunction.intr_shapef,
        jnp.array([0.25, 0.25, 0.25, 0.25, 0.08, 0.12, 0.32, 0.48]),
    )

    np.testing.assert_allclose(shapefunction.intr_shapef_grad.shape, (8, 2))

    np.testing.assert_allclose(
        shapefunction.intr_shapef_grad,
        jnp.array(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-0.4, -0.8], [0.4, -1.2], [-1.6, 0.8], [1.6, 1.2]]
        ).reshape((8, 2)),
    )
