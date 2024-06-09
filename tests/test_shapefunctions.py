"""Unit tests for the Interactions state."""

import jax.numpy as jnp

import pymudokon as pm
import numpy as np


def test_create():
    """Unit test to check the creation of the Interactions object."""
    stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    stencil_size, dim = stencil.shape
    num_particles = 2
    shapefunction = pm.ShapeFunction(
        intr_hashes=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        intr_shapef=jnp.zeros((num_particles * stencil_size), dtype=jnp.float32),
        intr_ids=jnp.arange(num_particles * stencil_size).astype(jnp.int32),
        intr_shapef_grad=jnp.zeros((num_particles * stencil_size, dim), dtype=jnp.float32),
        stencil=stencil,
    )
    assert isinstance(shapefunction, pm.ShapeFunction)


def test_vmap_intr():
    """Unit test for vectorized particle-node interaction mapping."""
    positions = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

    origin = jnp.array([0.0, 0.0])

    inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

    grid_size = jnp.array([3, 3])

    stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    num_particles = positions.shape[0]
    stencil_size, dim = stencil.shape

    intr_ids = jnp.arange(num_particles * stencil_size)

    shapefunction = pm.ShapeFunction(
        jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
        jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
        stencil=stencil,
        intr_ids=intr_ids,
    )
    intr_dist, intr_hashes = shapefunction.vmap_intr(intr_ids, positions, origin, inv_node_spacing, grid_size, dim)
    np.testing.assert_allclose(
        intr_dist,
        jnp.array(
            [
                [
                    0.5,
                    0.5,
                ],
                [
                    -0.5,
                    0.5,
                ],
                [
                    0.5,
                    -0.5,
                ],
                [
                    -0.5,
                    -0.5,
                ],
                [
                    0.5,
                    0.5,
                ],
                [
                    -0.5,
                    0.5,
                ],
                [
                    0.5,
                    -0.5,
                ],
                [
                    -0.5,
                    -0.5,
                ],
                [
                    0.6,
                    0.8,
                ],
                [
                    -0.4,
                    0.8,
                ],
                [
                    0.6,
                    -0.2,
                ],
                [-0.4, -0.2],
            ]
        ),
    )
    np.testing.assert_allclose(intr_hashes.shape, (12))
    np.testing.assert_allclose(intr_hashes, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 1, 2, 4, 5]))
