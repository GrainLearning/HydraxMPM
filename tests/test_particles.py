"""Unit tests for the Particles dataclass."""

import jax.numpy as jnp
import numpy as np
import pytest

import pymudokon as pm


@pytest.mark.parametrize(
    "dim, exp_shape_scalars, exp_shape_vectors, exp_shape_tensors",
    [
        (1, (2,), (2,), (2, 1)),
        (2, (2,), (2, 2), (2, 2, 2)),
        (3, (2,), (2, 3), (2, 3, 3)),
    ],
)
def test_create(dim, exp_shape_scalars, exp_shape_vectors, exp_shape_tensors):
    """Unit test to initialize particles over multiple dimensions with 2 particles."""
    num_particles = 2
    particles = pm.Particles.create(
        position_stack=jnp.zeros((num_particles, dim)),
    )
    assert particles.position_stack.shape == (num_particles, dim)
    assert particles.velocity_stack.shape == (num_particles, dim)


def test_calculate_volume():
    """Unit test to calculate the volume of the particles.

    Volume calculation is based on the background grid discretization.
    """
    particles = pm.Particles.create(
        position_stack=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )

    particles = particles.calculate_volume(node_spacing=0.5, particles_per_cell=1)

    np.testing.assert_allclose(particles.volume_stack, jnp.array([0.125, 0.125]))
    np.testing.assert_allclose(particles.volume0_stack, jnp.array([0.125, 0.125]))


def test_refresh():
    """Unit test to refresh the state of the particles."""
    particles = pm.Particles.create(
        position_stack=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
        velocity_stack=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
    )

    particles = particles.replace(
        L_stack=jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    )
    particles = particles.refresh()

    np.testing.assert_allclose(particles.L_stack, jnp.zeros((2, 2, 2)))
