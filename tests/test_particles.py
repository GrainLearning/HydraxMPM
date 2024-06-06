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
        positions=jnp.zeros((num_particles, dim)),
    )
    assert particles.positions.shape == (num_particles, dim)
    assert particles.velocities.shape == (num_particles, dim)
    assert particles.masses.shape == (num_particles,)
    assert particles.species.shape == (num_particles,)
    assert particles.volumes.shape == (num_particles,)
    assert particles.volumes_original.shape == (num_particles,)
    assert particles.velgrads.shape == (num_particles, dim, dim)
    assert particles.stresses.shape == (num_particles, 3, 3)
    assert particles.forces.shape == (num_particles, dim)
    assert particles.F.shape == (num_particles, dim, dim)
    assert particles.ids.shape == (num_particles,)
    assert particles.positions.dtype == jnp.float32
    assert particles.velocities.dtype == jnp.float32
    assert particles.masses.dtype == jnp.float32
    assert particles.species.dtype == jnp.int32
    assert particles.volumes.dtype == jnp.float32
    assert particles.volumes_original.dtype == jnp.float32
    assert particles.velgrads.dtype == jnp.float32
    assert particles.stresses.dtype == jnp.float32
    assert particles.forces.dtype == jnp.float32
    assert particles.F.dtype == jnp.float32
    assert particles.ids.dtype == jnp.int32
    assert particles.ids[0] != particles.ids[1]


def test_calculate_volume():
    """Unit test to calculate the volume of the particles.

    Volume calculation is based on the background grid discretization.
    """
    particles = pm.Particles.create(positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    particles = particles.calculate_volume(node_spacing=0.5, particles_per_cell=1)

    np.testing.assert_allclose(particles.volumes, jnp.array([0.125, 0.125]))
    np.testing.assert_allclose(particles.volumes_original, jnp.array([0.125, 0.125]))


def test_refresh():
    """Unit test to refresh the state of the particles."""
    particles = pm.Particles.create(
        positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
        velocities=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
    )

    particles = particles.replace(velgrads=jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]))
    particles = particles.refresh()

    np.testing.assert_allclose(particles.velgrads, jnp.zeros((2, 2, 2)))
