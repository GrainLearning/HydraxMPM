"""Unit tests for the Particles dataclass."""

import jax.numpy as jnp
import numpy as np
import pytest
import chex

import pymudokon as pm

def test_create():
    """Unit test to initialize particles over 2 particles."""
    num_particles = 2
    particles = pm.Particles(
        position_stack=jnp.zeros((num_particles, 2)),
    )
    chex.assert_shape(particles.position_stack,(num_particles,2))

    config = pm.MPMConfig(
        origin=[0.0,0.0],
        end=[1.,1.0,],
        cell_size = 0.1,
        num_points =1
    )

    particles = pm.Particles(
        mpm_config = config,
        position_stack=jnp.zeros((num_particles, 2)),
    )
    chex.assert_shape(particles.position_stack,(num_particles,2))


# def test_calculate_volume():
#     """Unit test to calculate the volume of the particles.

#     Volume calculation is based on the background grid discretization.
#     """
#     particles = pm.Particles.create(
#         position_stack=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
#     )

#     particles = particles.calculate_volume(node_spacing=0.5, particles_per_cell=1)

#     np.testing.assert_allclose(particles.volume_stack, jnp.array([0.125, 0.125]))
#     np.testing.assert_allclose(particles.volume0_stack, jnp.array([0.125, 0.125]))


# def test_refresh():
#     """Unit test to refresh the state of the particles."""
#     particles = pm.Particles.create(
#         position_stack=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
#         velocity_stack=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
#     )

#     particles = particles.replace(
#         L_stack=jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
#     )
#     particles = particles.refresh()

#     np.testing.assert_allclose(particles.L_stack, jnp.zeros((2, 2, 2)))
test_create()