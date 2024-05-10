"""Unit tests for the ParticlesContainer class.

Test and examples on how to use the ParticlesContainer class to to setup/update particle state


The module contains the following main components:

- TestParticles.test_init:
    Unit test to initialize the ParticlesContainer class.
- TestParticles.test_calculate_volume:
    Unit test for volume calculation of the particles.
- TestParticles.test_refresh:
    Unit test for resetting variables of the ParticlesContainer state.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestParticles(unittest.TestCase):
    """Unit tests for the ParticlesContainer and functions."""

    @staticmethod
    def test_init():
        """Unit test to initialize the ParticlesContainer class."""
        particles_state = pm.core.particles.init(
            positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            velocities=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
        )

        assert isinstance(particles_state, pm.core.particles.ParticlesContainer)

        np.testing.assert_allclose(particles_state.masses_array, jnp.zeros(2))

        np.testing.assert_allclose(
            particles_state.positions_array, jnp.array([[0.0, 0.0], [1.0, 1.0]])
        )

        np.testing.assert_allclose(
            particles_state.velocities_array, jnp.array([[0.0, 0.0], [1.0, 2.0]])
        )

    @staticmethod
    def test_calculate_volume():
        """Unit test to calculate the volume of the particles.

        Volume calculation is based on the background grid discretization.
        """
        particles_state = pm.core.particles.init(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )

        particles_state = pm.core.particles.calculate_volume(
            particles_state, node_spacing=0.5, particles_per_cell=1
        )

        np.testing.assert_allclose(particles_state.volumes_array, jnp.array([0.125, 0.125]))
        np.testing.assert_allclose(
            particles_state.volumes_original_array, jnp.array([0.125, 0.125])
        )

    @staticmethod
    def test_refresh():
        """Unit test to refresh the state of the particles."""
        particles_state = pm.particles.init(
            positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            velocities=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
        )

        particles_state = particles_state._replace(
            velgrad_array=jnp.array(
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
            )
        )
        particles_state = pm.particles.refresh(particles_state)

        np.testing.assert_allclose(particles_state.velgrad_array, jnp.zeros((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
