"""Unit tests for the Particles dataclass."""

import unittest

import jax.numpy as jnp
import numpy as np

import jax

import pymudokon as pm


class TestParticles(unittest.TestCase):
    """Unit tests for the Particles dataclass and functions."""

    @staticmethod
    def test_register():
        """Unit test to initialize particles."""
        particles = pm.Particles.create(
            positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            velocities=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
        )
        
        
        jax.lax.scan

        assert isinstance(particles, pm.Particles)

        np.testing.assert_allclose(particles.masses, jnp.zeros(2))

        np.testing.assert_allclose(particles.positions, jnp.array([[0.0, 0.0], [1.0, 1.0]]))

        np.testing.assert_allclose(particles.velocities, jnp.array([[0.0, 0.0], [1.0, 2.0]]))

    @staticmethod
    def test_calculate_volume():
        """Unit test to calculate the volume of the particles.

        Volume calculation is based on the background grid discretization.
        """
        particles = pm.Particles.create(positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        print(particles.positions)

        particles = particles.calculate_volume(node_spacing=0.5, particles_per_cell=1)

        np.testing.assert_allclose(particles.volumes, jnp.array([0.125, 0.125]))
        np.testing.assert_allclose(particles.volumes_original, jnp.array([0.125, 0.125]))

    @staticmethod
    def test_refresh():
        """Unit test to refresh the state of the particles."""
        particles = pm.Particles.create(
            positions=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            velocities=jnp.array([[0.0, 0.0], [1.0, 2.0]]),
        )

        particles = particles.replace(velgrads=jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]))
        particles = particles.refresh()

        np.testing.assert_allclose(particles.velgrads, jnp.zeros((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
