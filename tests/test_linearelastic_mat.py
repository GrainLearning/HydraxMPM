"""Unit tests for the isotropic linear elastic material module.

Test and examples on how to use the isotropic linear elastic material
module to setup/update material state

The module contains the following main components:

- TestLinearElastic.test_init:
    Unit test to initialize the isotropic linear elastic material.
- TestLinearElastic.test_vmap_update:
    Unit test for vectorized update of the isotropic linear elastic material.
- TestLinearElastic.test_update_stress:
    Unit test for updating stress and strain for all particles.
- TestLinearElastic.test_solve:
    Unit test for solving the linear elastic material.

"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestLinearElastic(unittest.TestCase):
    """Unit tests for the isotropic linear elastic material module."""

    @staticmethod
    def test_init():
        """Test the initialization of the isotropic linear elastic material."""
        material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.2, num_particles=2, dim=3)

        assert isinstance(material, pm.LinearIsotropicElastic)
        np.testing.assert_allclose(material.E, 1000.0)
        np.testing.assert_allclose(material.nu, 0.2)
        np.testing.assert_allclose(material.G, 416.666667)
        np.testing.assert_allclose(material.K, 555.5555555555557)
        np.testing.assert_allclose(
            material.eps_e, jnp.zeros((2, 3, 3), dtype=jnp.float32)
        )

    @staticmethod
    def test_vmap_update():
        """Test the vectorized update of the isotropic linear elastic material."""
        material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.2, num_particles=2, dim=2)

        vel_grad = jnp.stack([jnp.eye(2), jnp.eye(2)])

        stress, eps_e = jax.vmap(
            pm.material.linearelastic_mat.vmap_update, in_axes=(0, 0, None, None, None), out_axes=(0, 0)
        )(material.eps_e, vel_grad, material.G, material.K, 0.001)

        np.testing.assert_allclose(
            stress,
            jnp.array(
                [
                    [
                        [1.1111112, 0.0, 0.0],
                        [0.0, 1.1111112, 0.0],
                        [0.0, 0.0, 1.1111112],
                    ],
                    [
                        [1.1111112, 0.0, 0.0],
                        [0.0, 1.1111112, 0.0],
                        [0.0, 0.0, 1.1111112],
                    ],
                ]
            ),
        )
        np.testing.assert_allclose(
            eps_e,
            jnp.array(
                [
                    [[0.001, 0.0], [0.0, 0.001]],
                    [[0.001, 0.0], [0.0, 0.001]],
                ]
            ),
        )

    @staticmethod
    def test_update_stress():
        """Test the update of stress and strain for all particles."""
        particles = pm.Particles.register(
            positions=jnp.array([[0.0, 0.0], [1.0, 1.0]])
        )

        particles = particles.replace(
            velgrads = jnp.stack([jnp.eye(2), jnp.eye(2)])
        )

        material = pm.LinearIsotropicElastic.register(
            E=1000.0, nu=0.2, num_particles=2, dim=2
            )

        particle, material = material.update_stress(
            particles, 0.001
        )

        np.testing.assert_allclose(
            particle.stresses,
            jnp.array(
                [
                    [
                        [1.1111112, 0.0, 0.0],
                        [0.0, 1.1111112, 0.0],
                        [0.0, 0.0, 1.1111112],
                    ],
                    [
                        [1.1111112, 0.0, 0.0],
                        [0.0, 1.1111112, 0.0],
                        [0.0, 0.0, 1.1111112],
                    ],
                ]
            ),
        )
        np.testing.assert_allclose(
            material.eps_e,
            jnp.array(
                [
                    [[0.001, 0.0], [0.0, 0.001]],
                    [[0.001, 0.0], [0.0, 0.001]],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
