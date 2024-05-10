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
        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=3)

        assert isinstance(material, pm.linearelastic_mat.LinearElasticContainer)
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
        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        vel_grad = jnp.stack([jnp.eye(2), jnp.eye(2)])

        stress, eps_e = jax.vmap(
            pm.linearelastic_mat.vmap_update, in_axes=(0, 0, None, None, None)
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
        pass


if __name__ == "__main__":
    unittest.main()
