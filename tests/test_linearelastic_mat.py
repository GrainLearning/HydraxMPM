"""Unit tests for the isotropic linear elastic material module."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""
    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2)

    assert isinstance(material, pm.LinearIsotropicElastic)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)


def test_update_stress_3d():
    """Unit test the isotropic linear elastic material for 3d."""
    particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1, 0.0]]))

    particles = particles.replace(L_stack=jnp.stack([jnp.eye(3) * 0.1]))

    material = pm.LinearIsotropicElastic.create(E=0.1, nu=0.1)

    particles, material = material.update_from_particles(particles, 0.1)
    expected_stress_stack = jnp.array(
        [
            [
                [
                    0.00125,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.00125,
                    0.0,
                ],
                [0.0, 0.0, 0.00125],
            ]
        ]
    )

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack)


def test_update_stress_2d():
    """Unit test the isotropic linear elastic material for 2d."""
    particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1]]))

    particles = particles.replace(L_stack=jnp.stack([jnp.eye(3) * 0.1]))

    material = pm.LinearIsotropicElastic.create(E=0.1, nu=0.1)

    particles, material = material.update_from_particles(particles, 0.1)
    expected_stress_stack = jnp.array(
        [
            [
                [
                    0.00125,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.00125,
                    0.0,
                ],
                [0.0, 0.0, 0.00125],
            ]
        ]
    )

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack, rtol=1e-3)
