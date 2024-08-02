"""Unit tests for the mu I rheology module."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test the initialization drucker prager material."""

    material = pm.MuI.create(
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        phi_c=0.648,
        I_phi=0.5,
        rho_p=2000,
        d=0.0053,
    )

    assert isinstance(material, pm.MuI)

    np.testing.assert_allclose(material.mu_s, 0.38)
    np.testing.assert_allclose(material.mu_d, 0.64)
    np.testing.assert_allclose(material.I_0, 0.279)
    np.testing.assert_allclose(material.I_phi, 0.5)
    np.testing.assert_allclose(material.phi_c, 0.648)
    np.testing.assert_allclose(material.rho_p, 2000)
    np.testing.assert_allclose(material.d, 0.0053)


def test_update_stress_3d():
    """Unit test the isotropic linear elastic material for 3d."""
    particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1, 0.0]]))

    particles = particles.replace(
        volume_stack=jnp.array([0.019]),
        volume0_stack=jnp.array([0.2]),
        stress_stack=jnp.array([[[0.0, 0.0, 0.0]]]),
        L_stack=jnp.stack([jnp.eye(3) * 0.1]),
    )

    material = pm.MuI.create(
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        phi_c=0.648,
        I_phi=0.5,
        rho_p=2000,
        d=0.0053,
    )

    particles, material = material.update_from_particles(particles, 0.1)

    expected_stress_stack = jnp.array(
        [
            [
                [-3.79382e-18, 0.00000e00, 0.00000e00],
                [0.00000e00, -3.79382e-18, 0.00000e00],
                [0.00000e00, 0.00000e00, -3.79382e-18],
            ]
        ]
    )

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack)


def test_update_stress_2d():
    import warnings

    warnings.warn("Test not implemented")
