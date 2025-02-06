"""Unit tests for the isotropic linear elastic material module."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""

    material = hdx.LinearIsotropicElastic(E=1000.0, nu=0.2)

    assert isinstance(material, hdx.LinearIsotropicElastic)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)


def test_update_stress_3d():
    material = hdx.LinearIsotropicElastic(E=0.1, nu=0.1, dt=0.1)

    new_stress = material.update_ip(
        stress_prev=jnp.zeros((3, 3)), dim=3, F=jnp.eye(3), deps=jnp.eye(3) * 0.01
    )

    expected_stress = jnp.array(
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
    )

    np.testing.assert_allclose(new_stress, expected_stress)


def test_update_stress_2d():
    """Unit test the isotropic linear elastic material for 2d."""

    material = hdx.LinearIsotropicElastic(E=0.1, nu=0.1)

    new_stress = material.update_ip(
        stress_prev=jnp.zeros((3, 3)), dim=2, F=jnp.eye(3), deps=jnp.eye(3) * 0.01
    )
    expected_stress = jnp.array(
        [
            [0.00113636, 0.0, 0.0],
            [0.0, 0.00113636, 0.0],
            [0.0, 0.0, 0.00022727],
        ]
    )

    np.testing.assert_allclose(new_stress, expected_stress, rtol=1e-3)
