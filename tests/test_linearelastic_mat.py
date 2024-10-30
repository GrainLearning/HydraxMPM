"""Unit tests for the isotropic linear elastic material module."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )
    material = hdx.LinearIsotropicElastic(config=config, E=1000.0, nu=0.2)

    assert isinstance(material, hdx.LinearIsotropicElastic)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)


def test_update_stress_3d():
    """Unit test the isotropic linear elastic material for 3d."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    material = hdx.LinearIsotropicElastic(E=0.1, nu=0.1, config=config)

    new_stress = material.update_ip(
        stress_prev=jnp.zeros((3, 3)), F=jnp.eye(3), L=jnp.eye(3) * 0.1, phi=None
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

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    material = hdx.LinearIsotropicElastic(E=0.1, nu=0.1, config=config)

    new_stress = material.update_ip(
        stress_prev=jnp.zeros((3, 3)), F=jnp.eye(3), L=jnp.eye(3) * 0.1, phi=None
    )
    expected_stress = jnp.array(
        [
            [
                0.00113636,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.00113636,
                0.0,
            ],
            [0.0, 0.0, 0.00022727],
        ]
    )

    np.testing.assert_allclose(new_stress, expected_stress, rtol=1e-3)
