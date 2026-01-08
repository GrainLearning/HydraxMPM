"""Unit tests for the isotropic linear elastic material module."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx



def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""

    material = hdx.LinearElasticLaw(E=1000.0, nu=0.2)

    assert isinstance(material, hdx.LinearElasticLaw)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)



def test_update_stress_3d():
    material = hdx.LinearElasticLaw(E=0.1, nu=0.1)
    dt = 0.1

    # We want deps = 0.01 * I
    # deps = 0.5 * (L + L.T) * dt
    # 0.01 * I = 0.5 * (L + L.T) * 0.1
    # 0.2 * I = L + L.T
    # Let L = 0.1 * I
    L = jnp.eye(3) * 0.1
    stress_prev = jnp.zeros((3, 3))

    mp_state = hdx.MaterialPointState.create(
        L_stack=jnp.array([L]),
        stress_stack=jnp.array([stress_prev]),
        position_stack=jnp.zeros((1, 3)),
        velocity_stack=jnp.zeros((1, 3)),
        mass_stack=jnp.ones(1),
        volume_stack=jnp.ones(1),
    )

    law_state = material.create_state(mp_state)

    new_mp_state, _ = material.update(
        material_points_state=mp_state,
        law_state=law_state,
        dt=dt
    )

    new_stress = new_mp_state.stress_stack[0]
    
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
