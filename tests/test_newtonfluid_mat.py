import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""
    constitutive_laws = hdx.NewtonFluid(K=2.0 * 10**6, viscosity=0.2)
    assert isinstance(constitutive_laws, hdx.NewtonFluid)


def test_update_stress_3d():
    constitutive_laws = hdx.NewtonFluid(K=2.0 * 10**6, viscosity=0.001, rho_0=1400)

    rho = 1399  # volume0/volume

    stress = constitutive_laws.update_ip(
        stress_prev=jnp.zeros((3, 3)),
        F=jnp.eye(3),
        L=jnp.eye(3) * 0.1,
        rho=jnp.ones(1) * rho,
        dim=3,
    )

    expected_stress_stack = jnp.eye(3) * 9978.771

    np.testing.assert_allclose(stress, expected_stress_stack, rtol=1e-3)
