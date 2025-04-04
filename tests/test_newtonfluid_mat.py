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

    rho_rho_0 = 1400 / 1390  # volume0/volume

    stress = constitutive_laws.update_ip(
        deps_dt=jnp.eye(3), rho_rho_0=jnp.ones(1) * rho_rho_0
    )

    expected_stress_stack = jnp.eye(3) * -102920.055

    np.testing.assert_allclose(stress, expected_stress_stack, rtol=1e-3)
