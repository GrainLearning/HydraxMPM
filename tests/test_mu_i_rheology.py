import jax.numpy as jnp

import hydraxmpm as hdx
from hydraxmpm.constitutive_laws import constitutive_law
import numpy as np


def test_create():
    """Unit test the initialization drucker prager material."""
    # init by pressure
    constitutive_law = hdx.MuI_incompressible(
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        K=50 * 2000 * 9.8 * 0.4,
        d=0.0053,
        rho_p=2000,
        init_by_density=False,
    )
    assert isinstance(constitutive_law, hdx.MuI_incompressible)

    material_points = hdx.MaterialPoints(stress_stack=jnp.array([jnp.eye(3) * -1e5]))
    constitutive_law, material_points = constitutive_law.init_state(material_points)

    np.testing.assert_allclose(material_points.p_stack, jnp.array([1e5]))
    phi_stack = material_points.phi_stack(constitutive_law.rho_p)
    np.testing.assert_allclose(phi_stack, jnp.array([0.0005]))
    # init by density
    constitutive_law = hdx.MuI_incompressible(
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        K=50 * 2000 * 9.8 * 0.4,
        d=0.0053,
        rho_p=2000,
        rho_0=1900,
        init_by_density=True,
    )
    assert isinstance(constitutive_law, hdx.MuI_incompressible)

    material_points = hdx.MaterialPoints(stress_stack=jnp.array([jnp.eye(3) * -1e5]))
    constitutive_law, material_points = constitutive_law.init_state(material_points)

    np.testing.assert_allclose(material_points.p_stack, jnp.array([0.0]))
