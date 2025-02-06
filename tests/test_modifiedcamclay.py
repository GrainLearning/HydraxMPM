"""Unit tests for the modified cam clay constitutive_law module."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization drucker prager constitutive_law."""

    constitutive_law = hdx.ModifiedCamClay(
        nu=0.2, M=1.2, R=1.0, lam=0.8, kap=0.1, ln_N=1.0, init_by_density=False
    )

    assert isinstance(constitutive_law, hdx.ModifiedCamClay)

    material_points = hdx.MaterialPoints(stress_stack=jnp.array([jnp.eye(3) * -1e5]))
    constitutive_law, material_points = constitutive_law.init_state(material_points)

    np.testing.assert_allclose(constitutive_law.nu, 0.2)
    np.testing.assert_allclose(constitutive_law.M, 1.2)
    np.testing.assert_allclose(constitutive_law.R, 1)
    np.testing.assert_allclose(constitutive_law.lam, 0.8)
    np.testing.assert_allclose(constitutive_law.kap, 0.1)
    np.testing.assert_allclose(constitutive_law.px_hat_stack.at[0].get(), 1e5)
