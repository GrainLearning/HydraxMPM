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
