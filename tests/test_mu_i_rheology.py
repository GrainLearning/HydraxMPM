import jax.numpy as jnp

import hydraxmpm as hdx


def test_create():
    """Unit test the initialization drucker prager material."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    material = hdx.MuI_incompressible(
        config=config,
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        rho_p=2000,
        K=50 * 2000 * 9.8 * 0.4,
        d=0.0053,
    )

    assert isinstance(material, hdx.MuI_incompressible)


def test_update_stress_3d():
    """Unit test the isotropic linear elastic material for 3d."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    material = hdx.MuI_incompressible(
        config=config,
        mu_s=0.38,
        mu_d=0.64,
        I_0=0.279,
        rho_p=2000,
        K=50 * 2000 * 9.8 * 0.4,
        d=0.0053,
    )

    phi_0 = 0.65

    p_ref = material.get_p_ref(phi_0)

    stress_prev = p_ref * jnp.eye(3)

    _ = material.update_ip(
        stress_prev=stress_prev, F=jnp.eye(3), L=jnp.eye(3) * 0.1, phi=phi_0
    )
