# """Unit tests for Newtonian fluid."""

import equinox as eqx
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
    material = hdx.NewtonFluid(
        config=config,
        K=2.0 * 10**6,
        viscosity=0.2,
    )

    assert isinstance(material, hdx.NewtonFluid)


def test_update_stress_2d():
    """Unit test the isotropic linear elastic material for 2d."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=1,
        dt=0.1,
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.1]]),
    )

    particles = eqx.tree_at(
        lambda state: (
            state.volume_stack,
            state.volume0_stack,
            state.stress_stack,
            state.L_stack,
        ),
        particles,
        (
            jnp.array([0.019]),
            jnp.array([0.2]),
            jnp.eye(3).reshape(-1, 3, 3),
            jnp.stack([jnp.eye(3) * 0.1]),
        ),
    )

    material = hdx.NewtonFluid(
        config=config,
        K=2.0 * 10**6,
        viscosity=0.001,
    )

    particles, material = material.update_from_particles(particles)

    expected_stress_stack = jnp.array([jnp.eye(3)]) * -2.863947e13

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack, rtol=1e-3)
