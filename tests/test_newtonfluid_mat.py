"""Unit tests for Newtonian fluid."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test the initialization of the isotropic linear elastic material."""
    material = pm.NewtonFluid.create(
        K=2.0 * 10**6,
        viscosity=0.2,
    )

    assert isinstance(material, pm.NewtonFluid)


def test_update_stress_2d():
    """Unit test the isotropic linear elastic material for 2d."""
    particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1]]))

    particles = particles.replace(
        volume_stack=jnp.array([0.019]),
        volume0_stack=jnp.array([0.2]),
        stress_stack=jnp.array([[[0.0, 0.0, 0.0]]]),
        L_stack=jnp.stack([jnp.eye(3) * 0.1]),
    )
    material = pm.NewtonFluid.create(
        K=2.0 * 10**6,
        viscosity=0.001,
    )

    particles, material = material.update_from_particles(particles, 0.1)

    expected_stress_stack = jnp.array([jnp.eye(3)]) * -2.863947e13

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack, rtol=1e-3)


def test_update_stress_3d():
    """Unit test the isotropic linear elastic material for 3d."""
    particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1, 0.0]]))

    particles = particles.replace(
        volume_stack=jnp.array([0.019]),
        volume0_stack=jnp.array([0.2]),
        stress_stack=jnp.array([[[0.0, 0.0, 0.0]]]),
        L_stack=jnp.stack([jnp.eye(3) * 0.1]),
    )
    material = pm.NewtonFluid.create(
        K=2.0 * 10**6,
        viscosity=0.001,
    )

    particles, material = material.update_from_particles(particles, 0.1)

    expected_stress_stack = jnp.array([jnp.eye(3)]) * -2.863947e13

    np.testing.assert_allclose(particles.stress_stack, expected_stress_stack)


def test_update_stress_2d():
    import warnings

    warnings.warn("Test not implemented")
