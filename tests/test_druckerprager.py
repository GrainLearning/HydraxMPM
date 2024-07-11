"""Unit tests for the drucker prager material module."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test the initialization drucker prager material."""
    material = pm.DruckerPrager.create(
        E=1000.0,
        nu=0.2,
        friction_angle=jnp.deg2rad(30.0),
        dilatancy_angle=jnp.deg2rad(25.0),
        cohesion=0.0,
        H=0.1,
        num_particles=2,
    )

    assert isinstance(material, pm.DruckerPrager)
    np.testing.assert_allclose(material.E, 1000.0)
    np.testing.assert_allclose(material.nu, 0.2)
    np.testing.assert_allclose(material.G, 416.666667)
    np.testing.assert_allclose(material.K, 555.5555555555557)

    np.testing.assert_allclose(material.eta, 0.69282037)
    np.testing.assert_allclose(material.eta_hat, 0.56801546)

    np.testing.assert_allclose(material.xi, 1.2)
    np.testing.assert_allclose(material.c0, 0.0)


def test_update_stress_update_benchmark():
    material = pm.DruckerPrager.create(
        E=1000.0,
        nu=0.2,
        H=0.1,
        friction_angle=jnp.deg2rad(30.0),
        dilatancy_angle=jnp.deg2rad(25.0),
        cohesion=0.0,
    )
    material.update_stress_benchmark(
        strain_rate=jnp.eye(3).reshape(-1, 3, 3), volume_fraction=jnp.array([0.5]), dt=0.01
    )


test_create()
test_update_stress_update_benchmark()

# def test_update_stress_3d():
# """Unit test the drucker-parcker for 3d."""
#     particles = pm.Particles.create(positions=jnp.array([[0.1, 0.1, 0.0]]))

#     particles = particles.replace(velgrads=jnp.stack([jnp.eye(3) * 0.1]))

#     material = pm.LinearIsotropicElastic.create(E=0.1, nu=0.1, num_particles=1)

#     particles, material = material.update_stress(particles, 0.1)
#     expected_stresses = jnp.array(
#         [
#             [
#                 [
#                     0.00125,
#                     0.0,
#                     0.0,
#                 ],
#                 [
#                     0.0,
#                     0.00125,
#                     0.0,
#                 ],
#                 [0.0, 0.0, 0.00125],
#             ]
#         ]
#     )

#     np.testing.assert_allclose(particles.stresses, expected_stresses)


# def test_update_stress_2d():
#     """Unit test the isotropic linear elastic material for 2d."""
#     particles = pm.Particles.create(positions=jnp.array([[0.1, 0.1]]))

#     particles = particles.replace(velgrads=jnp.stack([jnp.eye(2) * 0.1]))

#     material = pm.LinearIsotropicElastic.create(E=0.1, nu=0.1, num_particles=1)

#     particles, material = material.update_stress(particles, 0.1)
#     expected_stresses = jnp.array(
#         [
#             [
#                 [
#                     0.00113636,
#                     0.0,
#                     0.0,
#                 ],
#                 [
#                     0.0,
#                     0.00113636,
#                     0.0,
#                 ],
#                 [0.0, 0.0, 0.00022727],
#             ]
#         ]
#     )

#     np.testing.assert_allclose(particles.stresses, expected_stresses, rtol=1e-3)
