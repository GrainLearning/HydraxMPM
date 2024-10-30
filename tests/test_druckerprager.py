# """Unit tests for the drucker prager material module."""

# import jax.numpy as jnp
# import numpy as np

# import pymudokon as pm


# def test_create():
#     """Unit test the initialization drucker prager material."""
#     material = pm.DruckerPrager.create(
#         E=1000.0,
#         nu=0.2,
#         M=1.2,
#         M2=1.0,
#         M_hat=0.8,
#         c0=0.0,
#         H=0.1,
#         num_particles=2,
#     )
#     assert isinstance(material, pm.DruckerPrager)

#     np.testing.assert_allclose(material.E, 1000.0)
#     np.testing.assert_allclose(material.nu, 0.2)
#     np.testing.assert_allclose(material.G, 416.666667)
#     np.testing.assert_allclose(material.K, 555.5555555555557)

#     np.testing.assert_allclose(material.M, 1.2)
#     np.testing.assert_allclose(material.M_hat, 0.8)

#     np.testing.assert_allclose(material.M2, 1.0)
#     np.testing.assert_allclose(material.c0, 0.0)


# def test_update_stress_3d():
#     """Unit test the isotropic linear elastic material for 3d."""
#     particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1, 0.0]]))

#     particles = particles.replace(L_stack=jnp.stack([jnp.eye(3) * 0.1]))

#     stress_ref = jnp.eye(3) * -1e5
#     material = pm.DruckerPrager.create(
#         E=1000.0,
#         nu=0.2,
#         M=1.2,
#         M2=1.0,
#         M_hat=0.8,
#         c0=0.0,
#         H=0.1,
#         num_particles=1,
#         stress_ref_stack=jnp.array([stress_ref]),
#     )

#     particles, material = material.update_from_particles(particles, 0.1)

#     import warnings

#     warnings.warn("Not finished implmenting test")

#     expected_stress_stack = jnp.array(
#         [[[-99983.336, 0.0, 0.0], [0.0, -99983.336, 0.0], [0.0, 0.0, -99983.336]]]
#     )
#     np.testing.assert_allclose(particles.stress_stack, expected_stress_stack)


# def test_update_stress_2d():
#     import warnings

#     warnings.warn("Test not implemented")
