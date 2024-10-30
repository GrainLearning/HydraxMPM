# """Unit tests for the modified cam clay material module."""

# import jax.numpy as jnp
# import numpy as np

# import pymudokon as pm


# def test_create():
#     """Unit test the initialization drucker prager material."""

#     stress_ref = jnp.eye(3) * -1e5
#     material = pm.ModifiedCamClay.create(
#         nu=0.2,
#         M=1.2,
#         R=2.0,
#         lam=0.8,
#         kap=0.1,
#         Vs=2,
#         stress_ref_stack=jnp.array([stress_ref]),
#     )

#     assert isinstance(material, pm.ModifiedCamClay)

#     np.testing.assert_allclose(material.nu, 0.2)
#     np.testing.assert_allclose(material.M, 1.2)
#     np.testing.assert_allclose(material.R, 2)
#     np.testing.assert_allclose(material.lam, 0.8)
#     np.testing.assert_allclose(material.kap, 0.1)
#     np.testing.assert_allclose(material.p_c_stack.at[0].get(), 2e5)


# def test_update_stress_3d():
#     """Unit test the isotropic linear elastic material for 3d."""
#     particles = pm.Particles.create(position_stack=jnp.array([[0.1, 0.1, 0.0]]))

#     particles = particles.replace(
#         L_stack=-jnp.stack([jnp.eye(3) * 0.1]),
#         stress_stack=-jnp.stack([jnp.eye(3) * 1e5]),
#     )

#     stress_ref = jnp.eye(3) * -1e5
#     material = pm.ModifiedCamClay.create(
#         nu=0.2,
#         M=1.2,
#         R=2.0,
#         lam=0.8,
#         kap=0.1,
#         Vs=2,
#         stress_ref_stack=jnp.array([stress_ref]),
#     )

#     particles, material = material.update_from_particles(particles, 0.1)

#     expected_stress_stack = jnp.array(
#         [[[-142857.14, 0.0, 0.0], [0.0, -142857.14, 0.0], [0.0, 0.0, -142857.14]]]
#     )
#     np.testing.assert_allclose(particles.stress_stack, expected_stress_stack)


# def test_update_stress_2d():
#     import warnings

#     warnings.warn("Test not implemented")
