# """Unit tests for rigid particles."""

# import jax.numpy as jnp
# import numpy as np

# import pymudokon as pm


# def test_init():
#     """Unit test to initialize rigid particles."""

#     rigid_particles = pm.RigidParticles.create(
#         position_stack=jnp.array([[0.45, 0.21], [0.8, 0.4]]),
#         velocity_stack=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
#         shapefunction=pm.CubicShapeFunction.create(num_particles=2, dim=2),
#     )
#     assert isinstance(rigid_particles, pm.RigidParticles)


# def test_apply_on_node_moments():
#     """Unit test apply rigid particles on nodes moments."""
#     nodes = pm.Nodes.create(
#         origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5
#     )

#     particles = pm.Particles.create(
#         position_stack=jnp.array([[0.2, 0.7]]), mass_stack=jnp.array([1.0])
#     )

#     shapefunctions = pm.LinearShapeFunction.create(num_particles=1, dim=2)

#     shapefunctions, _ = shapefunctions.calculate_shapefunction(
#         origin=nodes.origin,
#         inv_node_spacing=nodes.inv_node_spacing,
#         grid_size=nodes.grid_size,
#         position_stack=particles.position_stack,
#     )

#     r_shapefunctions = pm.LinearShapeFunction.create(num_particles=1, dim=2)

#     rigid_particles = pm.RigidParticles.create(
#         position_stack=jnp.array([[0.7, 0.2]]),
#         velocity_stack=jnp.array([[0.0, 0.0]]),
#         shapefunction=r_shapefunctions,
#     )
#     nodes = nodes.replace(
#         moment_nt_stack=jnp.ones(nodes.moment_nt_stack.shape, dtype=jnp.float32),
#         mass_stack=jnp.ones(nodes.mass_stack.shape, dtype=jnp.float32),
#     )
#     nodes, rigid_particles = rigid_particles.apply_on_nodes_moments(
#         nodes, particles, shapefunctions
#     )
#     expected_moment_nt_stack = jnp.array(
#         [
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [0.76923066, 1.1538463],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#             [
#                 1.0,
#                 1.0,
#             ],
#         ]
#     )
#     np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)
