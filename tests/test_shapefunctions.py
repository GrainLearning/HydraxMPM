# """Unit tests for the Interactions state."""

# import jax.numpy as jnp
# import numpy as np

# import pymudokon as pm


# def test_create():
#     """Unit test to check the creation of the Interactions object."""
#     stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
#     stencil_size, dim = stencil.shape
#     num_particles = 2
#     shapefunction = pm.ShapeFunction(
#         intr_hash_stack=jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
#         intr_shapef_stack=jnp.zeros((num_particles * stencil_size), dtype=jnp.float32),
#         intr_id_stack=jnp.arange(num_particles * stencil_size).astype(jnp.int32),
#         intr_shapef_grad_stack=jnp.zeros(
#             (num_particles * stencil_size, dim), dtype=jnp.float32
#         ),
#         stencil=stencil,
#     )
#     assert isinstance(shapefunction, pm.ShapeFunction)


# def test_vmap_intr():
#     """Unit test for vectorized particle-node interaction mapping."""
#     position_stack = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

#     origin = jnp.array([0.0, 0.0])

#     inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

#     grid_size = jnp.array([3, 3])

#     stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
#     num_particles = position_stack.shape[0]
#     stencil_size, dim = stencil.shape

#     intr_id_stack = jnp.arange(num_particles * stencil_size)

#     shapefunction = pm.ShapeFunction(
#         jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
#         jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
#         jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
#         stencil=stencil,
#         intr_id_stack=intr_id_stack,
#     )
#     intr_dist_stack, intr_hash_stack = shapefunction.vmap_intr(
#         intr_id_stack, position_stack, origin, inv_node_spacing, grid_size
#     )

#     np.testing.assert_allclose(intr_hash_stack.shape, (12))
#     np.testing.assert_allclose(
#         intr_hash_stack, jnp.array([0, 3, 1, 4, 0, 3, 1, 4, 3, 6, 4, 7])
#     )
