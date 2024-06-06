"""Unit tests for the Interactions state."""

import jax.numpy as jnp

import pymudokon as pm
import numpy as np

def test_create():
    """Unit test to check the creation of the Interactions object."""
    stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    stencil_size, dim = stencil.shape
    num_particles = 2
    shapefunction = pm.ShapeFunction(
        jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
        jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
        stencil,
    )
    assert isinstance(shapefunction, pm.ShapeFunction)


def test_vmap_interactions():
    """Unit test for vectorized particle-node interaction mapping."""
    positions = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

    origin = jnp.array([0.0, 0.0])

    inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

    grid_size = jnp.array([3, 3])

    stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    num_particles = positions.shape[0]
    stencil_size, dim = stencil.shape
    
    shapefunction = pm.ShapeFunction(
        jnp.zeros((num_particles * stencil_size), dtype=jnp.int32),
        jnp.zeros((num_particles, stencil_size), dtype=jnp.float32),
        jnp.zeros((num_particles, stencil_size, dim), dtype=jnp.float32),
        stencil
    )
    intr_dist, intr_bins, intr_hashes = shapefunction.vmap_get_interactions(positions, origin, inv_node_spacing, grid_size)

    np.testing.assert_allclose(
        intr_dist,
        jnp.array(
            [
                [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
                [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
                [[0.6, 0.8], [-0.4, 0.8], [0.6, -0.2], [-0.4, -0.2]],
            ]
        ),
    )


# class TestInteractions(unittest.TestCase):
#     """Unit tests for the Interactions state."""

#     @staticmethod
#     def test_init():
#         """Unit test to check the initialization of the Interactions state."""
#         # 2D linear element stencil size of 4
#         stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

#         interactions = pm.Interactions.create(stencil=stencil, num_particles=2)

#         assert isinstance(interactions, pm.Interactions)

#         np.testing.assert_allclose(interactions.intr_dist, jnp.zeros((8, 2, 1), dtype=jnp.float32))

#         np.testing.assert_allclose(interactions.intr_bins, jnp.zeros((8, 2, 1), dtype=jnp.int32))

#         np.testing.assert_allclose(interactions.intr_hashes, jnp.zeros((8), dtype=jnp.int32))

#     @staticmethod
#     def test_vmap_interactions():
#         """Unit test for vectorized particle-node interaction mapping."""
#         positions = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

#         origin = jnp.array([0.0, 0.0])

#         inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

#         grid_size = jnp.array([3, 3])  # 3x3 grid

#         stencil = jnp.array([[                  [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
#                     [[0.6, 0.8], [-0.4, 0.8], [0.6, -0.2], [-0.4, -0.2]],
#                 ]
#             ),
#         )

#         np.testing.assert_allclose(intr_bins.shape, (3, 4, 2))

#         np.testing.assert_allclose(
#             intr_bins,
#             jnp.array(
#                 [
#                     [[0, 0], [1, 0], [0, 1], [1, 1]],
#                     [[0, 0], [1, 0], [0, 1], [1, 1]],
#                     [[1, 0], [2, 0], [1, 1], [2, 1]],
#                 ]
#             ),
#         )

#      ashes, jnp.array([[0, 1, 3, 4], [0, 1, 3, 4], [1, 2, 4, 5]]))

#     @staticmethod
#     def test_get_interactions():
#         """Unit test to get the particle-node pair interactions (top-level)."""
#         particles = pm.Particles.create(positions=jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]]))
#         nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

#         stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

#         interactions = pm.Interactions.create(stencil=stencil, num_particles=3)  # unused intentionally

#         interactions = interactions.get_interactions(particles, nodes)

#         np.testing.assert_allclose(interactions.intr_dist.shape, (12, 2, 1))

#         np.testing.assert_allclose(
#             interactions.intr_dist,
#             jnp.array(
#                 [
#                     [[0.5], [0.5]],
#                     [[-0.5], [0.5]],
#                     [[0.5], [-0.5]],
#                     [[-0.5], [-0.5]],
#                     [[0.5], [0.5]],
#                     [[-0.5], [0.5]],
#                     [[0.5], [-0.5]],
#                     [[-0.5], [-0.5]],
#                     [[0.6], [0.8]],
#                     [[-0.4], [0.8]],
#                     [[0.6], [-0.2]],
#                     [[-0.4], [-0.2]],
#                 ]
#             ),
#         )

#         np.testing.assert_allclose(interactions.intr_bins.shape, (12, 2, 1))

#         np.testing.assert_allclose(
#             interactions.intr_bins,
#             jnp.array(
#                 [
#                     [[0], [0]],
#                     [[1], [0]],
#                     [[0], [1]],
#                     [[1], [1]],
#                     [[0], [0]],
#                     [[1], [0]],
#                     [[0], [1]],
#                     [[1], [1]],
#                     [[1], [0]],
#                     [[2], [0]],
#                     [[1], [1]],
#                     [[2], [1]],
#                 ]
#             ),
#         )

#         # hashes must be reshaped to (num_particles * stencil_size,)
#         np.testing.assert_allclose(interactions.intr_hashes.shape, (12,))
#         np.testing.assert_allclose(interactions.intr_hashes, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 1, 2, 4, 5]))


# if __name__ == "__main__":
#     unittest.main()
