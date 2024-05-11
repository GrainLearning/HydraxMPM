"""Unit tests for the InteractionsContainer state.

Tests and examples show how to uses interactions.

The module contains the following main components:
- TestInteractions.test_init:
    Unit test to check the initialization of the
    InteractionsContainer state.
- TestInteractions.test_vmap_interactions:
    Unit test for vectorized particle-node interaction mapping.
- TestInteractions.test_get_interactions:
    Unit test to get the particle-node pair interactions (top-level).
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestInteractions(unittest.TestCase):
    """Unit tests for the InteractionsContainer state."""

    @staticmethod
    def test_init():
        """Unit test to check the initialization of the InteractionsContainer state."""
        # 2D linear element stencil size of 4
        stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        interactions = pm.core.interactions.init(stencil_array=stencil_array, num_particles=2)

        assert isinstance(interactions, pm.interactions.InteractionsContainer)

        np.testing.assert_allclose(interactions.intr_dist_array, jnp.zeros((8, 2, 1), dtype=jnp.float32))

        np.testing.assert_allclose(interactions.intr_bins_array, jnp.zeros((8, 2, 1), dtype=jnp.int32))

        np.testing.assert_allclose(interactions.intr_hashes_array, jnp.zeros((8), dtype=jnp.int32))

    @staticmethod
    def test_vmap_interactions():
        """Unit test for vectorized particle-node interaction mapping."""
        positions = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

        origin = jnp.array([0.0, 0.0])

        inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

        grid_size = jnp.array([3, 3])  # 3x3 grid

        stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        intr_dist_array, intr_bins_array, intr_hashes_array = jax.vmap(
            pm.interactions.vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(positions, stencil_array, origin, inv_node_spacing, grid_size)

        np.testing.assert_allclose(intr_dist_array.shape, (3, 4, 2))

        np.testing.assert_allclose(
            intr_dist_array,
            jnp.array(
                [
                    [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
                    [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
                    [[0.6, 0.8], [-0.4, 0.8], [0.6, -0.2], [-0.4, -0.2]],
                ]
            ),
        )

        np.testing.assert_allclose(intr_bins_array.shape, (3, 4, 2))

        np.testing.assert_allclose(
            intr_bins_array,
            jnp.array(
                [
                    [[0, 0], [1, 0], [0, 1], [1, 1]],
                    [[0, 0], [1, 0], [0, 1], [1, 1]],
                    [[1, 0], [2, 0], [1, 1], [2, 1]],
                ]
            ),
        )

        np.testing.assert_allclose(intr_hashes_array.shape, (3, 4))
        np.testing.assert_allclose(intr_hashes_array, jnp.array([[0, 1, 3, 4], [0, 1, 3, 4], [1, 2, 4, 5]]))

    @staticmethod
    def test_get_interactions():
        """Unit test to get the particle-node pair interactions (top-level)."""
        particles = pm.core.particles.init(positions=jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]]))
        nodes = pm.core.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=1,
        )

        stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        interactions = pm.core.interactions.init(
            stencil_array=stencil_array, num_particles=3
        )  # unused intentionally

        interactions = pm.core.interactions.get_interactions(interactions, particles, nodes)

        np.testing.assert_allclose(interactions.intr_dist_array.shape, (12, 2, 1))

        np.testing.assert_allclose(
            interactions.intr_dist_array,
            jnp.array(
                [
                    [[0.5], [0.5]], [[-0.5], [0.5]], [[0.5], [-0.5]], [[-0.5], [-0.5]],
                    [[0.5], [0.5]], [[-0.5], [0.5]], [[0.5], [-0.5]], [[-0.5], [-0.5]],
                    [[0.6], [0.8]], [[-0.4], [0.8]], [[0.6], [-0.2]], [[-0.4], [-0.2]],
                ]
            ),
        )

        np.testing.assert_allclose(interactions.intr_bins_array.shape, (12, 2, 1))

        np.testing.assert_allclose(
            interactions.intr_bins_array,
            jnp.array(
                [
                    [[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]],
                    [[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]],
                    [[1], [0]], [[2], [0]], [[1], [1]], [[2], [1]],
                ]
            ),
        )

        # hashes must be reshaped to (num_particles * stencil_size,)
        np.testing.assert_allclose(interactions.intr_hashes_array.shape, (12,))
        np.testing.assert_allclose(
            interactions.intr_hashes_array, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 1, 2, 4, 5])
        )


if __name__ == "__main__":
    unittest.main()
