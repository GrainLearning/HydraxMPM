"""Unit tests for the Interactions state."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestInteractions(unittest.TestCase):
    """Unit tests for the Interactions state."""

    @staticmethod
    def test_init():
        """Unit test to check the initialization of the Interactions state."""
        # 2D linear element stencil size of 4
        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        interactions = pm.Interactions.register(stencil=stencil, num_particles=2)

        assert isinstance(interactions, pm.Interactions)

        np.testing.assert_allclose(interactions.intr_dist, jnp.zeros((8, 2, 1), dtype=jnp.float32))

        np.testing.assert_allclose(interactions.intr_bins, jnp.zeros((8, 2, 1), dtype=jnp.int32))

        np.testing.assert_allclose(interactions.intr_hashes, jnp.zeros((8), dtype=jnp.int32))

    @staticmethod
    def test_vmap_interactions():
        """Unit test for vectorized particle-node interaction mapping."""
        positions = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

        origin = jnp.array([0.0, 0.0])

        inv_node_spacing = 2.0  # 1.0 / 0.5 (node spacing of 0.5)

        grid_size = jnp.array([3, 3])  # 3x3 grid

        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        intr_dist, intr_bins, intr_hashes = jax.vmap(
            pm.core.interactions.vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(positions, stencil, origin, inv_node_spacing, grid_size)

        np.testing.assert_allclose(intr_dist.shape, (3, 4, 2))

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

        np.testing.assert_allclose(intr_bins.shape, (3, 4, 2))

        np.testing.assert_allclose(
            intr_bins,
            jnp.array(
                [
                    [[0, 0], [1, 0], [0, 1], [1, 1]],
                    [[0, 0], [1, 0], [0, 1], [1, 1]],
                    [[1, 0], [2, 0], [1, 1], [2, 1]],
                ]
            ),
        )

        np.testing.assert_allclose(intr_hashes.shape, (3, 4))
        np.testing.assert_allclose(intr_hashes, jnp.array([[0, 1, 3, 4], [0, 1, 3, 4], [1, 2, 4, 5]]))

    @staticmethod
    def test_get_interactions():
        """Unit test to get the particle-node pair interactions (top-level)."""
        particles = pm.Particles.register(positions=jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]]))
        nodes = pm.Nodes.register(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        interactions = pm.Interactions.register(stencil=stencil, num_particles=3)  # unused intentionally

        interactions = interactions.get_interactions(particles, nodes)

        np.testing.assert_allclose(interactions.intr_dist.shape, (12, 2, 1))

        np.testing.assert_allclose(
            interactions.intr_dist,
            jnp.array(
                [
                    [[0.5], [0.5]],
                    [[-0.5], [0.5]],
                    [[0.5], [-0.5]],
                    [[-0.5], [-0.5]],
                    [[0.5], [0.5]],
                    [[-0.5], [0.5]],
                    [[0.5], [-0.5]],
                    [[-0.5], [-0.5]],
                    [[0.6], [0.8]],
                    [[-0.4], [0.8]],
                    [[0.6], [-0.2]],
                    [[-0.4], [-0.2]],
                ]
            ),
        )

        np.testing.assert_allclose(interactions.intr_bins.shape, (12, 2, 1))

        np.testing.assert_allclose(
            interactions.intr_bins,
            jnp.array(
                [
                    [[0], [0]],
                    [[1], [0]],
                    [[0], [1]],
                    [[1], [1]],
                    [[0], [0]],
                    [[1], [0]],
                    [[0], [1]],
                    [[1], [1]],
                    [[1], [0]],
                    [[2], [0]],
                    [[1], [1]],
                    [[2], [1]],
                ]
            ),
        )

        # hashes must be reshaped to (num_particles * stencil_size,)
        np.testing.assert_allclose(interactions.intr_hashes.shape, (12,))
        np.testing.assert_allclose(interactions.intr_hashes, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 1, 2, 4, 5]))


if __name__ == "__main__":
    unittest.main()
