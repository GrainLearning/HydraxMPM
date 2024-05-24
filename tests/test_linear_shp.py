"""Unit tests for the linear shape functions."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestLinearShapeFunctions(unittest.TestCase):
    """Unit tests for linear shape functions."""

    @staticmethod
    def test_init():
        """Unit test to test initialization."""
        # 2D linear element stencil size of 4
        shapefunction = pm.LinearShapeFunction.register(num_particles=2, dim=2)

        assert isinstance(shapefunction, pm.LinearShapeFunction)

        np.testing.assert_allclose(shapefunction.intr_shapef, jnp.zeros((2, 4), dtype=jnp.float32))

        np.testing.assert_allclose(
            shapefunction.intr_shapef_grad,
            jnp.zeros((2, 4, 2), dtype=jnp.float32),
        )

    @staticmethod
    def test_linear_shapefunction_vmap():
        """Test the linear shape function calculation for a vectorized input."""
        # 2D linear element stencil size of 4
        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        positions = jnp.array([[0.25, 0.25], [0.8, 0.4]])

        inv_node_spacing = 2.0

        origin = jnp.array([0.0, 0.0])

        grid_size = jnp.array([3, 3])  # 3x3 grid

        intr_dist, intr_bins, intr_hashes = jax.vmap(
            pm.core.interactions.vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(positions, stencil, origin, inv_node_spacing, grid_size)

        shapef, shapef_grad = jax.vmap(
            pm.shapefunctions.linear.vmap_linear_shapefunction, in_axes=(0, None), out_axes=(0, 0)
        )(intr_dist.reshape(-1, 2, 1), inv_node_spacing)

        np.testing.assert_allclose(shapef.shape, (8, 1, 1))

        np.testing.assert_allclose(
            shapef,
            jnp.array([[[0.25]], [[0.25]], [[0.25]], [[0.25]], [[0.08]], [[0.12]], [[0.32]], [[0.48]]]),
        )

        np.testing.assert_allclose(shapef_grad.shape, (8, 2, 1))

        np.testing.assert_allclose(
            shapef_grad,
            [
                [[-1.0], [-1.0]],
                [[1.0], [-1.0]],
                [[-1.0], [1.0]],
                [[1.0], [1.0]],
                [[-0.4], [-0.8]],
                [[0.4], [-1.2]],
                [[-1.6], [0.8]],
                [[1.6], [1.2]],
            ],
        )

    @staticmethod
    def test_calculate_shapefunction():
        """Test the linear shape function for top level container input."""
        particles = pm.Particles.register(positions=jnp.array([[0.25, 0.25], [0.8, 0.4]]))

        nodes = pm.Nodes.register(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

        shapefunction = pm.LinearShapeFunction.register(num_particles=2, dim=2)

        shapefunction = shapefunction.get_interactions(particles, nodes)

        shapefunction = shapefunction.calculate_shapefunction(nodes)

        np.testing.assert_allclose(shapefunction.intr_shapef.shape, (8, 1, 1))

        np.testing.assert_allclose(
            shapefunction.intr_shapef,
            jnp.array([[[0.25]], [[0.25]], [[0.25]], [[0.25]], [[0.08]], [[0.12]], [[0.32]], [[0.48]]]),
        )

        np.testing.assert_allclose(shapefunction.intr_shapef_grad.shape, (8, 2, 1))

        np.testing.assert_allclose(
            shapefunction.intr_shapef_grad,
            [
                [[-1.0], [-1.0]],
                [[1.0], [-1.0]],
                [[-1.0], [1.0]],
                [[1.0], [1.0]],
                [[-0.4], [-0.8]],
                [[0.4], [-1.2]],
                [[-1.6], [0.8]],
                [[1.6], [1.2]],
            ],
        )


if __name__ == "__main__":
    unittest.main()
