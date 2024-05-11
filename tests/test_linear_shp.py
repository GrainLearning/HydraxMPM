"""Unit tests for the linear shape functions.

Tests and examples on how to use linear shape functions.

The module contains the following main components:
- TestShapeFunctions.test_init:
    Test initialization of the linear shape function state.
- TestShapeFunctions.test_linear_shapefunction_vmap:
    Test linear shape function vectorized calculation.
- TestShapeFunctions.test_calculate_shapefunction:
    Test linear shape function top level calculation.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestShapeFunctions(unittest.TestCase):
    """Unit tests for linear shape functions."""
    @staticmethod
    def test_init():
        """Unit test to test initialization."""
        # 2D linear element stencil size of 4
        shapefunction_state = pm.linear_shp.init(num_particles=2, stencil_size=4, dim=2)

        assert isinstance(shapefunction_state, pm.shapefunctions.linear_shp.ShapeFunctionContainer)

        np.testing.assert_allclose(shapefunction_state.shapef_array, jnp.zeros((2, 4), dtype=jnp.float32))

        np.testing.assert_allclose(
            shapefunction_state.shapef_grad_array,
            jnp.zeros((2, 4, 2), dtype=jnp.float32),
        )

    @staticmethod
    def test_linear_shapefunction_vmap():
        """Test the linear shape function calculation for a vectorized input."""
        # 2D linear element stencil size of 4
        stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        positions = jnp.array([[0.25, 0.25], [0.8, 0.4]])

        inv_node_spacing = 2.0

        origin = jnp.array([0.0, 0.0])

        grid_size = jnp.array([3, 3])  # 3x3 grid

        intr_dist_array, intr_bins_array, intr_hashes_array = jax.vmap(
            pm.interactions.vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(positions, stencil_array, origin, inv_node_spacing, grid_size)

        shapef_array, shapef_grad_array = jax.vmap(
            pm.linear_shp.vmap_linear_shapefunction, in_axes=(0, None), out_axes=(0, 0)
        )(intr_dist_array.reshape(-1,2,1), inv_node_spacing)


        np.testing.assert_allclose(shapef_array.shape, (8, 1, 1))

        np.testing.assert_allclose(
            shapef_array,
            jnp.array([[[0.25]], [[0.25]], [[0.25]], [[0.25]], [[0.08]], [[0.12]], [[0.32]], [[0.48]]]),
        )

        np.testing.assert_allclose(shapef_grad_array.shape, (8, 2, 1))

        np.testing.assert_allclose(
            shapef_grad_array,
            [
                [[-1.0], [-1.0]], [[1.0], [-1.0]], [[-1.0], [1.0]], [[1.0], [1.0]],
                [[-0.4], [-0.8]],[[0.4], [-1.2]],[[-1.6], [0.8]],[[1.6], [1.2]],
            ],
        )

    @staticmethod
    def test_calculate_shapefunction():
        """Test the linear shape function for top level container input."""
        particles_state = pm.core.particles.init(positions=jnp.array([[0.25, 0.25], [0.8, 0.4]]))

        nodes_state = pm.core.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=1,
        )

        stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        interactions_state = pm.core.interactions.init(
            stencil_array=stencil_array, num_particles=3
        )  # unused intentionally

        interactions_state = pm.core.interactions.get_interactions(interactions_state, particles_state, nodes_state)

        shapefunction_state = pm.linear_shp.init( # noqa: F841
            num_particles=2, stencil_size=4, dim=2
        )

        shapefunction_state = pm.linear_shp.calculate_shapefunction(
            shapefunction_state, nodes_state, interactions_state
        )

        np.testing.assert_allclose(shapefunction_state.shapef_array.shape, (8, 1, 1))

        np.testing.assert_allclose(
            shapefunction_state.shapef_array,
            jnp.array([[[0.25]], [[0.25]], [[0.25]], [[0.25]], [[0.08]], [[0.12]], [[0.32]], [[0.48]]]),
        )

        np.testing.assert_allclose(shapefunction_state.shapef_grad_array.shape, (8, 2, 1))

        np.testing.assert_allclose(
            shapefunction_state.shapef_grad_array,
            [
                [[-1.0], [-1.0]], [[1.0], [-1.0]], [[-1.0], [1.0]], [[1.0], [1.0]],
                [[-0.4], [-0.8]],[[0.4], [-1.2]],[[-1.6], [0.8]],[[1.6], [1.2]],
            ]
        )


if __name__ == "__main__":
    unittest.main()
