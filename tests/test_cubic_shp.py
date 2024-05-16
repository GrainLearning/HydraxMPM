"""Unit tests for the linear shape functions."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestCubicShapeFunctions(unittest.TestCase):
    """Unit tests for linear shape functions."""

    @staticmethod
    def test_init():
        """Unit test to test initialization."""
        # 2D linear element stencil size of 4
        shapefunction = pm.CubicShapeFunction.register(num_particles=2, dim=2)

        assert isinstance(shapefunction, pm.CubicShapeFunction)

        np.testing.assert_allclose(shapefunction.shapef, jnp.zeros((2, 16), dtype=jnp.float32))

        np.testing.assert_allclose(
            shapefunction.shapef_grad,
            jnp.zeros((2, 16, 2), dtype=jnp.float32),
        )

    @staticmethod
    def test_cubic_shapefunction_vmap():
        """Test the cubic shape function calculation for a vectorized input."""
        # 2D cubic element stencil size of 16
        stencil = jnp.array(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [2, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [2, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
                [2, 1],
                [-1, 2],
                [0, 2],
                [1, 2],
                [2, 2],
            ]
        )

        positions = jnp.array([[0.25, 0.25], [0.3, 0.4]])

        inv_node_spacing = 16.0

        origin = jnp.array([0.0, 0.0])

        grid_size = jnp.array([18, 18])  # 3x3 grid

        species = jnp.ones(grid_size).astype(jnp.int32)

        species = species.at[:, 0].set(1)
        species = species.at[0, :].set(1)
        species = species.at[-1, :].set(1)
        species = species.at[:, -1].set(1)
        species = species.at[:, 1].set(2)
        species = species.at[1, :].set(2)

        species = species.at[:, -2].set(4)
        species = species.at[-2, :].set(4)
        species = species.reshape(-1)

        intr_dist, intr_bins, intr_hashes = jax.vmap(
            pm.core.interactions.vmap_interactions,
            in_axes=(0, None, None, None, None),
            out_axes=(0, 0, 0),
        )(positions, stencil, origin, inv_node_spacing, grid_size)

        intr_species = species.take(intr_hashes, axis=0).reshape(-1, 1, 1)

        shapef, shapef_grad = jax.vmap(
            pm.shapefunctions.cubic.vmap_cubic_shapefunction, in_axes=(0, 0, None), out_axes=(0, 0)
        )(intr_dist.reshape(-1, 2, 1), intr_species, inv_node_spacing)

        np.testing.assert_allclose(shapef.shape, (32, 1, 1))

        # TODO need further testing..

    @staticmethod
    def test_calculate_shapefunction():
        """Test the cubic shape function for top level container input."""
        positions = jnp.array([[0.25, 0.25], [0.3, 0.4]])
        particles = pm.Particles.register(positions=positions)

        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.1,
            particles_per_cell=1,
        )

        shapefunction = pm.CubicShapeFunction.register(num_particles=2, dim=2)

        nodes = shapefunction.set_nodes_species(nodes)

        interactions = pm.Interactions.register(stencil_size=4, num_particles=3, dim=2)  # unused intentionally

        interactions = interactions.get_interactions(particles, nodes, shapefunction)

        shapefunction = shapefunction.calculate_shapefunction(nodes, interactions)

        np.testing.assert_allclose(shapefunction.shapef.shape, (32, 1, 1))

        # TODO need further testing..


if __name__ == "__main__":
    unittest.main()
