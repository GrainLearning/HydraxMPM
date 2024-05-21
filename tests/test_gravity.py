"""Unit tests for the NodesContainer class."""

import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class Gravity(unittest.TestCase):
    """Unit tests for the DirichletBox and functions."""

    @staticmethod
    def test_init():
        """Unit test to initialize the NodesContainer class."""

        box = pm.Gravity.register(jnp.array([0.0, 0.0]))

        assert isinstance(box, pm.Gravity)

    @staticmethod
    def test_apply_on_node_moments():
        """Unit test to initialize the NodesContainer class."""
        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=1,
        )

        grav = pm.Gravity.register(jnp.array([0.0, 9.8]))

        nodes = nodes.replace(
            masses=jnp.ones(nodes.num_nodes_total) * 1.0,
            moments=jnp.zeros((nodes.num_nodes_total, 2)),
            moments_nt=jnp.ones((nodes.num_nodes_total, 2)),
        )

        nodes, grav = grav.apply_on_nodes_moments(nodes=nodes, dt=0.01)

        expected_moments = jnp.array(
            [
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
                [0.0, 0.098],
            ]
        )

        expected_moments_nt = jnp.array(
            [
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
                [1.0, 1.098],
            ]
        )

        np.testing.assert_allclose(nodes.moments, expected_moments, rtol=1e-3)
        np.testing.assert_allclose(nodes.moments_nt, expected_moments_nt, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
