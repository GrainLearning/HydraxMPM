"""Unit tests for the NodesContainer class."""

import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm

# TODO add test for node species


class TestNodes(unittest.TestCase):
    """Unit tests for the TestNodes and functions."""

    @staticmethod
    def test_init():
        """Unit test to initialize the NodesContainer class."""
        nodes = pm.Nodes.create(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
        )

        assert isinstance(nodes, pm.Nodes)

        assert nodes.num_nodes_total == 9

        np.testing.assert_allclose(nodes.masses, jnp.zeros(9))

        np.testing.assert_allclose(nodes.moments_nt, jnp.zeros((9, 2)))

        np.testing.assert_allclose(nodes.moments, jnp.zeros((9, 2)))

    @staticmethod
    def test_refresh():
        """Unit test to refresh/reset the state of the nodes."""
        nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

        nodes = nodes.replace(masses=jnp.ones(9).astype(jnp.float32))

        np.testing.assert_allclose(nodes.masses, jnp.ones(9))

        nodes = nodes.refresh()

        np.testing.assert_allclose(nodes.masses, jnp.zeros(9))


if __name__ == "__main__":
    unittest.main()
