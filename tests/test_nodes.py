"""Unit tests for the NodesContainer class.

Test and examples on how to use the NodesContainer class to to setup/update particle state

The module contains the following main components:
- TestNodes.test_init:
    Unit test to initialize the NodesContainer class.
- TestParticles.test_refresh:
    Unit test for resetting variables of the NodesContainer state.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestNodes(unittest.TestCase):
    """Unit tests for the TestNodes and functions."""

    @staticmethod
    def test_init():
        """Unit test to initialize the NodesContainer class."""
        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=1,
        )

        assert isinstance(nodes, pm.core.nodes.NodesContainer)

        assert nodes.num_nodes_total == 9

        np.testing.assert_allclose(nodes.masses_array, jnp.zeros(9))

        np.testing.assert_allclose(nodes.moments_nt_array, jnp.zeros((9, 2)))

        np.testing.assert_allclose(nodes.moments_array, jnp.zeros((9, 2)))

    @staticmethod
    def test_refresh():
        """Unit test to refresh/reset the state of the nodes."""
        nodes = pm.core.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=1,
        )

        nodes = nodes._replace(masses_array=jnp.ones(9).astype(jnp.float32))

        np.testing.assert_allclose(nodes.masses_array, jnp.ones(9))

        nodes = pm.core.nodes.refresh(nodes)

        np.testing.assert_allclose(nodes.masses_array, jnp.zeros(9))


if __name__ == "__main__":
    unittest.main()
