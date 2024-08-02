"""Unit tests for the Nodes dataclass."""

import jax.numpy as jnp
import numpy as np
import pytest

import pymudokon as pm


@pytest.mark.parametrize("dim, exp_num_nodes", [(1, 3), (2, 9), (3, 27)])
def test_create(dim, exp_num_nodes):
    """Unit test to create grid nodes over multiple dimensions."""
    nodes = pm.Nodes.create(
        origin=jnp.zeros(dim),
        end=jnp.ones(dim),
        node_spacing=0.5,
    )

    assert isinstance(nodes, pm.Nodes)

    assert nodes.num_nodes_total == exp_num_nodes


def test_refresh():
    """Unit test to reset node state."""
    nodes = pm.Nodes.create(
        origin=jnp.zeros(3),
        end=jnp.ones(3),
        node_spacing=0.5,
    )

    nodes = nodes.replace(mass_stack=jnp.ones(9).astype(jnp.float32))

    np.testing.assert_allclose(nodes.mass_stack, jnp.ones(9))

    nodes = nodes.refresh()

    np.testing.assert_allclose(nodes.mass_stack, jnp.zeros(9))
