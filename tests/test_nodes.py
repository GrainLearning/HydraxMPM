"""Unit tests for the Nodes dataclass."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to create grid nodes over multiple dimensions."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=1,
    )
    nodes = hdx.Nodes(config)

    assert isinstance(nodes, hdx.Nodes)

    assert nodes.config.num_cells == 121

    print(nodes.intr_dist_stack)


def test_refresh():
    """Unit test to reset node state."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0], end=[1.0, 1.0, 1.0], cell_size=0.5, num_points=1
    )
    nodes = hdx.Nodes(config)

    nodes = eqx.tree_at(
        lambda state: (state.mass_stack), nodes, (jnp.ones(9).astype(jnp.float32))
    )
    np.testing.assert_allclose(nodes.mass_stack, jnp.ones(9))

    nodes = nodes.refresh()

    np.testing.assert_allclose(nodes.mass_stack, jnp.zeros(9))
