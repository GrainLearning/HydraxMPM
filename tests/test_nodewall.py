"""Unit tests for the node wall class."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_init():
    """Unit test to initialize the node wall class."""
    wall = pm.NodeWall.create(wall_type=0, wall_dim=0, node_id_stack=jnp.array([0, 1, 2]))

    assert isinstance(wall, pm.NodeWall)


def test_apply_on_node_moments():
    """Unit to test update of NodeWalls."""
    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5
    )

    wall = pm.NodeWall.create(wall_type=0, wall_dim=0, node_id_stack=jnp.array([0, 1, 2]))

    # fixed all type 0
    nodes = nodes.replace(moment_nt_stack=jnp.ones((nodes.num_nodes_total, 2)))
    nodes, wall = wall.apply_on_nodes_moments(nodes)

    expected_moment_nt_stack = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)

    # fix in a dimension type 1
    wall = pm.NodeWall.create(wall_type=1, wall_dim=0, node_id_stack=jnp.array([0, 1, 2]))
    nodes = nodes.replace(moment_nt_stack=jnp.ones((nodes.num_nodes_total, 2)))
    nodes, wall = wall.apply_on_nodes_moments(nodes)

    expected_moment_nt_stack = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)

    # slip min in a dimension type 2
    wall = pm.NodeWall.create(wall_type=2, wall_dim=0, node_id_stack=jnp.array([0, 1, 2]))

    nodes = nodes.replace(
        moment_nt_stack=jnp.zeros((nodes.num_nodes_total, 2))
        .at[0, :]
        .set(-1)
        .at[1, :]
        .set(1)
    )

    nodes, wall = wall.apply_on_nodes_moments(nodes)

    expected_moment_nt_stack = jnp.array(
        [
            [-1, -1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)
    # slip max in a dimension type 3
    wall = pm.NodeWall.create(wall_type=3, wall_dim=0, node_id_stack=jnp.array([0, 1, 2]))

    nodes = nodes.replace(
        moment_nt_stack=jnp.zeros((nodes.num_nodes_total, 2))
        .at[0, :]
        .set(-1)
        .at[1, :]
        .set(1)
    )

    nodes, wall = wall.apply_on_nodes_moments(nodes)

    expected_moment_nt_stack = jnp.array(
        [
            [0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)
