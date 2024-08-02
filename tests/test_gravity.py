"""Unit tests for the Gravity data class."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test to initialize gravity."""
    box = pm.Gravity.create(jnp.array([0.0, 0.0]))

    assert isinstance(box, pm.Gravity)


def test_apply_on_node_moments_2d():
    """Unit test to apply gravity force on Nodes."""
    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5
    )

    grav = pm.Gravity.create(jnp.array([0.0, 9.8]))

    nodes = nodes.replace(
        mass_stack=jnp.ones(nodes.num_nodes_total) * 1.0,
        moment_stack=jnp.zeros((nodes.num_nodes_total, 2)),
        moment_nt_stack=jnp.ones((nodes.num_nodes_total, 2)),
    )

    nodes, grav = grav.apply_on_nodes_moments(nodes=nodes, dt=0.01)

    expected_moment_nt_stack = jnp.array(
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

    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack, rtol=1e-3)


def test_apply_on_node_moments_3d():
    """Unit test to apply gravity force on Nodes in 3D."""
    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0, 0.0]),
        end=jnp.array([1.0, 1.0, 1.0]),
        node_spacing=0.5,
    )

    grav = pm.Gravity.create(jnp.array([0.0, 0.0, 9.8]))

    nodes = nodes.replace(
        mass_stack=jnp.ones(nodes.num_nodes_total) * 1.0,
        moment_stack=jnp.zeros((nodes.num_nodes_total, 3)),
        moment_nt_stack=jnp.ones((nodes.num_nodes_total, 3)),
    )

    nodes, grav = grav.apply_on_nodes_moments(nodes=nodes, dt=0.01)

    expected_moment_nt_stack = np.repeat(jnp.array([[1.0, 1.0, 1.098]]), 27, axis=0)

    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack, rtol=1e-3)
