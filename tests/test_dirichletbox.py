"""Unit tests for the NodesContainer class."""

import jax.numpy as jnp

import pymudokon as pm


def test_init():
    """Unit test to initialize the NodesContainer class."""

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.array([1.0, 1.0, 1.0]), node_spacing=0.1)

    box = pm.DirichletBox.create(nodes)

    assert isinstance(box, pm.DirichletBox)


def test_apply_on_node_moments():
    """Unit test to initialize the NodesContainer class."""
    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.1)

    box = pm.DirichletBox.create()

    box.apply_on_nodes_moments(nodes)

    # 3D fix
    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.array([1.0, 1.0, 1.0]), node_spacing=0.1)

    box = pm.DirichletBox.create()

    box.apply_on_nodes_moments(nodes)

    assert isinstance(box, pm.DirichletBox)

test_init()
# test_apply_on_node_moments()