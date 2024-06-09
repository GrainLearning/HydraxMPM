"""Unit tests for the linear shape functions."""

import jax.numpy as jnp

import pymudokon as pm
import numpy as np


def test_create():
    shapefunction = pm.CubicShapeFunction.create(num_particles=2, dim=2)

    assert isinstance(shapefunction, pm.CubicShapeFunction)

    np.testing.assert_allclose(shapefunction.intr_shapef, jnp.zeros((32,1,1), dtype=jnp.float32))

    np.testing.assert_allclose(
        shapefunction.intr_shapef_grad,
        jnp.zeros((32, 2,1), dtype=jnp.float32),
    )


def test_calculate_shapefunction():
    """Test the cubic shape function for top level container input."""
    positions = jnp.array([[0.25, 0.25], [0.3, 0.4]])
    particles = pm.Particles.create(positions=positions)

    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]),
        end=jnp.array([1.0, 1.0]),
        node_spacing=0.1
    )

    shapefunction = pm.CubicShapeFunction.create(num_particles=2, dim=2)

    nodes = shapefunction.set_boundary_nodes(nodes)


    shapefunction = shapefunction.calculate_shapefunction(nodes,particles)
        
  
