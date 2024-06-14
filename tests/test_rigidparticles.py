
"""Unit tests for the NodesContainer class."""

import jax.numpy as jnp

import pymudokon as pm

import numpy as np

def test_init():
    """Unit test to initialize the NodesContainer class."""
    # nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.array([1.0, 1.0, 1.0]), node_spacing=0.1)

    rigid_particles = pm.RigidParticles.create(
        positions=jnp.array([[0.45, 0.21], [0.8, 0.4]]),
        velocities=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        shapefunction=pm.CubicShapeFunction.create(num_particles=2, dim=2)
    )

    assert isinstance(rigid_particles, pm.RigidParticles)


def test_apply_on_node_moments():
    """Unit test to initialize the NodesContainer class."""
    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)



    particles=pm.Particles.create(
            positions=jnp.array([[0.2, 0.7]]),
            masses=jnp.array([1.0])
    )


    shapefunctions = pm.LinearShapeFunction.create(num_particles=1, dim=2)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles.positions)


    r_shapefunctions = pm.LinearShapeFunction.create(num_particles=1, dim=2)

    rigid_particles = pm.RigidParticles.create(
        positions=jnp.array([[0.7, 0.2]]),
        velocities=jnp.array([[0.0, 0.0]]),
        shapefunction=r_shapefunctions
    )
    nodes = nodes.replace(
        moments_nt = jnp.ones(nodes.moments_nt.shape, dtype=jnp.float32),
        masses = jnp.ones(nodes.masses.shape, dtype=jnp.float32)
    )
    nodes, rigid_particles = rigid_particles.apply_on_nodes_moments(
        nodes,
        particles,
        shapefunctions
    )
    expected_moments_nt = jnp.array([
        [1.,         1.,        ],
        [1.,         1.,        ],
        [1.,         1.,        ],
        [1.,         1.,        ],
        [0.76923066, 1.1538463  ],
        [1.,         1.,        ],
        [1.,         1.,        ],
        [1.,         1.,        ],
        [1.,         1.,        ]]
    )
    np.testing.assert_allclose(nodes.moments_nt,expected_moments_nt)


test_init()
test_apply_on_node_moments()
