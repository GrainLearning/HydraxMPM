"""Unit tests for rigid particles."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

def test_create():
    """Unit test to initialize the rigid particles class."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.1,
        num_points=2,
        shapefunction="linear",
    )

    box = hdx.RigidParticles(
        config,
        position_stack=jnp.array([[0.45, 0.21], [0.8, 0.4]]),
        velocity_stack=jnp.array([[0.0, 0.0], [0.0, 0.0]]),
    )

    assert isinstance(box, hdx.RigidParticles)


def test_call_2d():
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.5,
        num_points=2,
        shapefunction="linear",
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.array([[0.2, 0.7]]),
        mass_stack=jnp.array([1.0,1.0]),
    )

    nodes = hdx.Nodes(config)
    
    usl = hdx.USL(config)
    
    nodes = usl.p2g(
        particles,
        nodes
    )

    rigid_particles = hdx.RigidParticles(
        config=config,
        position_stack=jnp.array([[0.7, 0.2]]),
        velocity_stack=jnp.array([[0.0, 0.0]]),
    )

    nodes = eqx.tree_at(
        lambda state: (
            state.mass_stack,
            state.moment_nt_stack,
        ),
        nodes,
        (nodes.mass_stack.at[:].set(1.0), nodes.moment_nt_stack.at[:].set(1.0)),
    )

    nodes, rigid_particles = rigid_particles.apply_on_nodes(nodes, particles, 0)

    expected_moment_nt_stack = jnp.array(
        [
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
            [0.76923066, 1.1538463],
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
            ],
        ]
    )
    np.testing.assert_allclose(nodes.moment_nt_stack, expected_moment_nt_stack)

