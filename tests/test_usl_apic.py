"""Unit tests for the APIC USL Solver."""

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to initialize usl solver."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.5,
        num_points=2,
        dt=0.001,
    )

    solver = hdx.USL_APIC(config)
    expected_Dp = jnp.array(
        [
            [0.08333334, 0.0, 0.0],
            [0.0, 0.08333334, 0.0],
            [0.0, 0.0, 0.08333334],
        ]
    )
    np.testing.assert_allclose(solver.Dp, expected_Dp, rtol=1e-3)

    assert isinstance(solver, hdx.USL_APIC)


def test_p2g_2d():
    """Unit test to perform particle-to-grid transfer for 2D."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.001,
        shapefunction="linear",
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
    )

    nodes = hdx.Nodes(config)

    nodes = nodes.get_interactions(particles.position_stack)

    solver = hdx.USL_APIC(config)

    nodes = solver.p2g(nodes=nodes, particles=particles)

    expected_mass_stack = jnp.array([0.27, 0.09, 0.03, 0.01])
    np.testing.assert_allclose(nodes.mass_stack, expected_mass_stack, rtol=1e-3)

    expected_node_moment_stack = jnp.array(
        [[0.27, 0.27], [0.09, 0.09], [0.03, 0.03], [0.01, 0.01]]
    )
    
    

    np.testing.assert_allclose(
        nodes.moment_stack, expected_node_moment_stack, rtol=1e-3
    )


def test_p2g_3d():
    """Unit test to perform particle-to-grid transfer in 3D."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.001,
        shapefunction="linear",
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
        velocity_stack=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    nodes = hdx.Nodes(config)

    usl = hdx.USL_APIC(config=config)

    @jax.jit
    def usl_p2g(usl, particles, nodes):
        nodes = nodes.get_interactions(particles.position_stack)
        nodes = usl.p2g(particles, nodes)
        return nodes

    nodes = usl_p2g(usl, particles, nodes)

    # note these values have not been verified analytically
    # expected_mass_stack = jnp.array(
    #     [0.189, 0.081, 0.063, 0.027, 0.02100001, 0.009, 0.007, 0.003]
    # )
    expected_mass_stack = jnp.array(
        [0.189, 0.081,0.021,  0.009, 0.063, 0.027,  0.007, 0.003]
    )
    
    np.testing.assert_allclose(nodes.mass_stack, expected_mass_stack, rtol=1e-3)

    expected_node_moment_stack = jnp.array(
        [
            [0.189, 0.189, 0.189],
            [0.081, 0.081, 0.081],
            [0.02099999, 0.02099999, 0.02099999],
            [0.009, 0.009, 0.009],
            [0.063, 0.063, 0.063],
            [0.027, 0.027, 0.027],
            [0.007, 0.007, 0.007],
            [0.003, 0.003, 0.003],
        ]
    )

    np.testing.assert_allclose(
        nodes.moment_stack, expected_node_moment_stack, rtol=1e-3
    )


def test_g2p_2d():
    """Unit test to perform grid-to-particle transfer for 2D."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    nodes = hdx.Nodes(config)

    usl = hdx.USL_APIC(config=config)

    @jax.jit
    def usl_p2g_g2p(usl, particles, nodes):
        nodes = nodes.get_interactions(particles.position_stack)
        nodes = usl.p2g(particles, nodes)
        particles, solver = usl.g2p(particles, nodes)
        return particles, solver

    particles, solver = usl_p2g_g2p(usl, particles, nodes)

    expected_volumes = jnp.array([0.6265, 0.358])

    np.testing.assert_allclose(particles.volume_stack, expected_volumes, rtol=1e-3)

    expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(particles.velocity_stack, expected_velocities, rtol=1e-3)

