"""Unit tests for the USL Solver."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


def test_create():
    """Unit test to initialize usl solver."""

    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=1,
        dt=0.001,
    )
    usl = pm.USL(config, alpha=0.1)
    assert isinstance(usl, pm.USL)


def test_p2g_2d():
    """Unit test to perform particle-to-grid transfer for 2D."""
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=1.0,
        num_points=2,
        dt=0.001,
    )

    particles = pm.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
    )

    nodes = pm.Nodes(config)

    shapefunction = pm.LinearShapeFunction(config)

    grid = pm.GridStencilMap(config)

    grid = grid.partition(particles.position_stack)

    shapefunction = shapefunction.get_shapefunctions(grid=grid, particles=particles)

    usl = pm.USL(config=config, alpha=0.99)

    nodes = usl.p2g(
        particles=particles, nodes=nodes, shapefunctions=shapefunction, grid=grid
    )

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
    config = pm.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.001,
    )

    particles = pm.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
        velocity_stack=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    nodes = pm.Nodes(config)

    shapefunction = pm.LinearShapeFunction(config)

    grid = pm.GridStencilMap(config)

    grid = grid.partition(particles.position_stack)

    shapefunction = shapefunction.get_shapefunctions(grid=grid, particles=particles)

    usl = pm.USL(config=config, alpha=0.99)

    nodes = usl.p2g(
        particles=particles, nodes=nodes, shapefunctions=shapefunction, grid=grid
    )

    # note these values have not been verified analytically
    expected_mass_stack = jnp.array(
        [0.189, 0.081, 0.063, 0.027, 0.02100001, 0.009, 0.007, 0.003]
    )
    np.testing.assert_allclose(nodes.mass_stack, expected_mass_stack, rtol=1e-3)
    expected_node_moment_stack = jnp.array(
        [
            [0.189, 0.189, 0.189],
            [0.081, 0.081, 0.081],
            [0.063, 0.063, 0.063],
            [0.027, 0.027, 0.027],
            [0.02099999, 0.02099999, 0.02099999],
            [0.009, 0.009, 0.009],
            [0.007, 0.007, 0.007],
            [0.003, 0.003, 0.003],
        ]
    )

    np.testing.assert_allclose(
        nodes.moment_stack, expected_node_moment_stack, rtol=1e-3
    )


def test_g2p_2d():
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    particles = pm.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    nodes = pm.Nodes(config)

    shapefunction = pm.LinearShapeFunction(config)

    grid = pm.GridStencilMap(config)

    grid = grid.partition(particles.position_stack)

    shapefunction = shapefunction.get_shapefunctions(grid=grid, particles=particles)

    usl = pm.USL(config=config, alpha=0.99)

    nodes = usl.p2g(
        nodes=nodes, particles=particles, shapefunctions=shapefunction, grid=grid
    )
    particles = usl.g2p(
        particles=particles, nodes=nodes, shapefunctions=shapefunction, grid=grid
    )

    expected_volume_stack = jnp.array([0.49855555, 0.2848889])

    np.testing.assert_allclose(particles.volume_stack, expected_volume_stack, rtol=1e-3)

    expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        particles.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_position_stack = jnp.array([[0.2, 0.35], [0.2, 0.35]])
    np.testing.assert_allclose(
        particles.position_stack, expected_position_stack, rtol=1e-3
    )

    expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        particles.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_L_stack = jnp.array(
        [
            [
                [-1.944444, -1.944444, 0.0],
                [-0.9333334, -0.9333334, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [-1.944444, -1.944444, 0.0],
                [-0.9333334, -0.9333334, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )

    np.testing.assert_allclose(particles.L_stack, expected_L_stack, rtol=1e-3)

    expected_F_stack = jnp.array(
        [
            [
                [0.8055556, -0.1944444, 0.0],
                [-0.09333334, 0.90666664, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.8055556, -0.1944444, 0.0],
                [-0.09333334, 0.90666664, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
    )

    np.testing.assert_allclose(particles.F_stack, expected_F_stack, rtol=1e-3)


def test_g2p_3d():
    config = pm.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=1.0,
        num_points=2,
        dt=0.1,
    )

    particles = pm.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
        velocity_stack=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    nodes = pm.Nodes(config)

    shapefunction = pm.LinearShapeFunction(config)

    grid = pm.GridStencilMap(config)

    grid = grid.partition(particles.position_stack)

    shapefunction = shapefunction.get_shapefunctions(grid=grid, particles=particles)

    usl = pm.USL(config=config, alpha=0.99)

    nodes = usl.p2g(
        particles=particles, nodes=nodes, shapefunctions=shapefunction, grid=grid
    )
    particles = usl.g2p(
        particles=particles, nodes=nodes, shapefunctions=shapefunction, grid=grid
    )

    expected_volume_stack = jnp.array([0.4402222222222, 0.25155553])

    np.testing.assert_allclose(
        particles.volume_stack[:1], expected_volume_stack[:1], rtol=1e-3
    )
    expected_velocity_stack = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(
        particles.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_position_stack = jnp.array([[0.2, 0.35, 0.4], [0.2, 0.35, 0.4]])

    np.testing.assert_allclose(
        particles.position_stack, expected_position_stack, rtol=1e-3
    )

    expected_L_stack = jnp.array(
        [
            [
                [-1.9444444444444446, -1.9444444444444446, -1.9444444444444446],
                [-0.9333333333333332, -0.9333333333333332, -0.9333333333333332],
                [-0.8333333333333333, -0.8333333333333333, -0.8333333333333333],
            ],
            [
                [-1.9444444444444446, -1.9444444444444446, -1.9444444444444446],
                [-0.9333333333333332, -0.9333333333333332, -0.9333333333333332],
                [-0.8333333333333333, -0.8333333333333333, -0.8333333333333333],
            ],
        ]
    )

    np.testing.assert_allclose(particles.L_stack, expected_L_stack, rtol=1e-3)

    expected_F_stack = jnp.array(
        [
            [
                [0.8055556, -0.1944444, -0.1944444],
                [-0.09333335, 0.90666664, -0.09333335],
                [-0.08333334, -0.08333334, 0.9166667],
            ],
            [
                [0.8055556, -0.1944444, -0.1944444],
                [-0.09333335, 0.90666664, -0.09333335],
                [-0.08333334, -0.08333334, 0.9166667],
            ],
        ]
    )

    np.testing.assert_allclose(particles.F_stack, expected_F_stack, rtol=1e-3)


def test_update():
    """Unit test to update the state of the USL solver."""

    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.5,
        num_points=2,
        dt=0.001,
    )

    particles = pm.Particles(
        config=config,
        position_stack=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
        velocity_stack=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
        volume_stack=jnp.array([1.0, 0.2]),
        mass_stack=jnp.array([1.0, 3.0]),
    )

    nodes = pm.Nodes(config)

    shapefunctions = pm.LinearShapeFunction(config)

    usl = pm.USL(config=config, alpha=0.1)
    grid = pm.GridStencilMap(config)

    usl, particles, nodes, shapefunctions, grid, [], [] = usl.update(
        particles=particles,
        nodes=nodes,
        shapefunctions=shapefunctions,
        grid=grid,
        material_stack=(),
        forces_stack=(),
    )


pm.set_default_gpu(1)

test_update()
