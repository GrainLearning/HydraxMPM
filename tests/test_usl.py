"""Unit tests for the USL Solver."""

import jax.numpy as jnp
import numpy as np
import jax
import pymudokon as pm


def test_create():
    """Unit test to initialize usl solver."""
    particles = pm.Particles.create(positions=jnp.array([[1.0, 2.0], [0.3, 0.1]]))
    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

    shapefunctions = pm.LinearShapeFunction.create(2, 2)

    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)

    usl = pm.USL.create(
        particles=particles,
        nodes=nodes,
        shapefunctions=shapefunctions,
        materials=[material],
        alpha=0.1,
        dt=0.001,
    )

    assert isinstance(usl, pm.USL)


def test_p2g():
    """Unit test to perform particle-to-grid transfer."""
    # 2D
    particles = pm.Particles.create(
        positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]),
        end=jnp.array([1.0, 1.0]),
        node_spacing=1.0,
    )

    particles = particles.replace(
        masses=jnp.array([0.1, 0.3]),
        volumes=jnp.array([0.7, 0.4]),
        volumes_original=jnp.array([0.7, 0.4]),
        stresses=jnp.stack([jnp.eye(3)] * 2),
    )

    shapefunctions = pm.LinearShapeFunction.create(2, 2)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles)

    nodes = pm.solvers.usl.p2g(
        nodes=nodes,
        particles=particles,
        shapefunctions=shapefunctions,
        dt=0.1,
    )

    expected_mass = jnp.array([0.27, 0.03, 0.09, 0.01])
    np.testing.assert_allclose(nodes.masses, expected_mass, rtol=1e-3)

    expected_node_moments = jnp.array([[0.27, 0.27], [0.03, 0.03], [0.09, 0.09], [0.01, 0.01]])
    np.testing.assert_allclose(nodes.moments, expected_node_moments, rtol=1e-3)

def test_p2g_3D():

    particles = pm.Particles.create(
        positions=jnp.array([[0.1, 0.25, 0.25], [0.1, 0.25, 0.6]]),
        velocities=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )

    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0, 0.0]),
        end=jnp.array([1.0, 1.0, 1.0]),
        node_spacing=1.0,
    )

    particles = particles.replace(
        masses=jnp.array([0.1, 0.3]),
        volumes=jnp.array([0.7, 0.4]),
        volumes_original=jnp.array([0.7, 0.4]),
        stresses=jnp.stack([jnp.eye(3)] * 2),
    )

    shapefunctions = pm.LinearShapeFunction.create(2, 3)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles)

    nodes = pm.solvers.usl.p2g(
        nodes=nodes,
        particles=particles,
        shapefunctions=shapefunctions,
        dt=0.1,
    )
    # note these values have not been verified analytically
    expected_mass = jnp.array([0.13162498, 0.014625, 0.04387499, 0.004875, 0.13837501, 0.015375, 0.046125, 0.005125])
    expected_node_moments = jnp.array(
        [[0.13162498, 0.13162498, 0.13162498],
        [0.014625,   0.014625,   0.014625  ],
        [0.04387499, 0.04387499, 0.04387499],
        [0.004875,   0.004875,   0.004875  ],
        [0.13837501, 0.13837501, 0.13837501],
        [0.015375,   0.015375,   0.015375  ],
        [0.046125,   0.046125,   0.046125  ],
        [0.005125,   0.005125,   0.005125  ]]
    )

    np.testing.assert_allclose(nodes.masses, expected_mass, rtol=1e-3)

    np.testing.assert_allclose(nodes.moments, expected_node_moments, rtol=1e-3)

def test_g2p():
    """Unit test to perform grid-to-particle transfer."""
    particles = pm.Particles.create(
        positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    particles = particles.replace(
        masses=jnp.array([0.1, 0.3]),
        volumes=jnp.array([0.7, 0.4]),
        volumes_original=jnp.array([0.7, 0.4]),
        stresses=jnp.stack([jnp.eye(3)] * 2),
    )

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=1.0)

    shapefunctions = pm.LinearShapeFunction.create(2, 2)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles)

    nodes = pm.solvers.usl.p2g(
        nodes=nodes,
        particles=particles,
        shapefunctions=shapefunctions,
        dt=0.1,
    )

    particles = pm.solvers.usl.g2p(
        particles=particles,
        nodes=nodes,
        shapefunctions=shapefunctions,
        alpha=0.99,
        dt=0.1,
    )

    # # expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    # # expected_velgrads = jnp.array(
    # #     [
    # #         [[-3.0555546e00, -2.9802322e-08], [-1.1175871e-08, -1.4666667e00]],
    # #         [[-3.0555546e00, -2.9802322e-08], [-1.1175871e-08, -1.4666667e00]],
    # #     ]
    # # )
    # # expected_F = jnp.array(
    # #     [
    # #         [[6.9444454e-01, -2.9802323e-09], [-1.1175871e-09, 8.5333335e-01]],
    # #         [[6.9444454e-01, -2.9802323e-09], [-1.1175871e-09, 8.5333335e-01]],
    # #     ]
    # # )
    # # expected_volumes = jnp.array([0.41481486, 0.23703706])
    # # expected_positions = jnp.array([[0.2, 0.35], [0.2, 0.35]])

    # # np.testing.assert_allclose(particles.velocities, expected_velocities, rtol=1e-3)
    # # np.testing.assert_allclose(particles.velgrads, expected_velgrads, rtol=1e-3)
    # # np.testing.assert_allclose(particles.F, expected_F, rtol=1e-3)
    # # np.testing.assert_allclose(particles.volumes, expected_volumes, rtol=1e-3)
    # # np.testing.assert_allclose(particles.positions, expected_positions, rtol=1e-3)

    # #3D
    # particles = pm.Particles.create(
    #     positions=jnp.array([[0.1, 0.25, 0.25], [0.1, 0.25, 0.6]]),
    #     velocities=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    # )

    # particles = particles.replace(
    #     masses=jnp.array([0.1, 0.3]),
    #     volumes=jnp.array([0.7, 0.4]),
    #     volumes_original=jnp.array([0.7, 0.4]),
    #     stresses=jnp.stack([jnp.eye(3)] * 2),
    # )

    # nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.array([1.0, 1.0, 1.0]), node_spacing=1.0)

    # shapefunctions = pm.LinearShapeFunction.create(2, 3)

    # shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles)

    # nodes = pm.solvers.usl.p2g(
    #     nodes=nodes,
    #     particles=particles,
    #     shapefunctions=shapefunctions,
    #     dt=0.1,
    # )

    # particles = pm.solvers.usl.g2p(
    #     particles=particles,
    #     nodes=nodes,
    #     shapefunctions=shapefunctions,
    #     alpha=0.99,
    #     dt=0.1,
    # )
    # # TODO add checks


def test_update():
    """Unit test to update the state of the USL solver."""
    particles = pm.Particles.create(
        positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
        velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
        volumes=jnp.array([1.0, 0.2]),
        masses=jnp.array([1.0, 3.0]),
    )

    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]),
        end=jnp.array([1.0, 1.0]),
        node_spacing=0.5,
    )

    shapefunctions = pm.LinearShapeFunction.create(2, 2)

    material = pm.Material()
    force = pm.Forces()

    usl = pm.USL.create(
        particles=particles,
        nodes=nodes,
        materials=[material],
        forces=[force],
        shapefunctions=shapefunctions,
        alpha=0.1,
        dt=0.001,
    )

    usl = usl.update()


def test_solve():
    """Unit test to solve the USL solver."""
    particles = pm.Particles.create(
        positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
        velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
        volumes=jnp.array([1.0, 0.2]),
        masses=jnp.array([1.0, 3.0]),
    )

    nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

    material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2, dim=2)

    shapefunctions = pm.LinearShapeFunction.create(2, 2)

    usl = pm.USL.create(
        particles=particles, nodes=nodes, materials=[material], shapefunctions=shapefunctions, alpha=0.9, dt=0.001
    )
    
    @jax.tree_util.Partial
    def some_callback(package):
        step, usl = package  # unused intentionally

    usl = usl.solve(num_steps=10,output_step=2, output_function=some_callback)

test_g2p()