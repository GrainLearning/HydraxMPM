"""Unit tests for the APIC USL Solver."""

import jax.numpy as jnp
import numpy as np
from regex import P

import hydraxmpm as hdx
import jax


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
        shapefunction=hdx.SHAPEFUNCTION.linear,
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
        shapefunction=hdx.SHAPEFUNCTION.linear,
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

    expected_volumes = jnp.array([0.49855555, 0.2848889])

    np.testing.assert_allclose(particles.volume_stack, expected_volumes, rtol=1e-3)

    expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(particles.velocities, expected_velocities, rtol=1e-3)


test_g2p_2d()


# #     expected_positions = jnp.array([[0.2, 0.35], [0.2, 0.35]])
# #     np.testing.assert_allclose(particles.positions, expected_positions, rtol=1e-3)

# #     expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])
# #     np.testing.assert_allclose(particles.velocities, expected_velocities, rtol=1e-3)

# #     expected_velgrads = jnp.array(
# #         [[[-1.944444, -1.944444], [-0.9333334, -0.9333334]], [[-1.944444, -1.944444], [-0.9333334, -0.9333334]]]
# #     )

# #     np.testing.assert_allclose(particles.velgrads, expected_velgrads, rtol=1e-3)
# #     expected_F = jnp.array(
# #         [[[0.8055556, -0.1944444], [-0.09333334, 0.90666664]], [[0.8055556, -0.1944444], [-0.09333334, 0.90666664]]]
# #     )

# #     np.testing.assert_allclose(particles.F, expected_F, rtol=1e-3)


# # def test_g2p_3d():
# #     """Unit test to perform grid to particle transfer in 3D."""
# #     particles = pm.Particles.create(
# #         positions=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
# #         velocities=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
# #     )

# #     nodes = pm.Nodes.create(
# #         origin=jnp.array([0.0, 0.0, 0.0]),
# #         end=jnp.array([1.0, 1.0, 1.0]),
# #         node_spacing=1.0,
# #     )

# #     particles = particles.replace(
# #         masses=jnp.array([0.1, 0.3]),
# #         volumes=jnp.array([0.7, 0.4]),
# #         volumes_original=jnp.array([0.7, 0.4]),
# #         stresses=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
# #     )

# #     shapefunctions = pm.LinearShapeFunction.create(2, 3)

# #     shapefunctions = shapefunctions.calculate_shapefunction(nodes, particles.positions)

# #     nodes = pm.solvers.usl.p2g(
# #         nodes=nodes,
# #         particles=particles,
# #         shapefunctions=shapefunctions,
# #         dt=0.1,
# #     )
# #     particles = pm.solvers.usl.g2p(
# #         particles=particles,
# #         nodes=nodes,
# #         shapefunctions=shapefunctions,
# #         alpha=0.99,
# #         dt=0.1,
# #     )

# #     expected_volumes = jnp.array([0.4402222222222, 0.25155553])

# #     np.testing.assert_allclose(particles.volumes[:1], expected_volumes[:1], rtol=1e-3)

# #     expected_velocities = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
# #     np.testing.assert_allclose(particles.velocities, expected_velocities, rtol=1e-3)

# #     expected_positions = jnp.array([[0.2, 0.35, 0.4], [0.2, 0.35, 0.4]])

# #     np.testing.assert_allclose(particles.positions, expected_positions, rtol=1e-3)

# #     expected_velgrads = jnp.array(
# #         [
# #             [
# #                 [-1.9444444444444446, -1.9444444444444446, -1.9444444444444446],
# #                 [-0.9333333333333332, -0.9333333333333332, -0.9333333333333332],
# #                 [-0.8333333333333333, -0.8333333333333333, -0.8333333333333333],
# #             ],
# #             [
# #                 [-1.9444444444444446, -1.9444444444444446, -1.9444444444444446],
# #                 [-0.9333333333333332, -0.9333333333333332, -0.9333333333333332],
# #                 [-0.8333333333333333, -0.8333333333333333, -0.8333333333333333],
# #             ],
# #         ]
# #     )

# #     np.testing.assert_allclose(particles.velgrads, expected_velgrads, rtol=1e-3)

# #     expected_F = jnp.array(
# #         [
# #             [
# #                 [0.8055556, -0.1944444, -0.1944444],
# #                 [-0.09333335, 0.90666664, -0.09333335],
# #                 [-0.08333334, -0.08333334, 0.9166667],
# #             ],
# #             [
# #                 [0.8055556, -0.1944444, -0.1944444],
# #                 [-0.09333335, 0.90666664, -0.09333335],
# #                 [-0.08333334, -0.08333334, 0.9166667],
# #             ],
# #         ]
# #     )

# #     np.testing.assert_allclose(particles.F, expected_F, rtol=1e-3)


# # def test_update():
# #     """Unit test to update the state of the USL solver."""
# #     particles = pm.Particles.create(
# #         positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
# #         velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
# #         volumes=jnp.array([1.0, 0.2]),
# #         masses=jnp.array([1.0, 3.0]),
# #     )

# #     nodes = pm.Nodes.create(
# #         origin=jnp.array([0.0, 0.0]),
# #         end=jnp.array([1.0, 1.0]),
# #         node_spacing=0.5,
# #     )

# #     shapefunctions = pm.LinearShapeFunction.create(2, 2)

# #     material = pm.Material()
# #     force = pm.Forces()

# #     usl = pm.USL.create(
# #         particles=particles,
# #         nodes=nodes,
# #         materials=[material],
# #         forces=[force],
# #         shapefunctions=shapefunctions,
# #         alpha=0.1,
# #         dt=0.001,
# #     )

# #     usl = usl.update()


# # def test_solve():
# #     """Unit test to solve the USL solver."""
# #     particles = pm.Particles.create(
# #         positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
# #         velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
# #         volumes=jnp.array([1.0, 0.2]),
# #         masses=jnp.array([1.0, 3.0]),
# #     )

# #     nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.5)

# #     material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.2, num_particles=2)

# #     shapefunctions = pm.LinearShapeFunction.create(2, 2)

# #     usl = pm.USL.create(
# #         particles=particles, nodes=nodes, materials=[material], shapefunctions=shapefunctions, alpha=0.9, dt=0.001
# #     )

# #     @jax.tree_util.Partial
# #     def some_callback(package):
# #         step, usl = package  # unused intentionally

# #     usl = usl.solve(num_steps=10, output_step=2, output_function=some_callback)


# test_create()
# test_p2g_2d()
# test_g2p_2d()
