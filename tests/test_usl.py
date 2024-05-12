import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestUSL(unittest.TestCase):
    @staticmethod
    def test_init():
        particles = pm.Particles.register(
            positions=jnp.array([[1.0, 2.0], [0.3, 0.1]])
        )
        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        shapefunctions = pm.LinearShapeFunction.register(
            2,4,2)
        
        material = pm.LinearIsotropicElastic.register(
            E=1000.0, nu=0.2, num_particles=2, dim=2
        )

        usl = pm.USL.register(
            particles=particles,
            nodes=nodes,
            shapefunctions= shapefunctions,
            materials=[material],
            alpha=0.1,
            dt=0.001,
        )

        assert isinstance(usl, pm.USL)

    @staticmethod
    def test_p2g():
        particles = pm.Particles.register(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        particles = particles.replace(
            masses=jnp.array([0.1, 0.3]),
            volumes=jnp.array([0.7, 0.4]),
            volumes_original=jnp.array([0.7, 0.4]),
            stresses=jnp.stack([jnp.eye(3)] * 2),
        )

        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        shapefunctions = pm.LinearShapeFunction.register(
            2,4,2)

        interactions = pm.Interactions.register(stencil, 2)
        
        interactions = interactions.get_interactions(
            particles, nodes
        )

        shapefunctions = shapefunctions.calculate_shapefunction(
            nodes, interactions
        )

        nodes = pm.solvers.usl_solver.p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            interactions=interactions,
            dt=0.1,
        )

        expected_mass = jnp.array([0.27, 0.03, 0.09, 0.01])
        np.testing.assert_allclose(nodes.masses, expected_mass, rtol=1e-3)

        expected_node_moments = jnp.array(
            [[0.27, 0.27], [0.03, 0.03], [0.09, 0.09], [0.01, 0.01]]
        )
        np.testing.assert_allclose(
            nodes.moments, expected_node_moments, rtol=1e-3
        )

    @staticmethod
    def test_g2p():
        particles = pm.Particles.register(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        particles = particles.replace(
            masses=jnp.array([0.1, 0.3]),
            volumes=jnp.array([0.7, 0.4]),
            volumes_original=jnp.array([0.7, 0.4]),
            stresses=jnp.stack([jnp.eye(3)] * 2),
        )

        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        stencil = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        shapefunctions = pm.LinearShapeFunction.register(2,4,2)

        interactions = pm.Interactions.register(stencil, 2)
        
        interactions = interactions.get_interactions(
            particles, nodes
        )

        shapefunctions = shapefunctions.calculate_shapefunction(
            nodes, interactions
        )

        nodes = pm.solvers.usl_solver.p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            interactions=interactions,
            dt=0.1,
        )

        particles = pm.solvers.usl_solver.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            interactions=interactions,
            alpha=0.99,
            dt=0.1
        )
        # TODO finish this test

        # expected_particle_volumes = jnp.array([0.4402222222222, 0.4402222222222])
        # expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])

        # print(f" volumes - got {particles.volumes}")
        # print(f" volumes - expected {expected_particle_volumes}")
        # print(f" velocities - got {particles.velocities}")
        # print(f" positions - got {particles.positions}")


        # print()
        # np.testing.assert_allclose(
        #     particles.volumes, expected_particle_volumes, rtol=1e-3
        # )


    @staticmethod
    def test_update():
        particles = pm.Particles.register(
            positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
            velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.2, num_particles=2, dim=2)
        
        shapefunctions = pm.LinearShapeFunction.register(2,4,2)
        
        usl = pm.USL.register(
            particles=particles,
            nodes=nodes,
            materials=[material],
            shapefunctions=shapefunctions,
            alpha=0.1,
            dt=0.001)

        usl = usl.update()

    @staticmethod
    def test_solve():
        particles = pm.Particles.register(
            positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
            velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.Nodes.register(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.2, num_particles=2, dim=2)

        shapefunctions = pm.LinearShapeFunction.register(2,4,2)
        
        usl = pm.USL.register(
            particles=particles,
            nodes=nodes,
            materials=[material],
            shapefunctions=shapefunctions,
            alpha=0.9, dt=0.001)

        def some_callback(package):
            usl, step = package  # unused intentionally
            pass

        usl = usl.solve(num_steps=10, output_function=some_callback)


if __name__ == "__main__":
    unittest.main()
