import unittest

import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestUSL(unittest.TestCase):
    @staticmethod
    def test_init():
        particles = pm.particles.init(
            positions=jnp.array([[1.0, 2.0], [0.3, 0.1]])
        )
        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(
            E=1000.0, nu=0.2, num_particles=2, dim=2
        )

        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.1,
            dt=0.001,
        )

        assert isinstance(usl, pm.usl.USLContainer)

    @staticmethod
    def test_p2g():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        particles = particles._replace(
            masses_array=jnp.array([0.1, 0.3]),
            volumes_array=jnp.array([0.7, 0.4]),
            volumes_original_array=jnp.array([0.7, 0.4]),
            stresses_array=jnp.stack([jnp.eye(3)] * 2),
        )

        material = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )

        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.99,
            dt=0.1,
        )

        interactions = pm.core.interactions.get_interactions(
            usl.interactions, particles, nodes
        )

        shapefunctions = pm.linear_shp.calculate_shapefunction(
            usl.shapefunctions, usl.nodes, interactions
        )

        nodes = pm.usl.p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            interactions=interactions,
            dt=usl.dt,
        )

        expected_mass = jnp.array([0.27, 0.03, 0.09, 0.01])
        np.testing.assert_allclose(nodes.masses_array, expected_mass, rtol=1e-3)

        expected_node_moments = jnp.array(
            [[0.27, 0.27], [0.03, 0.03], [0.09, 0.09], [0.01, 0.01]]
        )
        np.testing.assert_allclose(
            nodes.moments_array, expected_node_moments, rtol=1e-3
        )

    @staticmethod
    def test_g2p():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        particles = particles._replace(
            masses_array=jnp.array([0.1, 0.3]),
            volumes_array=jnp.array([0.7, 0.4]),
            volumes_original_array=jnp.array([0.7, 0.4]),
            stresses_array=jnp.stack([jnp.eye(3)] * 2),
        )

        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        material = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )
        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.99,
            dt=0.1,
        )

        material = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )

        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.99,
            dt=0.1,
        )

        interactions = pm.core.interactions.get_interactions(
            usl.interactions, particles, nodes
        )

        shapefunctions = pm.linear_shp.calculate_shapefunction(
            usl.shapefunctions, usl.nodes, interactions
        )

        nodes = pm.usl.p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            interactions=interactions,
            dt=usl.dt,
        )

        particles = pm.usl.g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            interactions=interactions,
            alpha=usl.alpha,
            dt=usl.dt,
        )

        expected_particle_volumes = jnp.array([0.4402222222222, 0.4402222222222])
        expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])

        print(f" volumes - got {particles.volumes_array}")
        print(f" volumes - expected {expected_particle_volumes}")
        print(f" velocities - got {particles.velocities_array}")
        print(f" positions - got {particles.positions_array}")


        print()
        # np.testing.assert_allclose(
        #     particles.volumes_array, expected_particle_volumes, rtol=1e-3
        # )


    @staticmethod
    def test_update():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
            velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.1,
            dt=0.001)

        usl = pm.usl.update(usl)

    @staticmethod
    def test_solve():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
            velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl = pm.usl.init(
            particles=particles,
            nodes=nodes,
            material=material,
            alpha=0.9, dt=0.001)

        def some_callback(package):
            usl, step = package  # unused intentionally
            pass

        usl = pm.usl.solve(usl, num_steps=10, output_function=some_callback)


if __name__ == "__main__":
    # unittest.main()
    TestUSL.test_g2p()
