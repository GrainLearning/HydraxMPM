import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestUSL(unittest.TestCase):
    @staticmethod
    def test_init():
        particles = pm.particles.init(positions=jnp.array([[1.0, 2.0], [0.3, 0.1]]))
        nodes = pm.nodes_container.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl = pm.usl_container.init(particles=particles, nodes=nodes, materials=material, alpha=0.1, dt=0.001)
        assert isinstance(usl, pm.usl_container.USLContainer)

    @staticmethod
    def test_p2g():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        nodes = pm.nodes_container.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        particles = particles._replace(
            masses_array=jnp.array([0.1, 0.3]),
            volumes_array=jnp.array([0.7, 0.4]),
            volumes_original_array=jnp.array([0.7, 0.4]),
        )

        material = pm.linearelastic_mat.init(E=0.1, nu=0.1, num_particles=2, dim=2)

        usl = pm.usl_container.init(particles=particles, nodes=nodes, materials=material, alpha=0.99, dt=0.1)

        shapefunction = pm.linear_shp.init(num_particles=2, dim=2)

        interactions = pm.interactions.init(num_particles=2, dim=2)

        interactions = pm.interactions.get_interactions(interactions, particles, nodes)

        shapefunction = jax.vmap(pm.linear_shp.linear_shapefunction, in_axes=(None, 0, None))(
            shapefunction, interactions.intr_dist_array, nodes.inv_node_spacing
        )

        nodes = pm.usl_container.p2g(
            usl=usl, nodes=nodes, particles=particles, shapefunction=shapefunction, interactions=interactions
        )

        expected_mass = jnp.array([0.27, 0.03, 0.09, 0.01])
        np.testing.assert_allclose(nodes.masses_array, expected_mass, rtol=1e-3)
        expected_node_moments = jnp.array([[0.27, 0.27], [0.03, 0.03], [0.09, 0.09], [0.01, 0.01]])
        np.testing.assert_allclose(nodes.moments_array, expected_node_moments, rtol=1e-3)

    @staticmethod
    def test_g2p():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        nodes = pm.nodes_container.init(
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

        material = pm.linearelastic_mat.init(E=0.1, nu=0.1, num_particles=2, dim=2)

        usl = pm.usl_container.init(particles=particles, nodes=nodes, materials=material, alpha=0.99, dt=0.1)

        shapefunction = pm.linear_shp.init(num_particles=2, dim=2)

        interactions = pm.interactions.init(num_particles=2, dim=2)

        interactions = pm.interactions.get_interactions(interactions, particles, nodes)

        shapefunction = jax.vmap(pm.linear_shp.linear_shapefunction, in_axes=(None, 0, None))(
            shapefunction, interactions.intr_dist_array, nodes.inv_node_spacing
        )

        nodes = pm.usl_container.p2g(
            usl=usl, nodes=nodes, particles=particles, shapefunction=shapefunction, interactions=interactions
        )

        particles = pm.usl_container.g2p(
            usl, nodes=nodes, particles=particles, shapefunction=shapefunction, interactions=interactions
        )

        expected_particle_volumes = jnp.array([0.4402222222222, 0.4402222222222])

        np.testing.assert_allclose(particles.volumes_array, expected_particle_volumes, rtol=1e-3)

    @staticmethod
    def test_update():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
            velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.nodes_container.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl = pm.usl_container.init(particles=particles, nodes=nodes, materials=material, alpha=0.1, dt=0.001)

        usl = pm.usl_container.update(usl)

    @staticmethod
    def test_solve():
        particles = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
            velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes = pm.nodes_container.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl = pm.usl_container.init(particles=particles, nodes=nodes, materials=material, alpha=0.9, dt=0.001)

        def some_callback(package):
            usl, step = package  # unused intentionally
            pass

        usl = pm.usl_container.solve(usl, num_steps=10, output_function=some_callback)


if __name__ == "__main__":
    # unittest.main()
    TestUSL.test_solve()
