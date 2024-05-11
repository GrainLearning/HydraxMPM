import unittest

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


class TestUSL(unittest.TestCase):
    @staticmethod
    def test_init():
        particles_state = pm.particles.init(
            positions=jnp.array([[1.0, 2.0], [0.3, 0.1]])
        )
        nodes_state = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material_state = pm.linearelastic_mat.init(
            E=1000.0, nu=0.2, num_particles=2, dim=2
        )

        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.1,
            dt=0.001,
        )

        assert isinstance(usl_state, pm.usl.USLContainer)

    @staticmethod
    def test_p2g():
        particles_state = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        nodes_state = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        particles_state = particles_state._replace(
            masses_array=jnp.array([0.1, 0.3]),
            volumes_array=jnp.array([0.7, 0.4]),
            volumes_original_array=jnp.array([0.7, 0.4]),
            stresses_array=jnp.stack([jnp.eye(3)] * 2),
        )

        material_state = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )

        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.99,
            dt=0.1,
        )

        interactions_state = pm.core.interactions.get_interactions(
            usl_state.interactions_state, particles_state, nodes_state
        )

        shapefunctions_state = pm.linear_shp.calculate_shapefunction(
            usl_state.shapefunctions_state, usl_state.nodes_state, interactions_state
        )

        nodes_state = pm.usl.p2g(
            nodes_state=nodes_state,
            particles_state=particles_state,
            shapefunctions_state=shapefunctions_state,
            interactions_state=interactions_state,
            dt=usl_state.dt,
        )

        expected_mass = jnp.array([0.27, 0.03, 0.09, 0.01])
        np.testing.assert_allclose(nodes_state.masses_array, expected_mass, rtol=1e-3)

        expected_node_moments = jnp.array(
            [[0.27, 0.27], [0.03, 0.03], [0.09, 0.09], [0.01, 0.01]]
        )
        np.testing.assert_allclose(
            nodes_state.moments_array, expected_node_moments, rtol=1e-3
        )

    @staticmethod
    def test_g2p():
        particles_state = pm.particles.init(
            positions=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocities=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        particles_state = particles_state._replace(
            masses_array=jnp.array([0.1, 0.3]),
            volumes_array=jnp.array([0.7, 0.4]),
            volumes_original_array=jnp.array([0.7, 0.4]),
            stresses_array=jnp.stack([jnp.eye(3)] * 2),
        )

        nodes_state = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=1.0,
            particles_per_cell=1,
        )

        material_state = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )
        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.99,
            dt=0.1,
        )

        material_state = pm.linearelastic_mat.init(
            E=0.1, nu=0.1, num_particles=2, dim=2
        )

        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.99,
            dt=0.1,
        )

        interactions_state = pm.core.interactions.get_interactions(
            usl_state.interactions_state, particles_state, nodes_state
        )

        shapefunctions_state = pm.linear_shp.calculate_shapefunction(
            usl_state.shapefunctions_state, usl_state.nodes_state, interactions_state
        )

        nodes_state = pm.usl.p2g(
            nodes_state=nodes_state,
            particles_state=particles_state,
            shapefunctions_state=shapefunctions_state,
            interactions_state=interactions_state,
            dt=usl_state.dt,
        )

        particles_state = pm.usl.g2p(
            particles_state=particles_state,
            nodes_state=nodes_state,
            shapefunctions_state=shapefunctions_state,
            interactions_state=interactions_state,
            alpha=usl_state.alpha,
            dt=usl_state.dt,
        )
        # ...
        # expected_particle_volumes = jnp.array([0.4402222222222, 0.4402222222222])

        # np.testing.assert_allclose(
        #     particles_state.volumes_array, expected_particle_volumes, rtol=1e-3
        # )


    @staticmethod
    def test_update():
        particles_state = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
            velocities=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes_state = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material_state = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.1,
            dt=0.001)

        usl_state = pm.usl.update(usl_state)

    @staticmethod
    def test_solve():
        particles_state = pm.particles.init(
            positions=jnp.array([[0.1, 0.1], [0.5, 0.1]]),
            velocities=jnp.array([[0.1, 0.1], [0.2, 0.2]]),
            volumes=jnp.array([1.0, 0.2]),
            masses=jnp.array([1.0, 3.0]),
        )

        nodes_state = pm.nodes.init(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([1.0, 1.0]),
            node_spacing=0.5,
            particles_per_cell=2,
        )

        material_state = pm.linearelastic_mat.init(E=1000.0, nu=0.2, num_particles=2, dim=2)

        usl_state = pm.usl.init(
            particles_state=particles_state,
            nodes_state=nodes_state,
            material_state=material_state,
            alpha=0.9, dt=0.001)

        def some_callback(package):
            usl_state, step = package  # unused intentionally
            pass

        usl = pm.usl.solve(usl_state, num_steps=10, output_function=some_callback)


# if __name__ == "__main__":
#     # unittest.main()
#     TestUSL.test_solve()
