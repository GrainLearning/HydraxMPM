"""Unit tests for the grid"""

import jax.numpy as jnp
import numpy as np


import hydraxmpm as hdx


def test_create():
    """Test two ways of initializing sovlver"""

    position_stack = jnp.array([[0.1, 0.2], [0.1, 0.2]])
    solver = hdx.MPMSolver(
        config=hdx.Config(total_time=1.0, dt=0.001, dim=2),
        material_points=hdx.MaterialPoints(position_stack=position_stack),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1),
    )


def test_get_point_grid_interactions():
    position_stack = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

    solver = hdx.MPMSolver(
        config=hdx.Config(total_time=1.0, dt=0.001, shapefunction="linear", dim=2),
        material_points=hdx.MaterialPoints(position_stack=position_stack),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.5),
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    solver = solver._get_particle_grid_interactions_batched()

    np.testing.assert_allclose(
        solver._intr_dist_stack,  # normalized at grid
        jnp.array(
            [
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.3, -0.4, -0.0],
                [-0.3, 0.1, -0.0],
                [0.2, -0.4, -0.0],
                [0.2, 0.1, -0.0],
            ]
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(solver._intr_hash_stack.shape, (12))
    np.testing.assert_allclose(
        solver._intr_hash_stack,
        jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 3, 4, 6, 7]).astype(jnp.uint32),
    )


# test if shapefunction and grid connectivity are correct when included in vmap
def test_vmap_get_interactions_scatter():
    position_stack = jnp.array([[0.25, 0.25], [0.25, 0.25], [0.8, 0.4]])

    solver = hdx.MPMSolver(
        config=hdx.Config(total_time=1.0, dt=0.001, shapefunction="linear", dim=2),
        material_points=hdx.MaterialPoints(position_stack=position_stack),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.5),
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    def p2g(point_id, shapef, shapef_grad_padded, intr_dist_padded):
        return 1.0

    solver, X_stack = solver.vmap_interactions_and_scatter(p2g)

    np.testing.assert_allclose(
        solver._intr_dist_stack,  # normalized at grid
        jnp.array(
            [
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.25, -0.25, -0.0],
                [-0.25, 0.25, -0.0],
                [0.25, -0.25, -0.0],
                [0.25, 0.25, -0.0],
                [-0.3, -0.4, -0.0],
                [-0.3, 0.1, -0.0],
                [0.2, -0.4, -0.0],
                [0.2, 0.1, -0.0],
            ]
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(solver._intr_hash_stack.shape, (12))
    np.testing.assert_allclose(
        solver._intr_hash_stack, jnp.array([0, 1, 3, 4, 0, 1, 3, 4, 3, 4, 6, 7])
    )


def test_map_p2g():
    position_stack = jnp.array([[0.45, 0.25], [0.25, 0.25], [0.8, 0.4]])

    stress = jnp.eye(3) * -100

    stress_stack = jnp.array([stress, stress / 3, stress / 2])
    solver = hdx.MPMSolver(
        config=hdx.Config(total_time=1.0, dt=0.001, shapefunction="linear", dim=2),
        material_points=hdx.MaterialPoints(
            position_stack=position_stack, stress_stack=stress_stack, density_ref=10.0
        ),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.5),
    )

    solver = solver.setup()

    p2g_pressure_stack = solver.map_p2g("p_stack")

    p2g2p_pressure_stack = solver.map_p2g2g("p_stack")

    p2g_p_stack = solver.p2g_p_stack
    p2g_q_vm_stack = solver.p2g_q_vm_stack

    p2g_q_p_stack = solver.p2g_q_p_stack

    p2g_dgamma_dt_stack = solver.p2g_dgamma_dt_stack
    p2g_gamma_stack = solver.p2g_gamma_stack

    # pass
