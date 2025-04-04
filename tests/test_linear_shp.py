"""Unit tests for the linear shape functions."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Test creation of cubic shape function."""

    solver = hdx.MPMSolver(
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.2], [0.1, 0.2]])
        ),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1),
        shapefunction="linear",
        dt=0.001,
        dim=2,
    )

    np.testing.assert_allclose(solver.shape_map._intr_shapef_stack, jnp.zeros((2 * 4)))

    np.testing.assert_allclose(
        solver.shape_map._intr_shapef_grad_stack,
        jnp.zeros((2 * 4, 3), dtype=jnp.float32),
    )


def test_calc_shp_2d():
    """Test the linear shape function for top level container input."""

    solver = hdx.MPMSolver(
        material_points=hdx.MaterialPoints(jnp.array([[0.45, 0.21], [0.8, 0.4]])),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1),
        shapefunction="linear",
        dt=0.001,
        dim=2,
    )

    shape_map = solver.shape_map._get_particle_grid_interactions_batched(
        solver.material_points, solver.grid
    )

    expected_shapef_stack = jnp.array(
        [0.45000005, 0.04999995, 0.45000005, 0.04999995, 1.0, 0.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(
        expected_shapef_stack, shape_map._intr_shapef_stack, rtol=1e-4
    )

    np.testing.assert_allclose(
        jnp.prod(shape_map._intr_shapef_stack.flatten(), axis=0), 0
    )

    expected_shapef_grad_stack = jnp.array(
        [
            [-9.000001, -5.0, 0.0],
            [-0.99999905, 5.0, 0.0],
            [9.000001, -5.0, 0.0],
            [0.99999905, 5.0, 0.0],
            [-0.0, -0.0, 0.0],
            [-0.0, 0.0, 0.0],
            [0.0, -0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        expected_shapef_grad_stack, shape_map._intr_shapef_grad_stack
    )


def test_calc_shp_3d():
    """Test the linear shape function for top level container input."""

    solver = hdx.MPMSolver(
        material_points=hdx.MaterialPoints(
            jnp.array([[0.45, 0.21, 0.1], [0.8, 0.4, 0.1]])
        ),
        grid=hdx.Grid(origin=[0.0, 0.0, 0.0], end=[1.0, 1.0, 1.0], cell_size=0.1),
        shapefunction="linear",
        dim=3,
    )

    shape_map = solver.shape_map._get_particle_grid_interactions_batched(
        solver.material_points, solver.grid
    )

    expected_shapef_stack = jnp.array(
        [
            0.45,
            0.05,
            0.45,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        expected_shapef_stack, shape_map._intr_shapef_stack, rtol=1e-5
    )
    np.testing.assert_allclose(
        jnp.prod(shape_map._intr_shapef_stack.flatten(), axis=0), 0
    )

    expected_shapef_grad_stack = jnp.array(
        [
            [
                -9.0,
                -5.0,
                0.0,
            ],
            [
                -1.0,
                5.0,
                0.0,
            ],
            [
                9.0,
                -5.0,
                0.0,
            ],
            [
                1.0,
                5.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    np.testing.assert_allclose(
        expected_shapef_grad_stack, shape_map._intr_shapef_grad_stack, rtol=1e-4
    )
