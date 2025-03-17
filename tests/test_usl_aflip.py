"""Unit tests for the APIC USL Solver."""

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


# test initialize
def test_create():
    position_stack = jnp.array([[0, 1.0], [0.2, 0.5]])

    solver = hdx.USL_ASFLIP(
        config=hdx.Config(dt=0.001, total_time=1),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.5),
        material_points=hdx.MaterialPoints(position_stack=position_stack),
    )
    expected_Dp = jnp.array(
        [
            [0.08333334, 0.0, 0.0],
            [0.0, 0.08333334, 0.0],
            [0.0, 0.0, 0.08333334],
        ]
    )
    np.testing.assert_allclose(solver.Dp, expected_Dp, rtol=1e-3)

    assert isinstance(solver, hdx.USL_ASFLIP)


# test 2D p2g with cubic shape functions
def test_p2g_2d():
    solver = hdx.USL_ASFLIP(
        config=hdx.Config(total_time=1, dt=0.001, shapefunction="cubic", dim=2),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=1.0, dim=2),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
        ),
        _setup_done=False,  # false, because we want to pad
    )

    solver = solver.setup()  # domain gets padded here

    @jax.jit
    def usl_p2g(solver, material_points, grid):
        shape_map, grid = solver.p2g(material_points, grid)
        return shape_map, grid

    shape_map, grid = usl_p2g(solver, solver.material_points, solver.grid)

    expected_mass_stack = jnp.array(
        [
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            3.4171855e-03,
            2.9742220e-02,
            1.5314080e-02,
            1.2656071e-04,
            0.0000000e00,
            0.0000000e00,
            1.8482782e-02,
            1.6086894e-01,
            8.2830392e-02,
            6.8453822e-04,
            0.0000000e00,
            0.0000000e00,
            6.2203016e-03,
            5.4139752e-02,
            2.7876213e-02,
            2.3037841e-04,
            0.0000000e00,
            0.0000000e00,
            4.6837995e-06,
            4.0766477e-05,
            2.0990399e-05,
            1.7347170e-07,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )
    np.testing.assert_allclose(grid.mass_stack, expected_mass_stack, rtol=1e-3)

    expected_moment_stack = [
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [3.4171855e-03, 3.4171855e-03],
        [2.9742220e-02, 2.9742220e-02],
        [1.5314080e-02, 1.5314080e-02],
        [1.2656071e-04, 1.2656071e-04],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [1.8482782e-02, 1.8482782e-02],
        [1.6086894e-01, 1.6086894e-01],
        [8.2830392e-02, 8.2830392e-02],
        [6.8453822e-04, 6.8453822e-04],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [6.2203016e-03, 6.2203016e-03],
        [5.4139752e-02, 5.4139752e-02],
        [2.7876213e-02, 2.7876213e-02],
        [2.3037841e-04, 2.3037841e-04],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [4.6837995e-06, 4.6837995e-06],
        [4.0766477e-05, 4.0766477e-05],
        [2.0990399e-05, 2.0990399e-05],
        [1.7347170e-07, 1.7347170e-07],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
        [0.0000000e00, 0.0000000e00],
    ]

    np.testing.assert_allclose(grid.moment_stack, expected_moment_stack, rtol=1e-3)


def test_g2p_2d():
    solver = hdx.USL_ASFLIP(
        config=hdx.Config(total_time=1, dt=0.001, shapefunction="cubic", dim=2),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
        ),
        _setup_done=False,  # false, because we want to pad
    )
    solver = solver.setup()

    @jax.jit
    def usl_p2g_g2p(solver, material_points, grid):
        new_shape_map, new_grid = solver.p2g(material_points, grid)
        new_solver, new_particles = solver.g2p(material_points, new_grid, new_shape_map)
        return new_particles

    material_points = usl_p2g_g2p(solver, solver.material_points, solver.grid)

    expected_volumes = jnp.array([1, 1])

    np.testing.assert_allclose(
        material_points.volume_stack, expected_volumes, rtol=1e-3
    )

    expected_velocities = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        material_points.velocity_stack, expected_velocities, rtol=1e-3
    )


# Check if update runs
def test_update():
    solver = hdx.USL_ASFLIP(
        config=hdx.Config(total_time=1, dt=0.001, shapefunction="cubic", dim=2),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
        ),
    )

    solver = solver.setup()

    solver = solver.update(0)
