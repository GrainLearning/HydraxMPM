"""Unit tests for the USL Solver."""

from xml import dom
import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to initialize usl solver."""
    position_stack = jnp.array([[0, 1.0], [0.2, 0.5]])

    usl = hdx.USL(
        dim=2,
        material_points=hdx.MaterialPoints(position_stack=position_stack),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1),
        alpha=0.1,
    )

    assert isinstance(usl, hdx.USL)


def test_p2g_2d():
    """Unit test to perform particle-to-grid transfer for 2D."""

    usl = hdx.USL(
        shapefunction="linear",
        dim=2,
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
        ),
        alpha=0.99,
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    @jax.jit
    def usl_p2g(solver, material_points, grid):
        shape_map, grid = solver.p2g(material_points, grid, dt=0.001)
        return shape_map, grid

    shape_map, grid = usl_p2g(usl, usl.material_points, usl.grid)

    expected_mass_stack = jnp.array([0.27, 0.09, 0.03, 0.01])

    np.testing.assert_allclose(grid.mass_stack, expected_mass_stack, rtol=1e-3)

    expected_node_moment_stack = jnp.array(
        [[0.27, 0.27], [0.09, 0.09], [0.03, 0.03], [0.01, 0.01]]
    )

    np.testing.assert_allclose(grid.moment_stack, expected_node_moment_stack, rtol=1e-3)


def test_p2g_3d():
    """Unit test to perform particle-to-grid transfer in 3D."""

    usl = hdx.USL(
        shapefunction="linear",
        dim=3,
        grid=hdx.Grid(origin=[0.0, 0.0, 0.0], end=[1.0, 1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
            velocity_stack=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
        ),
        alpha=0.99,
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    @jax.jit
    def usl_p2g(usl, material_points, grid):
        usl, nodes = usl.p2g(material_points, grid, dt=0.001)
        return usl, nodes

    usl, grid = usl_p2g(usl, usl.material_points, usl.grid)

    expected_mass_stack = jnp.array(
        [0.189, 0.081, 0.063, 0.027, 0.021, 0.009, 0.007, 0.003]
    )

    np.testing.assert_allclose(grid.mass_stack, expected_mass_stack, rtol=1e-3)

    expected_node_moment_stack = jnp.array(
        [
            [0.189, 0.189, 0.189],
            [0.081, 0.081, 0.081],
            [0.063, 0.063, 0.063],
            [0.027, 0.027, 0.027],
            [0.021, 0.021, 0.021],
            [0.009, 0.009, 0.009],
            [0.007, 0.007, 0.007],
            [0.003, 0.003, 0.003],
        ]
    )

    np.testing.assert_allclose(grid.moment_stack, expected_node_moment_stack, rtol=1e-3)


def test_g2p_2d():
    usl = hdx.USL(
        shapefunction="linear",
        dim=2,
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
            velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
            stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
        ),
        alpha=0.99,
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    @jax.jit
    def usl_p2g_g2p(solver, material_points, grid):
        new_shape_map, new_grid = solver.p2g(material_points, grid, dt=0.1)
        new_particles = solver.g2p(material_points, new_grid, new_shape_map, dt=0.1)
        return new_particles

    material_points = usl_p2g_g2p(usl, usl.material_points, usl.grid)

    expected_volume_stack = jnp.array([0.697986, 0.398849])

    np.testing.assert_allclose(
        material_points.volume_stack, expected_volume_stack, rtol=1e-3
    )

    expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])

    np.testing.assert_allclose(
        material_points.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_position_stack = jnp.array([[0.2, 0.35], [0.2, 0.35]])
    np.testing.assert_allclose(
        material_points.position_stack, expected_position_stack, rtol=1e-3
    )

    expected_velocity_stack = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        material_points.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_L_stack = jnp.array(
        [
            [
                [-0.019445, -0.019445, 0.0],
                [-0.009333334, -0.009333334, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [-0.019445, -0.019445, 0.0],
                [-0.009333334, -0.009333334, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )

    np.testing.assert_allclose(material_points.L_stack, expected_L_stack, rtol=1e-3)

    expected_F_stack = jnp.array(
        [
            [
                [9.980556e-01, -1.944451e-03, 0.0],
                [-9.333365e-04, 9.990667e-01, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [9.980556e-01, -1.944451e-03, 0.0],
                [-9.333365e-04, 9.990667e-01, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
    )

    np.testing.assert_allclose(material_points.F_stack, expected_F_stack, rtol=1e-3)


def test_g2p_3d():
    usl = hdx.USL(
        shapefunction="linear",
        dim=3,
        grid=hdx.Grid(origin=[0.0, 0.0, 0.0], end=[1.0, 1.0, 1.0], cell_size=1.0),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.25, 0.3], [0.1, 0.25, 0.3]]),
            velocity_stack=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            mass_stack=jnp.array([0.1, 0.3]),
            volume_stack=jnp.array([0.7, 0.4]),
            stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
        ),
        alpha=0.99,
        _setup_done=True,  # to avoid padding the domain on first setup
    )

    @jax.jit
    def usl_p2g_g2p(solver, material_points, grid):
        new_shape_map, new_grid = solver.p2g(material_points, grid, dt=0.1)
        new_particles = solver.g2p(material_points, new_grid, new_shape_map, dt=0.1)
        return new_particles

    material_points = usl_p2g_g2p(usl, usl.material_points, usl.grid)

    expected_volume_stack = jnp.array([0.697402, 0.25155553])

    np.testing.assert_allclose(
        material_points.volume_stack[:1], expected_volume_stack[:1], rtol=1e-3
    )

    expected_velocity_stack = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(
        material_points.velocity_stack, expected_velocity_stack, rtol=1e-3
    )

    expected_position_stack = jnp.array([[0.2, 0.35, 0.4], [0.2, 0.35, 0.4]])

    np.testing.assert_allclose(
        material_points.position_stack, expected_position_stack, rtol=1e-3
    )

    expected_L_stack = jnp.array(
        [
            [
                [-0.019445, -0.019445, -0.019445],
                [-0.00933333, -0.00933333, -0.00933333],
                [-0.00833333, -0.00833333, -0.00833333],
            ],
            [
                [-0.019445, -0.019445, -0.019445],
                [-0.00933333, -0.00933333, -0.00933333],
                [-0.00833333, -0.00833333, -0.00833333],
            ],
        ]
    )

    np.testing.assert_allclose(material_points.L_stack, expected_L_stack, rtol=1e-3)

    expected_F_stack = jnp.array(
        [
            [
                [9.980556e-01, -1.944446e-03, -1.944446e-03],
                [-9.333352e-04, 9.990667e-01, -9.333352e-04],
                [-8.333346e-04, -8.333346e-04, 9.991667e-01],
            ],
            [
                [9.980556e-01, -1.944446e-03, -1.944446e-03],
                [-9.333352e-04, 9.990667e-01, -9.333352e-04],
                [-8.333346e-04, -8.333346e-04, 9.991667e-01],
            ],
        ]
    )

    np.testing.assert_allclose(material_points.F_stack, expected_F_stack, rtol=1e-3)


def test_update():
    """Unit test to update the state of the USL solver."""

    usl = hdx.USL(
        dim=2,
        grid=hdx.Grid(
            origin=[0.0, 0.0],
            end=[1.0, 1.0],
            cell_size=0.5,
        ),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array([[0.1, 0.1], [0.7, 0.1]]),
            velocity_stack=jnp.array([[1.0, 2.0], [0.3, 0.1]]),
            volume_stack=jnp.array([1.0, 0.2]),
            mass_stack=jnp.array([1.0, 3.0]),
        ),
        alpha=0.9,
    )
    usl = usl.setup()
    usl = usl.update(1, dt=0.001)
