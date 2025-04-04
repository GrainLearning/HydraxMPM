"""Unit tests for the DirichletBox class."""

import equinox as eqx
import jax.numpy as jnp

import hydraxmpm as hdx


def test_init():
    """Unit test to initialize the BBOX class."""
    grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)

    box = hdx.Boundary()

    new_box, new_grid = box.init_ids(grid=grid, dim=2)

    print(new_box.id_stack.shape)

    assert new_box.id_stack.shape == (72,)


def test_call_2d():
    grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)

    new_moment_nt_stack = jnp.zeros((grid.num_cells, 2)).at[:].set(1.0)

    box = hdx.Boundary()

    new_grid = eqx.tree_at(
        lambda state: (state.moment_nt_stack),
        grid,
        (new_moment_nt_stack),
    )

    new_box, new_grid = box.init_ids(grid=grid, dim=2)

    new_box.apply_on_grid(material_points=None, grid=new_grid)
