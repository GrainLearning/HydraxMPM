"""Unit tests for the grid"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to create grid nodes over multiple dimensions."""

    grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)

    assert isinstance(grid, hdx.Grid)

    assert grid.num_cells == 121


test_create()


def test_post_init():
    # test grid padding outside of the domain
    grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)
    assert grid.num_cells == 121
    assert grid._is_padded == False
    grid = grid.init_padding("linear")
    assert grid.num_cells == 169
    assert grid._is_padded == True
    # grid must already be padded so it should not be padded again
    grid = grid.init_padding("linear")
    assert grid.num_cells == 169
    assert grid._is_padded == True


# test_post_init()
# def test_refresh():
#     """Unit test to reset node state."""

#     grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)

#     grid = eqx.tree_at(
#         lambda state: (state.mass_stack), grid, (jnp.ones(9).astype(jnp.float32))
#     )
#     np.testing.assert_allclose(grid.mass_stack, jnp.ones(9))

#     grid = grid.refresh()

#     np.testing.assert_allclose(grid.mass_stack, jnp.zeros(9))


# def test_properties():
#     """Unit test to create grid nodes over multiple dimensions."""

#     grid = hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=0.1)

#     position_stack = grid.position_stack

#     np.testing.assert_allclose(position_stack.shape, (121, 2))
