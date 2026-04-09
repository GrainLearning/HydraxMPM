import pytest
import jax.numpy as jnp
import equinox as eqx
from hydraxmpm.grid.grid import GridState

class TestGridTopology:
    def test_initialization_2d(self):
        origin = (0.0, 0.0)
        end = (1.0, 1.0)
        cell_size = 0.1
        grid = GridState.create(origin, end, cell_size)
        
        assert grid.dim == 2
        assert grid.origin == origin
        assert grid.end == end
        assert grid.cell_size == cell_size
        
        # (1.0 - 0.0) / 0.1 + 1 = 11
        expected_grid_size = (11, 11)
        assert grid.grid_size == expected_grid_size
        assert grid.num_cells == 121

    def test_initialization_3d(self):
        origin = (0.0, 0.0, 0.0)
        end = (1.0, 1.0, 1.0)
        cell_size = 0.5
        grid = GridState.create(origin, end, cell_size)
        
        assert grid.dim == 3
        expected_grid_size = (3, 3, 3)
        assert grid.grid_size == expected_grid_size
        assert grid.num_cells == 27

class TestGridState:
    def test_create_basic(self):
        origin = (0.0, 0.0)
        end = (1.0, 1.0)
        cell_size = 0.5
        # grid size should be (3, 3), num_cells = 9
        state = GridState.create(origin, end, cell_size)

        
        assert state.num_cells == 9
        assert state.dim == 2
        assert state.mass_stack.shape == (9,)
        assert state.moment_stack.shape == (9, 2)
        assert state.moment_nt_stack.shape == (9, 2)
        
        # Check initialization to zeros
        assert jnp.all(state.mass_stack == 0)
        assert jnp.all(state.moment_stack == 0)

