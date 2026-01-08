import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hydraxmpm as hdx


def test_material_point_base_state():
    position_stack = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    state = hdx.BaseMaterialPointState(
        position_stack=position_stack,
        velocity_stack=jnp.zeros_like(position_stack),
        mass_stack = jnp.array([1.0, 1.0]),
        )
    
    assert state.num_points == 2
    assert state.dim == 2
    np.testing.assert_array_equal(state.position_stack, position_stack)


def test_standard_material_point_defaults():
    """Test creation with minimal arguments (defaults)."""
    # 3D points
    position_stack = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    
    state = hdx.MaterialPointState.create(position_stack=position_stack)
    
    
    assert state.num_points == 2
    assert state.dim == 3

    # Check other defaults
    np.testing.assert_array_equal(state.velocity_stack, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(state.force_stack, jnp.zeros((2, 3)))
    np.testing.assert_array_equal(state.L_stack, jnp.zeros((2, 3, 3)))
    np.testing.assert_array_equal(state.stress_stack, jnp.zeros((2, 3, 3)))
    
    # F_stack default is identity
    expected_F = jnp.tile(jnp.eye(3), (2, 1, 1))
    np.testing.assert_array_equal(state.F_stack, expected_F)

def test_standard_material_point_state_no_position():
    # Should warn and default to origin
    # We expect multiple warnings here (no position, plus missing params)
    with pytest.warns(UserWarning, match="No position_stack provided"):
        state = hdx.MaterialPointState.create(
            cell_size=1.0,
            points_per_cell=1,
            density_per_particle=1.0,
            volume_per_particle=1.0
        )
        
    assert state.num_points == 1
    assert state.dim == 3
    np.testing.assert_array_equal(state.position_stack, jnp.array([[0.0, 0.0, 0.0]]))
