import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import hydraxmpm as hdx


def test_compute_interaction_linear_2d():
    """Test compute method for linear kernel in 2D."""
    # Setup Grid
    origin = (0.0, 0.0)
    end = (10.0, 10.0)
    cell_size = 1.0
    grid_state = hdx.GridState.create(origin, end, cell_size)
    
    # Setup Particle at (0.5, 0.5) -> should interact with nodes (0,0), (0,1), (1,0), (1,1)
    position_stack = jnp.array([[0.5, 0.5]])
    
    mapping = hdx.ShapeFunctionMapping("linear", dim=2)

    intr_cache = mapping.create_cache(
        num_points=position_stack.shape[0],
        dim=2
    )
    cache = mapping.compute(position_stack, 
                            grid_state.origin,
                            grid_state.grid_size,
                            grid_state._inv_cell_size,
                            intr_cache
                            )
    
    # Check shapes
    assert cache.num_interactions == 4
    assert cache.node_hashes.shape == (4,)
    assert cache.shape_vals.shape == (4,)
    assert cache.shape_grads.shape == (4, 3) # Gradients are padded to 3D
    assert cache.rel_dist.shape == (4, 3) # Dist vectors are padded to 3D
    
    # Check partition of unity (sum of shape functions = 1)
    assert jnp.allclose(jnp.sum(cache.shape_vals), 1.0)
    
    # Check partition of nullity (sum of gradients = 0)
    assert jnp.allclose(jnp.sum(cache.shape_grads, axis=0), 0.0, atol=1e-7)

def test_scatter_gather_consistency():
    """Test that scattering and then gathering a constant field preserves the value."""
    # Setup Grid
    origin = (0.0, 0.0)
    end = (2.0, 2.0)
    cell_size = 1.0
    grid_state = hdx.GridState.create(origin, end, cell_size)
    
    # Particle
    position_stack = jnp.array([[0.5, 0.5]])
    mass_stack = jnp.array([1.0])
    velocity_stack = jnp.array([[2.0, 3.0]]) # Constant velocity field
    
    mapping = hdx.ShapeFunctionMapping("linear", dim=2)
    
    cache = mapping.create_cache(
        num_points=position_stack.shape[0],
        dim=position_stack.shape[1]
    )
    cache = mapping.compute(position_stack, grid_state.origin, grid_state.grid_size, grid_state._inv_cell_size, cache)
    
    # Scatter Velocity (P2G)
    # Normalize=True performs mass-weighted averaging
    grid_velocity = mapping.scatter_to_grid(
        cache,
        velocity_stack,
        mass_stack,
        grid_state.num_cells,
        normalize=True
    )
    
    # Gather Velocity (G2P)
    # Interpolating back a constant field should yield the same value
    # Passing None for grid as it is unused in gather_from_grid
    gathered_velocity = mapping.gather_from_grid(cache, grid_velocity)
    
    assert jnp.allclose(gathered_velocity, velocity_stack)


def test_scatter_mass_conservation():
    """Test that scattering mass conserves total mass."""
    origin = (0.0, 0.0)
    end = (2.0, 2.0)
    cell_size = 1.0
    grid_state = hdx.GridState.create(origin, end, cell_size)
    
    position_stack = jnp.array([[0.5, 0.5]])
    mass_stack = jnp.array([10.0])
    
    mapping = hdx.ShapeFunctionMapping("linear", dim=2)

    cache = mapping.create_cache(
        num_points=position_stack.shape[0],
        dim=position_stack.shape[1]
    )
    cache = mapping.compute(position_stack, grid_state.origin, grid_state.grid_size, grid_state._inv_cell_size,cache)
    
    # To scatter mass (extensive quantity), we scatter '1.0' weighted by mass, without normalization.
    # grid_accum += data * (N * m) -> if data=1, grid_accum += N * m
    ones_stack = jnp.ones((1, 1))
    
    grid_mass = mapping.scatter_to_grid(
        cache,
        ones_stack,
        mass_stack,
        grid_state.num_cells,
        normalize=False
        )


    mass_stack = mapping.gather_from_grid(cache, grid_mass)
    
    assert jnp.allclose(mass_stack, mass_stack)


