"""Unit tests for the Gravity data class."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm

import equinox as eqx

def test_create():
    """Unit test to initialize gravity."""
    
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.5,
        num_points=2,
        shapefunction_type="linear",
    )

    box = pm.Gravity(config,gravity=jnp.array([0.0, 0.0]))

    assert isinstance(box, pm.Gravity)


def test_call_2d():
    """Unit test to apply gravity force on Nodes."""
    
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.5,
        num_points=2,
        shapefunction_type="linear",
        dt=0.01
    )
    nodes = pm.Nodes(config)

    grav = pm.Gravity(config,gravity=jnp.array([0.0, 9.8]))

    
    new_nodes = eqx.tree_at(
            lambda state: (state.mass_stack,state.moment_nt_stack),
            nodes,
            (jnp.ones(nodes.num_cells) * 1.0, jnp.ones((nodes.num_cells,config.dim)) * 1.0),
    )
    import jax

    
    new_nodes, new_grav = grav(nodes=new_nodes)



    expected_moment_nt_stack = jnp.array(
        [
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
            [1.0, 1.098],
        ]
    )

    
    np.testing.assert_allclose(new_nodes.moment_nt_stack, expected_moment_nt_stack, rtol=1e-3)

test_call_2d()