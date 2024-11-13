"""Unit tests for the DirichletBox class."""

import jax.numpy as jnp

import hydraxmpm as hdx


def test_init():
    """Unit test to initialize the DirichletBox class."""

    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.1,
        num_points=2,
        shapefunction="linear",
    )

    id_stack = jnp.arange(config.num_cells).reshape(config.grid_size)

    lower_ids_stack = id_stack.at[:, 0].get()

    box = hdx.NodeLevelSet(config, id_stack=lower_ids_stack)

    assert isinstance(box, hdx.NodeLevelSet)

def test_call_2d():
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.1,
        num_points=1,
        shapefunction="linear",
    )

    id_stack = jnp.arange(config.num_cells).reshape(config.grid_size)

    lower_ids_stack = id_stack.at[:, 0].get()

    box = hdx.NodeLevelSet(config, id_stack=lower_ids_stack)

    nodes = hdx.Nodes(config)

    position_stack = jnp.array([[0.05, 0.2]])

    particles = hdx.Particles(
        config=config,
        position_stack=position_stack,
        mass_stack = jnp.ones(1)
        )
    

    nodes, box = box.apply_on_nodes(
        particles =particles,
        nodes = nodes
    )
    


