"""Unit tests for the DirichletBox class."""

import jax.numpy as jnp

import pymudokon as pm
from pymudokon.shapefunctions import shapefunctions


def test_init():
    """Unit test to initialize the DirichletBox class."""

    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.1,
        num_points=2,
        shapefunction_type="linear",
    )

    id_stack = jnp.arange(config.num_cells).reshape(config.grid_size)

    lower_ids_stack = id_stack.at[:, 0].get()

    box = pm.NodeLevelSet(config, id_stack=lower_ids_stack)

    assert isinstance(box, pm.NodeLevelSet)


test_init()


def test_call_2d():
    config = pm.MPMConfig(
        origin=[0.0, 0.0],
        end=[1.0, 1.0],
        cell_size=0.1,
        num_points=1,
        shapefunction_type="linear",
    )

    id_stack = jnp.arange(config.num_cells).reshape(config.grid_size)

    lower_ids_stack = id_stack.at[:, 0].get()

    box = pm.NodeLevelSet(config, id_stack=lower_ids_stack)

    grid = pm.GridStencilMap(config)

    position_stack = jnp.array([[0.05, 0.2]])

    particles = pm.Particles(
        config=config,
        position_stack=position_stack,
        mass_stack = jnp.ones(1)
        )

    nodes = pm.Nodes(config)
    
    shapefunctions = pm.LinearShapeFunction(config)

    grid = grid.partition(particles.position_stack)
    
    shapefunctions = shapefunctions(grid,particles)

    # print(shapefunctions.shapef_grad_stack)
    nodes, box = box(nodes, grid, particles, shapefunctions)
    import jax
    jax.debug.print("moment_nt_stack{}",nodes.moment_nt_stack)

test_call_2d()

# def test_apply_on_node_moments():
#     """Unit to test update of DirichletBox."""
#     nodes = pm.Nodes.create(
#         origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=0.1
#     )

#     box = pm.DirichletBox.create(
#         nodes,
#         boundary_types=jnp.array([[2, 1], [1, 1]]),
#     )

#     box.apply_on_nodes_moments(nodes)

#     # passed the test if no error is raised
