# """Unit tests for the DirichletBox class."""

# import jax.numpy as jnp

# import pymudokon as pm


# def test_init():
#     """Unit test to initialize the DirichletBox class."""
#     nodes = pm.Nodes.create(
#         origin=jnp.array([0.0, 0.0, 0.0]),
#         end=jnp.array([1.0, 1.0, 1.0]),
#         node_spacing=0.1,
#     )

#     box = pm.DirichletBox.create(nodes)

#     assert isinstance(box, pm.DirichletBox)


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
