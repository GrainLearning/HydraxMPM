import jax.numpy as jnp


import pymudokon as pm


def test_run_solver():
    particles = pm.Particles.create(
        position_stack=jnp.array([[0.1, 0.25], [0.1, 0.25]]),
        velocity_stack=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    nodes = pm.Nodes.create(
        origin=jnp.array([0.0, 0.0]),
        end=jnp.array([1.0, 1.0]),
        node_spacing=1.0,
    )

    particles = particles.replace(
        mass_stack=jnp.array([0.1, 0.3]),
        volume_stack=jnp.array([0.7, 0.4]),
        volume0_stack=jnp.array([0.7, 0.4]),
        stress_stack=jnp.stack([jnp.ones((3, 3)), jnp.zeros((3, 3))]),
    )

    shapefunction = pm.LinearShapeFunction.create(2, 2)

    shapefunction, _ = shapefunction.calculate_shapefunction(
        origin=nodes.origin,
        inv_node_spacing=nodes.inv_node_spacing,
        grid_size=nodes.grid_size,
        position_stack=particles.position_stack,
    )

    usl = pm.USL.create(
        alpha=0.99,
        dt=0.1,
    )

    pm.run_solver(
        solver=usl,
        particles=particles,
        nodes=nodes,
        shapefunctions=shapefunction,
        material_stack=[],
        forces_stack=[],
    )
