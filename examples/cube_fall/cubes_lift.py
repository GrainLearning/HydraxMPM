"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""
# %%

import timeit

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 80) * domain_size


particle_spacing = cell_size / particles_per_cell

print("Creating simulation")


def create_block(block_start, block_size, spacing):
    """Create a block of particles in 2D space."""
    block_end = (block_start[0] + block_size, block_start[1] + block_size)
    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    block = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return block


# Create two blocks (cubes in 2D context)
block1 = create_block((3, 1), 2, particle_spacing)
block2 = create_block((7.5, 7), 2, particle_spacing)
block3 = create_block((1, 7), 2, particle_spacing)
block4 = create_block((5, 7), 2, particle_spacing)

# Stack all the positions together
pos = np.vstack([block1, block2, block3, block4])

rigid_x = jnp.arange(0, domain_size / 1.5, particle_spacing / 2)

rigid_pos = jnp.zeros((len(rigid_x), 2))

rigid_pos = rigid_pos.at[:, 0].set(rigid_x)

rigid_pos = rigid_pos.at[:, 1].set(0.5)

rigid_velocities = jnp.zeros((len(rigid_x), 2)).at[:, 1].set(0.05)

# Stack all the positions together
print("pos.shape", pos.shape)
particles = pm.Particles.create(positions=pos, original_density=1000)

nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([domain_size, domain_size]), node_spacing=cell_size)

print("nodes.num_nodes", len(nodes.moments))

shapefunctions = pm.CubicShapeFunction.create(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=10000.0, nu=0.1, num_particles=len(pos))


gravity = pm.Gravity.create(gravity=jnp.array([0.0, -0.0098]))

box = pm.DirichletBox.create(
    nodes,
    boundary_types=jnp.array([[0, 0], [3, 0]]),
)

rigid_particle_wall = pm.RigidParticles(
    positions=rigid_pos, velocities=rigid_velocities, shapefunction=pm.LinearShapeFunction.create(len(rigid_pos), 2)
)

solver = pm.USL.create(alpha=0.99, dt=0.003)


carry, accumulate = pm.run_solver(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[material],
    forces_stack=[gravity, box, rigid_particle_wall],
    num_steps=60000,
    store_every=1000,
    particles_keys=("positions", "velocities", "masses"),
    forces_keys=("positions",),
)


positions_stack, velocities_stack, masses_stack, rigid_positions_stack = accumulate

KE_stack = pm.get_KE(masses_stack, velocities_stack)


pm.plot_simple(
    origin=nodes.origin,
    end=nodes.end,
    positions_stack=positions_stack,
    rigid_positions_stack=rigid_positions_stack,
    scalars=KE_stack,
    scalars_name="KE",
    particles_plot_params={"clim": [jnp.min(KE_stack), jnp.max(KE_stack)], "point_size": 10},
)
