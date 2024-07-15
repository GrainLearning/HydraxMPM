"""3D cube falling"""

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 20) * domain_size

particle_spacing = cell_size / particles_per_cell

print("Creating simulation")


def create_block(block_start, block_size, spacing):
    """Create a block of particles in 3D space."""
    block_end = (block_start[0] + block_size, block_start[1] + block_size, block_start[1] + block_size)
    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    z = np.arange(block_start[2], block_end[2], spacing)
    block = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return block


# Create two blocks (cubes in 2D context)
block1 = create_block((3, 3, 3), 2, particle_spacing) + np.array([1.0, 1.0, 4.0])

block2 = create_block((3, 3, 3), 2, particle_spacing)

pos = np.vstack([block1, block2])

vels = jnp.zeros_like(pos).at[: len(block1), 2].set(-0.12).at[len(block2) :, 2].set(-0.18)

particles = pm.Particles.create(positions=pos, velocities=vels, original_density=1000)

nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.ones(3) * domain_size, node_spacing=cell_size, small_mass_cutoff=1e-3
)

shapefunctions = pm.LinearShapeFunction.create(len(pos), 3)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=10000.0, nu=0.1, num_particles=len(pos))

gravity = pm.Gravity.create(gravity=jnp.array([0.000, 0.0, -0.016]))

box = pm.DirichletBox.create(nodes)

solver = pm.USL.create(alpha=0.99, dt=0.0001)


carry, accumulate = pm.run_solver(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[material],
    forces_stack=[gravity, box],
    num_steps=200000,
    store_every=2000,
    particles_keys=("positions", "velocities", "masses"),
)

positions_stack, velocities_stack, masses_stack = accumulate

# print(positions_stack.shape)
KE_stack = pm.get_KE(masses_stack, velocities_stack)


pm.plot_simple(
    origin=nodes.origin,
    end=nodes.end,
    positions_stack=positions_stack,
    scalars=KE_stack,
    scalars_name="KE",
    particles_plot_params={"clim": [jnp.min(KE_stack), jnp.max(KE_stack)], "point_size": 10},
)
