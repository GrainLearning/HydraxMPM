"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""
# %%

import timeit
import jax.numpy as jnp
import numpy as np
import jax
import pymudokon as pm

import pyvista as pv

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 120) * domain_size

output_steps = 1000
total_steps = 120000

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
# block1 = create_block((1, 1), 2, particle_spacing)
block1 = create_block((6, 6), 2, particle_spacing)

rigid_x = jnp.arange(0, domain_size / 1.5, particle_spacing / 2)

rigid_pos = jnp.zeros((len(rigid_x), 2))

rigid_pos = rigid_pos.at[:, 0].set(rigid_x)

rigid_pos = rigid_pos.at[:, 1].set(4)

rigid_velocities = jnp.zeros((len(rigid_x), 2)).at[:, 1].set(0.1)

# Stack all the positions together
pos = np.vstack([block1])
print("pos.shape", pos.shape)
particles = pm.Particles.create(positions=pos, original_density=1000)

nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([domain_size, domain_size]), node_spacing=cell_size)

print("nodes.num_nodes", len(nodes.moments))

shapefunctions = pm.CubicShapeFunction.create(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos))


gravity = pm.Gravity.create(gravity=jnp.array([0.0, -0.0098]))
box = pm.DirichletBox.create(
    nodes,
    boundary_types=jnp.array([[0, 0], [3, 0]]),
)

rigid_particle_wall = pm.RigidParticles(
    positions=rigid_pos, velocities=rigid_velocities, shapefunction=pm.LinearShapeFunction.create(len(rigid_pos), 2)
)


usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[material],
    forces=[gravity, box, rigid_particle_wall],
    shapefunctions=shapefunctions,
    alpha=0.99,
    dt=0.001,
)

points_data_dict = {"points": [], "KE": []}
points_data_rigid_dict = {"points": []}


@jax.tree_util.Partial
def save_particles(package):
    steps, usl = package
    positions = usl.particles.positions

    points_data_dict["points"].append(positions)
    KE = pm.get_KE(
        usl.particles.masses,
        usl.particles.velocities,
    )
    points_data_dict["KE"].append(KE)

    rigid_positions = usl.forces[2].positions
    points_data_rigid_dict["points"].append(rigid_positions)

    print(f"output {steps}", end="\r")


print("Running simulation")

start = timeit.default_timer()
print("The start time is :", start)
usl = usl.solve(num_steps=total_steps, output_step=output_steps, output_function=save_particles)

print("The difference of time is :", timeit.default_timer() - start)

for key, value in points_data_dict.items():
    points_data_dict[key] = np.array(value)

pm.plot_simple(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([domain_size, domain_size]),
    particles_points=points_data_dict["points"],
    particles_scalars=points_data_dict["KE"],
    particles_scalar_name="KE",
    rigid_points=points_data_rigid_dict["points"],
    particles_plot_params={"point_size": 5},
)
