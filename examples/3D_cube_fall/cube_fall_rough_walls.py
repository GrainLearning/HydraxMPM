"""3D cube falling"""
import jax.numpy as jnp
import numpy as np
import jax
import pymudokon as pm

import pyvista as pv

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 20) * domain_size

output_steps = 10
total_steps = 31000

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
pos = create_block((5, 5, 5), 3, particle_spacing)

particles = pm.Particles.create(positions=pos, original_density=1000)

nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0, 0.0]), end=jnp.ones(3)*domain_size, node_spacing=cell_size
)

shapefunctions = pm.LinearShapeFunction.create(len(pos), 3)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos), dim=3)

gravity = pm.Gravity.create(gravity=jnp.array([0.000, 0.0, -0.0098]))

box = pm.DirichletBox.create(nodes)

usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[material],
    forces=[gravity,box],
    shapefunctions=shapefunctions,
    alpha=0.99,
    dt=0.001,

)



# Assuming origin and domain_size are defined
cube = pv.Cube(bounds = [0, domain_size, 0, domain_size, 0, domain_size])

# Save the cube to a file
cube.save('output/bbox.vtk')

@jax.tree_util.Partial
def save_particles(package):
    step, usl = package
    if step > 30000:
        positions = usl.particles.positions
        mean_velocity = jnp.mean(usl.particles.velocities, axis=1)

        points = pv.PolyData(positions.__array__())
        points.point_data["mean_velocity"] = mean_velocity.__array__()
        points.save(f"output/particles_{step}.vtk")



    # jnp.savez(f"output/particles_{steps}", positions=positions, mean_velocity=mean_velocity)
    print(f"output {step}", end="\r")

print("Running simulation")

usl = usl.solve(num_steps=total_steps, output_step=output_steps, output_function=save_particles)
