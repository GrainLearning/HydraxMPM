"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""
#%%

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
total_steps = 10000

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
block1 = create_block((1, 1), 2, particle_spacing)
block2 = create_block((7, 7), 2, particle_spacing)

# Stack all the positions together
pos = np.vstack([block1, block2])
print("pos.shape", pos.shape)
particles = pm.Particles.create(positions=pos, original_density=1000)

nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0]), end=jnp.array([domain_size, domain_size]), node_spacing=cell_size
)

print("nodes.num_nodes", len(nodes.moments))

shapefunctions = pm.CubicShapeFunction.create(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos), dim=2)


gravity = pm.Gravity.create(gravity=jnp.array([-0.001, -0.0098]))
box = pm.DirichletBox.create()

usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[material],
    forces=[gravity,box],
    shapefunctions=shapefunctions,
    alpha=0.99,
    dt=0.001,
)

@jax.tree_util.Partial
def save_particles(package):
    steps, usl = package
    positions = usl.particles.positions
    mean_velocity = jnp.mean(usl.particles.velocities, axis=1)
    jnp.savez(f"output/particles_{steps}", positions=positions, mean_velocity=mean_velocity)
    print(f"output {steps}", end="\r")


print("Running simulation")

start = timeit.default_timer()
print("The start time is :", start)
usl = usl.solve(num_steps=total_steps, output_step=output_steps, output_function=save_particles)

print("The difference of time is :", 
              timeit.default_timer() - start)


jax.make_jaxpr(usl.update)()

print("\n Plotting")
data = jnp.load(f"./output/particles_{output_steps}.npz")
positions = data["positions"]
mean_velocity = data["mean_velocity"]

points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()

cloud = pv.PolyData(points_3d)
cloud.point_data["mean_velocities"] = data["mean_velocity"]

pl = pv.Plotter()

box = pv.Box(bounds=[0, domain_size, 0, domain_size, 0, 0])

pl.add_mesh(box, style='wireframe', color='k', line_width=2)

pl.add_mesh(
    cloud,
    scalars="mean_velocities",
    style="points",
    show_edges=True,
    render_points_as_spheres=True,
    cmap="inferno",
    point_size=10,
    clim=[-0.1, 0.1],
)

pl.camera_position = "xy"
pl.open_gif("./figures/animation_cube_fall_rough_walls.gif")

for i in range(output_steps, total_steps, output_steps):
    data = jnp.load(f"./output/particles_{i}.npz")
    positions = data["positions"]
    mean_velocity = data["mean_velocity"]
    points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()
    cloud.points = points_3d
    cloud.point_data["mean_velocities"] = data["mean_velocity"]
    pl.write_frame()
pl.close()

