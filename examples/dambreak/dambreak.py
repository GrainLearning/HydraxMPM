import jax.numpy as jnp
import numpy as np

import pymudokon as pm
import pyvista as pv

# dam
dam_height = 2.0
dam_length = 4.0

# material parameters
rho = 997.5
bulk_modulus = 2.0 * 10**6
mu = 0.001


# gravity
g = -9.81


# USL
alpha = 1.0

# background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([6.0, 6.0])


cell_size = 6 / 69
# timestep
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c

particles_per_cell = 2

total_steps = 80000
output_steps = 1000
print(cell_size, dt, total_steps)

nodes = pm.Nodes.register(origin=origin, end=end, node_spacing=cell_size)

sep = cell_size / 2
x = np.arange(0, dam_length + sep, sep) + 3.5 * sep
y = np.arange(0, dam_height + sep, sep) + 3.5 * sep
xv, yv = np.meshgrid(x, y)
pnts = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

particles = pm.Particles.register(positions=jnp.array(pnts), original_density=rho)

nodes = pm.Nodes.register(origin=origin, end=end, node_spacing=cell_size)

shapefunctions = pm.CubicShapeFunction.register(len(pnts), 2)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions, ppc=particles_per_cell)

water = pm.NewtonFluid.register(K=bulk_modulus, viscosity=mu)

gravity = pm.Gravity.register(gravity=jnp.array([0.0, g]))
box = pm.DirichletBox.register()

usl = pm.USL.register(
    particles=particles,
    nodes=nodes,
    materials=[water],
    forces=[gravity, box],
    shapefunctions=shapefunctions,
    alpha=alpha,
    dt=dt,
)


def save_particles(package):
    steps, usl = package
    positions = usl.particles.positions
    mean_velocity = jnp.mean(usl.particles.velocities, axis=1)
    jnp.savez(f"output/particles_{steps}", positions=positions, mean_velocity=mean_velocity)
    print(f"output {steps}", end="\r")
    return usl


print("Running simulation")

usl = usl.solve(num_steps=total_steps, output_steps=output_steps, output_function=save_particles)

print("\n Plotting")
data = jnp.load(f"./output/particles_{output_steps}.npz")
positions = data["positions"]
mean_velocity = data["mean_velocity"]

points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()

cloud = pv.PolyData(points_3d)
cloud.point_data["mean_velocities"] = data["mean_velocity"]

pl = pv.Plotter()

box = pv.Box(bounds=[0, 6, 0, 6, 0, 0])

pl.add_mesh(box, style="wireframe", color="k", line_width=2)

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
pl.open_gif("./figures/animation_dambreak.gif")

for i in range(output_steps, total_steps, output_steps):
    data = jnp.load(f"./output/particles_{i}.npz")
    positions = data["positions"]
    mean_velocity = data["mean_velocity"]
    points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()
    cloud.points = points_3d
    cloud.point_data["mean_velocities"] = data["mean_velocity"]
    pl.write_frame()
pl.close()
