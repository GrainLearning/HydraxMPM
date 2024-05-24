"""Run two-sphere impact simulation."""

import jax.numpy as jnp
import numpy as np

import pymudokon as pm

import pyvista as pv


def create_circle(center: np.array, radius: float, cell_size: float, ppc: int = 2):
    """Generate a circle of particles.

    Args:
        center (np.array): center of the circle
        radius (float): radius of the circle
        cell_size (float): size of the background grid cells
        ppc (int, optional): particles per cell. Defaults to 2.

    Returns:
        np.array: coordinates of the particles
    """
    start, end = center - radius, center + radius
    spacing = cell_size / (ppc / 2)
    tol = +0.00005  # Add a tolerance to avoid numerical issues
    x = np.arange(start[0], end[0] + spacing, spacing) + 0.5 * spacing
    y = np.arange(start[1], end[1] + spacing, spacing) + 0.5 * spacing
    xv, yv = np.meshgrid(x, y)
    grid_coords = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (grid_coords[:, 1] - center[1]) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]


print("Creating simulation")
# Create circles of particles and concatenate them into a single array
cell_size = 0.05
particles_per_cell = 4
circle_radius = 0.2

circle1_center = np.array([0.255, 0.255])
circle2_center = np.array([0.745, 0.745])
circle_centers = np.array([circle1_center, circle2_center])
circles = [create_circle(center, circle_radius, cell_size, particles_per_cell) for center in circle_centers]
pos = np.vstack(circles)

velocities = [np.full(circle.shape, 0.1 if i == 0 else -0.1) for i, circle in enumerate(circles)]
vels = np.vstack(velocities)

particles = pm.Particles.register(positions=jnp.array(pos), velocities=jnp.array(vels), original_density=1000)

nodes = pm.Nodes.register(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=cell_size)

shapefunctions = pm.LinearShapeFunction.register(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.3, num_particles=len(pos), dim=2)

usl = pm.USL.register(
    particles=particles, nodes=nodes, materials=[material], shapefunctions=shapefunctions, alpha=0.98, dt=0.001
)


def save_particles(package):
    steps, usl = package
    positions = usl.particles.positions
    mean_velocity = jnp.mean(usl.particles.velocities, axis=1)
    jnp.savez(f"output/particles_{steps}", positions=positions, mean_velocity=mean_velocity)
    print(f"output {steps}", end="\r")
    return usl


print("Running simulation")
usl = usl.solve(num_steps=3000, output_steps=100, output_function=save_particles)

print("\n Plotting")
data = jnp.load(f"./output/particles_{100}.npz")
positions = data["positions"]
mean_velocity = data["mean_velocity"]

points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()

cloud = pv.PolyData(points_3d)
cloud.point_data["mean_velocities"] = data["mean_velocity"]

pl = pv.Plotter()

box = pv.Box(bounds=[0, 1, 0, 1, 0, 0])

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
pl.open_gif("./figures/animation_two_spheres_impact.gif")

for i in range(100, 3000, 100):
    data = jnp.load(f"./output/particles_{i}.npz")
    positions = data["positions"]
    mean_velocity = data["mean_velocity"]
    points_3d = jnp.pad(data["positions"], [(0, 0), (0, 1)], mode="constant").__array__()
    cloud.points = points_3d
    cloud.point_data["mean_velocities"] = data["mean_velocity"]
    pl.write_frame()
pl.close()
