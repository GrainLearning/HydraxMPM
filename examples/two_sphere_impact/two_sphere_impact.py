"""Run two-sphere impact simulation."""

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


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

particles = pm.Particles.create(positions=jnp.array(pos), velocities=jnp.array(vels), original_density=1000)

nodes = pm.Nodes.create(origin=jnp.array([0.0, 0.0]), end=jnp.array([1.0, 1.0]), node_spacing=cell_size)

shapefunctions = pm.LinearShapeFunction.create(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos))

usl = pm.USL.create(
    particles=particles, nodes=nodes, materials=[material], shapefunctions=shapefunctions, alpha=0.98, dt=0.001
)

points_data_dict = {"points": [], "KE": []}


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
    print(KE.mean())
    print(f"output {steps}", end="\r")


print("Running simulation")
usl = usl.solve(num_steps=3000, output_step=100, output_function=save_particles)

for key, value in points_data_dict.items():
    points_data_dict[key] = np.array(value)


pm.plot_simple(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([1, 1]),
    particles_points=points_data_dict["points"],
    particles_scalars=points_data_dict["KE"],
    particles_scalar_name="KE",
    particles_plot_params={"point_size": 5, "clim": [0.009, 0.0125]},
)
