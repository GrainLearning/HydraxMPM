"""This example shows two spheres impacting each other [1].

[1] Sulsky, Deborah, Zhen Chen, and Howard L. Schreyer.
"A particle method for history-dependent materials."
Computer methods in applied mechanics and engineering 118.1-2 (1994): 179-196.
"""


import jax.numpy as jnp
import numpy as np

import pymudokon as pm

# 1. Load config file and set global variables

# global
dt = 0.001
particles_per_cell = 4
# shape_function = "linear"
# output_directory = "output"
total_steps, output_steps, output_start = 3000, 100, 0

# particles
circle1_center = np.array([0.255, 0.255])
circle2_center = np.array([0.745, 0.745])
circle_radius = 0.2

# material


# solver
alpha = 1.0  # pure flip


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
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]


# 3. Create particles using the circles module (in same folder)
# circle_centers = np.array([circle1_center])
circle_centers = np.array([circle1_center, circle2_center])

cell_size = 0.05
# list of two circles
circles = np.array(
    [
        create_circle(center, circle_radius, cell_size, particles_per_cell)
        for center in circle_centers
    ]
)

# concatenate the two circles into a single array
pos = np.vstack(circles)

vel1 = np.ones(circles[0].shape) * 0.1
vel2 = np.ones(circles[1].shape) * -0.1
vels = np.vstack((vel1, vel2))




particles = pm.particles.init(
    positions=jnp.array(pos),
    velocities=jnp.array(vels),
    density=1000
)

particles = pm.particles.calculate_volume(particles, cell_size, particles_per_cell=4)

particles = particles._replace(
    masses_array=1000 * particles.volumes_array,
)

nodes = pm.nodes.init(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([1.0, 1.0]),
    node_spacing=cell_size,
    particles_per_cell=4
)
material = pm.linearelastic_mat.init(
        E=1000.0, nu=0.3, num_particles=len(pos), dim=2
)

usl = pm.usl.init(
    particles=particles, nodes=nodes, materials=material, alpha=alpha, dt=dt
)


import pyvista as pv


def some_callback(package):
    usl,step = package # unused intentionally

    print(f"[JAX] output {step}/{total_steps}")
    points = usl.particles.positions_array
    velocities = usl.particles.velocities_array
    points_3d = jnp.pad(points, [(0, 0), (0, 1)], mode='constant').__array__()

    velocities_3d = jnp.pad(velocities, [(0, 0), (0, 1)], mode='constant').__array__()

    # # Create a PolyData object from the points
    cloud = pv.PolyData(points_3d)

    # # Add velocities as point data
    cloud.point_data['velocities'] = velocities_3d

    cloud.save(f"./output/particles{step}.vtp")



usl = pm.usl.solve(
    usl,
    num_steps=total_steps,
    output_step=output_steps,
    output_function=some_callback
)
