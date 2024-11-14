"""Run two-sphere impact simulation."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

fname = "/delete_spheres_output.gif"


print("Creating simulation")


cell_size = 0.05
mps_per_cell = 4
circle_radius = 0.2


def create_circle(center: np.array, radius: float, cell_size: float, ppc: int = 2):
    """Generate a circle of material points.

    Args:
        center (np.array): center of the circle
        radius (float): radius of the circle
        cell_size (float): size of the background grid cells
        ppc (int, optional): material points per cell. Defaults to 2.

    Returns:
        np.array: coordinates of the material points
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


circle1_center = np.array([0.255, 0.255])
circle2_center = np.array([0.745, 0.745])
circle_centers = np.array([circle1_center, circle2_center])
circles = [
    create_circle(center, circle_radius, cell_size, mps_per_cell)
    for center in circle_centers
]
pos = np.vstack(circles)

velocities = [
    np.full(circle.shape, 0.1 if i == 0 else -0.1) for i, circle in enumerate(circles)
]
vels = np.vstack(velocities)

config = hdx.MPMConfig(
    origin=[0.0, 0.0],
    end=[1.0, 1.0],
    cell_size=cell_size,
    num_points=len(pos),
    shapefunction=hdx.SHAPEFUNCTION.linear,
    ppc=mps_per_cell,
    num_steps=3000,
    store_every=100,
    dt=0.001,
)

particles = hdx.Particles(
    config=config, position_stack=jnp.array(pos), velocity_stack=jnp.array(vels)
)


# interactions invalid

# nodes = hdx.Nodes(config)

# particles, nodes = hdx.discretize(
#     config, particles, nodes, density_ref=1000
# )
# material = hdx.LinearIsotropicElastic(config=config, E=1000.0, nu=0.3)

# solver = hdx.USL(config=config, alpha=0.98)

# print("Running and compiling")

# carry, accumulate = hdx.run_solver(
#     config=config,
#     solver=solver,
#     particles=particles,
#     nodes=nodes,
#     material_stack=[material],
#     particles_output=("position_stack", "velocity_stack", "mass_stack"),
# )

# print("Simulation done.. plotting might take a while")

# position_stack, velocity_stack, mass_stack = accumulate

# KE_stack = hdx.get_KE_stack(mass_stack, velocity_stack)


# pvplot_cmap_ke = hdx.PvPointHelper(
#     position_stack,
#     scalar_stack=KE_stack,
#     config=config,
#     scalar_name="p [J]",
#     subplot=(0, 0),
#     timeseries_options={
#         "clim": [0, 1],
#         "point_size": 25,
#         "render_points_as_spheres": True,
#         "scalar_bar_args": {
#             "vertical": True,
#             "height": 0.8,
#             "title_font_size": 35,
#             "label_font_size": 30,
#             "font_family": "arial",
#         },
#     },
# )

# plotter = hdx.make_pvplots(
#     [pvplot_cmap_ke],
#     plotter_options={"shape": (1, 1), "window_size": ([2048, 2048])},
#     file=config.dir_path + fname,
# )
