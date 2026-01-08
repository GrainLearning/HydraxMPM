"""
This example demonstrates the impact of two spheres on each other.

"""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx
import jax
import os



def create_circle(center: np.array, radius: float, cell_size: float, ppc: int = 2):
    """This function creates a circle of material points."""
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


# pre-configure
cell_size = 0.05

points_per_cell = 4

circle_radius = 0.2


circle1_center = np.array([0.255, 0.255])
circle2_center = np.array([0.745, 0.745])

circle_centers = np.array([circle1_center, circle2_center])
circles = [
    create_circle(center, circle_radius, cell_size, points_per_cell)
    for center in circle_centers
]
pos = np.vstack(circles)

velocities = [
        np.full(circle.shape, 0.1 if i == 0 else -0.1)
        for i, circle in enumerate(circles)
    ]
vels = np.vstack(velocities)

sim_builder = hdx.StandardSimBuilder()

sim_builder = sim_builder.add_grid(
    origin=(0.0, 0.0),
    end=(1.0, 1.0),
    cell_size=cell_size,
    padding=2
)

sim_builder = sim_builder.add_material_points(
    position_stack=pos,
    velocity_stack=vels,
    density_stack=jnp.full(len(pos), 1000.0),
    cell_size=cell_size,
    ppc=points_per_cell,
)


sim_builder = sim_builder.add_constitutive_law(
    hdx.LinearElasticLaw(E=1000.0, nu=0.3)
)

sim_builder = sim_builder.couple(shapefunction="linear")

sim_builder = sim_builder.add_solver(
    scheme="usl",
)

mpm_solver,sim_state = sim_builder.build()


# vis = hdx.RerunVisualizer(grid_topology=mpm_solver.couplings[0].grid_topology)

vtk = hdx.VTKVisualizer(
    output_dir="output_vtk/two_sphere_impact",
    grid_topology=mpm_solver.couplings[0].grid_topology,
    ppc=points_per_cell,
    relative_dir=__file__ 
)

sim_io = hdx.SimIO(output_dir="output_data", relative_dir=__file__)

sim_io.save_solver(mpm_solver, 
                   meta_info={
                       "desc": "Two sphere impact demo",
                       "ppc": points_per_cell,
                        "cell_size": cell_size,
                        "circle_radius": circle_radius,
                        "domain_size": (1.0, 1.0),
                       }
                   )

total_steps = int(2.3/0.001)

def log_simulation(sim_state: hdx.SimState):
    # vis.log_simulation(sim_state)
    vtk.log_simulation(sim_state)
    sim_io.save_step(sim_state)


def loop_body(i, sim_state):
    sim_state = mpm_solver.step(sim_state)
    jax.lax.cond(
        i % 100 == 0,
        lambda s: jax.debug.callback(log_simulation, s),
        lambda s: None,
        sim_state
    )
    return sim_state

@jax.jit
def run_sim(sim_state: hdx.SimState):
    sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
    return sim_state

sim_state = run_sim(sim_state)

