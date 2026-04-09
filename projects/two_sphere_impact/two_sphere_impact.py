"""
This example demonstrates the impact of two spheres on each other.

"""

import jax.numpy as jnp
import numpy as np


import hydraxmpm as hdx
import jax
import os


###########################################################################
# Simulation configuration
###########################################################################


cell_size = 0.05
points_per_cell = 4
circle_radius = 0.2
origin = (0.0, 0.0)
end = (1.0, 1.0)
dt = 0.001
total_steps = int(2.5 / dt)
output_steps = int(0.05 / dt)

###########################################################################
# Helper function to create circles of material points
###########################################################################


def create_circle(center: jnp.array, radius: float, cell_size: float, ppc: int = 2):
    """This function creates a circle of material points."""
    start, end = center - radius, center + radius
    spacing = cell_size / (ppc / 2)
    tol = +0.00005  # Add a tolerance to avoid numerical issues
    x = jnp.arange(start[0], end[0] + spacing, spacing) + 0.5 * spacing
    y = jnp.arange(start[1], end[1] + spacing, spacing) + 0.5 * spacing
    xv, yv = jnp.meshgrid(x, y)
    grid_coords = jnp.array(list(zip(xv.flatten(), yv.flatten()))).astype(jnp.float64)
    circle_mask = (grid_coords[:, 0] - center[0]) ** 2 + (
        grid_coords[:, 1] - center[1]
    ) ** 2 < radius**2 + tol
    return grid_coords[circle_mask]


###########################################################################
# Create the two circles of material points
###########################################################################


circle1_center = jnp.array([0.255, 0.255])
circle2_center = jnp.array([0.745, 0.745])

circle_centers = jnp.array([circle1_center, circle2_center])
circles = [
    create_circle(center, circle_radius, cell_size, points_per_cell)
    for center in circle_centers
]
pos = jnp.vstack(circles)

velocities = [
    jnp.full(circle.shape, 0.1 if i == 0 else -0.1) for i, circle in enumerate(circles)
]
vels = jnp.vstack(velocities)

density_stack = jnp.full(len(pos), 1000.0)


###########################################################################
# Setup simulation builder
###########################################################################


sim_builder = hdx.SimBuilder()


mp_id = sim_builder.add_material_points(
    position_stack=pos,
    velocity_stack=vels,
    density_stack=density_stack,
    cell_size=cell_size,
    ppc=points_per_cell,
)

grid_id = sim_builder.add_grid(
    origin=origin, end=(1.0, 1.0), cell_size=cell_size, padding=2
)

law_id = sim_builder.add_constitutive_law(law=hdx.LinearElasticLaw(E=1000.0, nu=0.3))

body_id = sim_builder.couple(shapefunction="linear")

solver_id = sim_builder.set_solver(
    scheme="usl",
)

solver_logic, sim_state = sim_builder.build(dt=dt)


###########################################################################
# Setup rerun visualizer
###########################################################################

viewer = hdx.RerunVisualizer(is_3d=False)

viewer.log_static_domain(origin=origin, end=end, cell_size=cell_size)

###########################################################################
# Main simulation loop with logging
###########################################################################

def log_simulation(sim_state: hdx.SimState):

    mp_state = sim_state.world.material_points[0]
    step = sim_state.step
    time = sim_state.time


    viewer.log_time(current_step=int(step), current_time=float(time))
    viewer.log_material_points(
        mp_state, v_min=0, v_max=0.25, property_name="velocity_stack"
    )


def loop_body(i, sim_state):
    sim_state = solver_logic(sim_state)
    jax.lax.cond(
        i % output_steps == 0,
        lambda s: jax.debug.callback(log_simulation, s),
        lambda s: None,
        sim_state,
    )
    return sim_state


@jax.jit
def run_sim(sim_state: hdx.SimState):
    sim_state = jax.lax.fori_loop(0, total_steps, loop_body, sim_state)
    return sim_state


sim_state = run_sim(sim_state)
