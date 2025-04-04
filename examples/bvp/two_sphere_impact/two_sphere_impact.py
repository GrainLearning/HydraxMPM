"""
This example demonstrates the impact of two spheres on each other.

"""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


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


if __name__ == "__main__":
    cell_size = 0.05

    mps_per_cell = 4

    circle_radius = 0.2

    circle1_center = np.array([0.255, 0.255])
    circle2_center = np.array([0.745, 0.745])

    circle_centers = np.array([circle1_center, circle2_center])
    circles = [
        create_circle(center, circle_radius, cell_size, mps_per_cell)
        for center in circle_centers
    ]
    pos = np.vstack(circles)

    # Give them opposite velocities
    velocities = [
        np.full(circle.shape, 0.1 if i == 0 else -0.1)
        for i, circle in enumerate(circles)
    ]
    vels = np.vstack(velocities)

    # Create the solver object
    usl = hdx.USL(
        config=hdx.Config(
            shapefunction="linear",
            ppc=mps_per_cell,
            num_steps=3000,
            store_every=100,
            dt=0.001,
            output=dict(
                material_points=(
                    "position_stack",  # We absolutely need this to visualize the material points
                    "KE_stack",
                )
            ),
            dim=2,
        ),
        material_points=hdx.MaterialPoints(
            position_stack=jnp.array(pos), velocity_stack=jnp.array(vels)
        ),
        constitutive_laws=hdx.LinearIsotropicElastic(E=1000.0, nu=0.3, rho_0=1000.0),
        grid=hdx.Grid(origin=[0.0, 0.0], end=[1.0, 1.0], cell_size=cell_size),
        alpha=0.98,
    )

    # Setup the solver
    usl = usl.setup()

    # This saves the output to ./output/*.npz
    usl.run()

    # Visualize the results
    hdx.viewer.view(
        usl.config,
        ["KE_stack"],
    )
