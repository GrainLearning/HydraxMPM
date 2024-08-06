import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm

# dam
dam_height = 2.0
dam_length = 4.0

# material parameters
rho = 997.5
bulk_modulus = 2.0 * 10**6
mu = 0.001


# gravity
g = -9.81


# background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([6.0, 6.0])


cell_size = 6 / 69

# timestep
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c


is_apic = True


particles_per_cell = 2

nodes = pm.Nodes.create(origin=origin, end=end, node_spacing=cell_size)

sep = cell_size / particles_per_cell
x = np.arange(0, dam_length + sep, sep) + 3.5 * sep
y = np.arange(0, dam_height + sep, sep) + 3.5 * sep
xv, yv = np.meshgrid(x, y)
pnts_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)
particles = pm.Particles.create(position_stack=jnp.array(pnts_stack))

nodes = pm.Nodes.create(
    origin=origin, end=end, node_spacing=cell_size, small_mass_cutoff=1e-12
)

shapefunctions = pm.CubicShapeFunction.create(len(pnts_stack), 2)

particles, nodes, shapefunctions = pm.discretize(
    particles, nodes, shapefunctions, ppc=particles_per_cell, density_ref=rho
)

water = pm.NewtonFluid.create(K=bulk_modulus, viscosity=mu)

gravity = pm.Gravity.create(gravity=jnp.array([0.0, g]))
# Fix this
# box = pm.DirichletBox.create(nodes, boundary_types=jnp.array([[3, 2], [3, 2]]))
box = pm.DirichletBox.create(
    nodes,
    boundary_types=(
        ("slip_negative_normal", "slip_positive_normal"),
        ("stick", "stick"),
    ),
)


if is_apic:
    solver = pm.USL_APIC.create(cell_size, dim=2, num_particles=len(pnts_stack), dt=dt)
else:
    solver = pm.USL.create(alpha=0.99, dt=dt)


carry, accumulate = pm.run_solver(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[water],
    forces_stack=[gravity, box],
    num_steps=20000,
    store_every=500,
    particles_output=("position_stack", "velocity_stack", "mass_stack"),
)

print("Simulation done.. plotting might take a while")

positions_stack, velocity_stack, mass_stack = accumulate


pm.plot_simple(
    origin=nodes.origin,
    end=nodes.end,
    positions_stack=positions_stack,
    scalars=velocity_stack,
    scalars_name="Velocity magnitude",
    particles_plot_params={
        "clim": [jnp.min(velocity_stack), jnp.max(velocity_stack)],
        "point_size": 10,
    },
)
