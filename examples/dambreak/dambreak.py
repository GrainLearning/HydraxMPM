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


# USL
alpha = 0.999

# background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([6.0, 6.0])


cell_size = 6 / 69

# timestep
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c

particles_per_cell = 2

nodes = pm.Nodes.create(origin=origin, end=end, node_spacing=cell_size)

sep = cell_size / particles_per_cell
x = np.arange(0, dam_length + sep, sep) + 3.5 * sep
y = np.arange(0, dam_height + sep, sep) + 3.5 * sep
xv, yv = np.meshgrid(x, y)
pnts = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

particles = pm.Particles.create(positions=jnp.array(pnts), original_density=rho)

nodes = pm.Nodes.create(origin=origin, end=end, node_spacing=cell_size, small_mass_cutoff=1e-12)

shapefunctions = pm.CubicShapeFunction.create(len(pnts), 2)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions, ppc=particles_per_cell)

water = pm.NewtonFluid.create(K=bulk_modulus, viscosity=mu)

gravity = pm.Gravity.create(gravity=jnp.array([0.0, g]))
box = pm.DirichletBox.create(nodes, boundary_types=jnp.array([[3, 2], [3, 2]]))


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
    particles_keys=("positions", "velocities", "masses"),
)

print("Simulation done.. plotting might take a while")

positions_stack, velocities_stack, masses_stack = accumulate

KE_stack = pm.get_KE(masses_stack, velocities_stack)

pm.plot_simple(
    origin=nodes.origin,
    end=nodes.end,
    positions_stack=positions_stack,
    scalars=KE_stack,
    scalars_name="KE",
    particles_plot_params={"clim": [jnp.min(KE_stack), jnp.max(KE_stack)], "point_size": 10},
)
