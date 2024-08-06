# %%
import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# granular column collapse
# aspect ratio 2.0, 1.0, 0.5
# width 0.2 [m]
# largest cell_size 0.0125 [m]
# smallest cell_size 0.00625 [m]
# dt = 3 *10**-6 [s]
# 16 MPs per cell (or 4)
# Solid density
# 2450 [kg/m^-3]
# Initial density 1500

# dam
aspect = 2.0
column_width = 0.2  # [m]
column_height = column_width * aspect  # [m]

domain_height = column_height * 1.2  # [m]
domain_length = 4 * column_width  # [m]
cell_size = 0.0125  # [m]

# material parameterss
rho = 1500  # [kg/m^3]
# dt = 3 * 10**-6  # [s]
dt = 3 * 10**-5


# gravity
g = -9.8  # [m/s^2]


# background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([domain_length, domain_height])

particles_per_cell = 4

nodes = pm.Nodes.create(origin=origin, end=end, node_spacing=cell_size)

sep = cell_size / particles_per_cell
x = np.arange(0, column_width + sep, sep) + 3.5 * sep
y = np.arange(0, column_height + sep, sep) + 5.5 * sep
xv, yv = np.meshgrid(x, y)

pnt_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

# Uncomment to plot the initial particle distribution
# import matplotlib.pyplot as plt
# plt.plot(pnt_stack[:, 0], pnt_stack[:, 1], "o", markersize=0.5)
# plt.xlim([0, domain_length])
# plt.ylim([0, domain_height])
# plt.show()


particles = pm.Particles.create(position_stack=jnp.array(pnt_stack))

nodes = pm.Nodes.create(
    origin=origin, end=end, node_spacing=cell_size, small_mass_cutoff=1e-12
)

shapefunctions = pm.CubicShapeFunction.create(len(pnt_stack), 2)

particles, nodes, shapefunctions = pm.discretize(
    particles, nodes, shapefunctions, ppc=particles_per_cell, density_ref=rho
)
stress_ref_stack = -1e2 * jnp.zeros((particles.position_stack.shape[0], 3, 3)).at[
    :, [0, 1, 2], [0, 1, 2]
].set(1)

clay = pm.ModifiedCamClay.create(
    nu=0.3, M=0.8, R=1, lam=0.186, kap=0.010, Vs=2.0, stress_ref_stack=stress_ref_stack
)
particles = particles.replace(stress_stack=stress_ref_stack)

# # dt = clay.get_timestep(cell_size, rho, pressure=1e6, factor=0.5)

gravity = pm.Gravity.create(gravity=jnp.array([0.0, g]))

box = pm.DirichletBox.create(
    nodes,
    boundary_types=(
        ("slip_negative_normal", "slip_positive_normal"),
        ("stick", "stick"),
    ),
)


solver = pm.USL_APIC.create(cell_size, dim=2, num_particles=len(pnt_stack), dt=dt)


carry, accumulate = pm.run_solver(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[clay],
    forces_stack=[gravity, box],
    num_steps=60000,
    store_every=1000,
    particles_output=("position_stack", "stress_stack", "mass_stack"),
)

print("Simulation done.. plotting might take a while")

position_stack, stress_stack, mass_stack = accumulate

pressure_stack = jax.vmap(pm.get_pressure_stack)(stress_stack)

pm.plot_simple(
    origin=nodes.origin,
    end=nodes.end,
    positions_stack=position_stack,
    scalars=pressure_stack,
    scalars_name="p [Pa]",
    particles_plot_params={
        "clim": [jnp.min(pressure_stack), jnp.max(pressure_stack)],
        "point_size": 10,
    },
    output_file=dir_path + "/mcc_collapse.gif",
)
