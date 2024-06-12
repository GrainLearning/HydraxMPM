import jax.numpy as jnp
import numpy as np

import pymudokon as pm
import pyvista as pv
import jax

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
alpha = 1.0

# background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([6.0, 6.0])


cell_size = 6 / 69
# timestep
c = np.sqrt(bulk_modulus / rho)
dt = 0.1 * cell_size / c

particles_per_cell = 2

total_steps = 20000
output_steps = 1000
print(cell_size, dt, total_steps)

nodes = pm.Nodes.create(origin=origin, end=end, node_spacing=cell_size)

sep = cell_size / particles_per_cell
x = np.arange(0, dam_length + sep, sep) + 3.5 * sep
y = np.arange(0, dam_height + sep, sep) + 3.5 * sep
xv, yv = np.meshgrid(x, y)
pnts = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

particles = pm.Particles.create(positions=jnp.array(pnts), original_density=rho)

nodes = pm.Nodes.create(
    origin=origin,
    end=end,
    node_spacing=cell_size,
    small_mass_cutoff=1e-12)

shapefunctions = pm.CubicShapeFunction.create(len(pnts), 2)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions, ppc=particles_per_cell)

water = pm.NewtonFluid.create(K=bulk_modulus, viscosity=mu)

gravity = pm.Gravity.create(gravity=jnp.array([0.0, g]))
box = pm.DirichletBox.create(nodes, boundary_types=jnp.array([[0, 0], [3, 0]]))

usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[water],
    forces=[gravity, box],
    shapefunctions=shapefunctions,
    alpha=alpha,
    dt=dt,
)

points_data_dict = {
    "points" : [],
    "KE" : []
}


@jax.tree_util.Partial
def save_particles(package):

    steps, usl = package
    positions = usl.particles.positions

    points_data_dict["points"].append(positions)
    KE = pm.get_KE( usl.particles.masses,pm.points_to_3D(usl.particles.velocities),)
    points_data_dict["KE"].append(KE)
    print(KE.mean())
 
    print(f"output {steps}", end="\r")


usl = usl.solve(num_steps=total_steps, output_step=output_steps, output_function=save_particles)


for key, value in points_data_dict.items():
    points_data_dict[key] = np.array(value)

pm.plot_simple_3D(
    points_data_dict,
    origin=origin,
    end=end,
    output_file="output.gif",
    plot_params={
        "scalars": "KE",
        "cmap": "viridis",
        "clim": [3, 7],
    }
)