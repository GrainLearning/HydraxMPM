"""Column collapse with small uniform pressure and solid volume fraction"""
from functools import partial

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import hydraxmpm as hdx
import pyvista as pv

dir_path = os.path.dirname(os.path.realpath(__file__))


# material parameters
phi_c = 0.648  # [-] rigid limit
phi_0 = 0.65  # [-] initial solid volume fraction
rho_p = 2000  # [kg/m^3] particle (skeletan density)
rho = rho_p * phi_0  # [kg/m^3] bulk density
d = 0.0053  # [m] particle diameter

# granular column
aspect = 2.0
column_width = 0.2  # [m]
column_height = column_width * aspect  # [m]
domain_height = column_height * 1.4  # [m]

# add pading for cubic shape functions
# domain_length = column_width + (5 * sep)*2 # [m]

# gravity
g = -9.8  # [m/s^2]

# time step [s]
dt = 3 * 10**-5

num_steps = 400000
store_every = 1000

domain_length = 6 * column_width  # [m]

_MPMConfig = partial(
    hdx.MPMConfig,
    project="mcc_flat",
    origin=np.array([0.0, 0.0]),
    end=np.array([domain_length, column_height]),
    # particles per cell
    ppc=4,
    cell_size=0.0125, # [m]
    shapefunction="cubic",
    num_steps=400000,
    store_every=1000,
    file=__file__,
)

# define spatial discretization (TODO: why a hard coded density_ref=997.5?)
_discretize = partial(hdx.discretize, density_ref=rho)

# separation of particles depend on the cell size and particles per cell
sep = hdx.get_sv(_MPMConfig, "cell_size") / hdx.get_sv(_MPMConfig, "ppc")


# # create domain and background grid
# origin, end = jnp.array([0.0, 0.0]), jnp.array([domain_length, domain_height])
# nodes = hdx.Nodes.create(
#     origin=origin, end=end, node_spacing=cell_size, small_mass_cutoff=1e-12
# )

# create column of particles with padding
x = np.arange(0, column_width + sep, sep) + 4 * sep
y = np.arange(0, column_height + sep, sep) + 7.5 * sep
xv, yv = np.meshgrid(x, y)

# h_shift = domain_length/2 - column_width/2
# x += h_shift

pnts_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

# determine number of material points
config = _MPMConfig(num_points=len(pnts_stack), dt=dt)

config.print_summary()

# create particles (material points) and grid nodes
particles = hdx.Particles(config=config, position_stack=jnp.array(pnts_stack))

nodes = hdx.Nodes(config)
# TODO: what does _discretize do to particles and nodes?
particles, nodes = _discretize(config=config, particles=particles, nodes=nodes)

# # initialize material and particles, perhaps there is a less verbose way?
# # get reference solid volume fraction particle mass  /volume
# phi_ref_stack = particles.get_phi_stack(rho_p)

stress_ref_stack = particles.get_stress_stack()
p_ref_stack = stress_ref_stack.trace() / 3

# define a modified cam clay material
_mcc = partial(hdx.ModifiedCamClay,
               nu=0.3,
               M=0.38186285175 * np.sqrt(3),
               R=1,
               lam=0.0186,
               kap=0.0010,
               p_ref_stack=p_ref_stack,
               rho_p=rho_p,
               )
# instantiate an MCC material
material = _mcc(config=config)

# time step depends on the cell_size, bulk modulus and initial density
# dt = (
#     0.1
#     * hdx.get_sv(_MPMConfig, "cell_size")
#     / np.sqrt(hdx.get_sv(_mcc, "K") / hdx.get_sv(_discretize, "density_ref"))
# )

# add forcing
gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, -9.81]))

# floor is rough boundary and walls are slip boundaries
# box = hdx.DirichletBox.create(
#     nodes,
#     boundary_types=(
#         ("slip_negative_normal", "slip_positive_normal"),
#         ("stick", "stick"),
#     ),
# )

box = hdx.NodeLevelSet(config=config, mu=0.4)

# solver = hdx.USL_APIC.create(cell_size, dim=2, num_particles=len(pnt_stack), dt=dt)
# start_time = time.time()

# bbox = pv.Box(
#     bounds=np.array(list(zip(jnp.pad(origin, [0, 1]), jnp.pad(end, [0, 1])))).flatten()
# )
# bbox.save(dir_path + "/output/mcc_flat/bbox.vtk")

# instantiate the solver
solver = hdx.USL_APIC(config=config)

# # not sure how to add this
# stress_ref_stack = material.stress_ref_stack

# particles = particles.replace(stress_stack=stress_ref_stack)


# save restart file
def io_vtk(carry, step):
    (
        solver,
        particles,
        nodes,
        shapefunctions,
        material_stack,
        forces_stack,
    ) = carry

    cloud = pv.PolyData(hdx.points_to_3D(particles.position_stack, 2))

    stress_reg_stack = hdx.post_processes_stress_stack(
        particles.stress_stack,
        particles.mass_stack,
        particles.position_stack,
        nodes,
        shapefunctions,
    )

    KE_stack = hdx.get_KE_stack(particles.mass_stack, particles.velocity_stack)

    KE_stack = jnp.nan_to_num(KE_stack, nan=0.0, posinf=0.0, neginf=0.0)

    print(KE_stack.sum())
    q_reg_stack = hdx.get_q_vm_stack(stress_reg_stack, dim=2)
    cloud["q_reg_stack"] = q_reg_stack

    phi_stack = particles.get_phi_stack(rho_p)

    cloud["phi_stack"] = phi_stack

    jax.debug.print("step {}", step)
    cloud.save(dir_path + f"/output/mcc_flat/particles_{step}.vtk")


# Run solver
carry = hdx.run_solver_io(
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box],
    callback=io_vtk,
)

print("--- %s seconds ---" % (time.time() - start_time))
(
    solver,
    particles,
    nodes,
    shapefunctions,
    material_stack,
    forces_stack,
) = carry


print("Simulation done.. plotting might take a while")
