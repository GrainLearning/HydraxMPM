"""Column collapse with small uniform pressure and solid volume fraction"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

import hydraxmpm as hdx

dir_path = os.path.dirname(os.path.realpath(__file__))

# initial bulk density material parameters
phi_0 = 0.65  # [-] initial solid volume fraction
rho_p = 2000  # [kg/m^3] particle (skeletan density)
rho_0 = rho_p * phi_0  # [kg/m^3] initial bulk density

# granular column
aspect = 2.0
column_width = 0.2  # [m]
column_height = column_width * aspect  # [m]


domain_length = 6 * column_width  # [m]
domain_height = column_height * 1.4  # [m]

config = hdx.MPMConfig(
    origin=np.array([0.0, 0.0]),
    end=np.array([domain_length, column_height]),
    project="mcc_flat",
    ppc=4,  # particle per cell
    cell_size=0.0125,  # [m]
    shapefunction="cubic",
    num_steps=400000,
    store_every=1000,
    default_gpu_id=0,
    dt=3 * 10**-5,  # time step[s]
    file=__file__,  # store current location of file for output
)

# define spatial discretization (TODO: why a hard coded density_ref=997.5?)
# _discretize = partial(hdx.discretize, density_ref=rho)

# Distance between material points depend on the cell size and particles per cell
sep = config.cell_size / config.ppc

# create column of particles with padding
x = np.arange(0, column_width + sep, sep) + 4 * sep
y = np.arange(0, column_height + sep, sep) + 7.5 * sep
xv, yv = np.meshgrid(x, y)

# h_shift = domain_length/2 - column_width/2
# x += h_shift

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

# update configuration to allocate number of points
config = config.replace(num_points=len(position_stack))

config.print_summary()

# create particles (material points) and grid nodes
particles = hdx.Particles(config=config, position_stack=position_stack)
nodes = hdx.Nodes(config)

material = hdx.ModifiedCamClay(
    config=config,
    nu=0.3,
    M=0.38186285175 * np.sqrt(3),
    R=1,
    lam=0.0186,
    kap=0.0010,
    rho_p=rho_p,
    ln_N=jnp.log(1.29),
    phi_ref_stack=phi_0 * jnp.ones(config.num_points),
)


def get_stress_ref(p_ref):
    return -p_ref * jnp.eye(3)


stress_ref_stack = jax.vmap(get_stress_ref)(material.p_ref_stack)

print(material.p_ref_stack)

particles = particles.replace(stress_stack=stress_ref_stack)


# # TODO: what does _discretize do to particles and nodes?
particles, nodes = hdx.discretize(config=config, particles=particles, nodes=nodes)
# # add gravity # [m/s^2]
# gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, -9.81 ]))

# box = hdx.NodeLevelSet(config,mu=0.7)

# solver = hdx.USL_APIC(config)


# start_time = time.time()

# material

# # save restart file
# def io_vtk(carry, step):
#     (
#         solver,
#         particles,
#         nodes,
#         shapefunctions,
#         material_stack,
#         forces_stack,
#     ) = carry

#     cloud = pv.PolyData(hdx.points_to_3D(particles.position_stack, 2))

#     stress_reg_stack = hdx.post_processes_stress_stack(
#         particles.stress_stack,
#         particles.mass_stack,
#         particles.position_stack,
#         nodes,
#         shapefunctions,
#     )

#     KE_stack = hdx.get_KE_stack(particles.mass_stack, particles.velocity_stack)

#     KE_stack = jnp.nan_to_num(KE_stack, nan=0.0, posinf=0.0, neginf=0.0)

#     print(KE_stack.sum())
#     q_reg_stack = hdx.get_q_vm_stack(stress_reg_stack, dim=2)
#     cloud["q_reg_stack"] = q_reg_stack

#     phi_stack = particles.get_phi_stack(rho_p)

#     cloud["phi_stack"] = phi_stack

#     jax.debug.print("step {}", step)
#     cloud.save(dir_path + f"/output/mcc_flat/particles_{step}.vtk")


# # Run solver
# carry = hdx.run_solver_io(
#     config=config,
#     solver=solver,
#     particles=particles,
#     nodes=nodes,
#     material_stack=[material],
#     forces_stack=[gravity, box],
#     callback=io_vtk,
# )

# print("--- %s seconds ---" % (time.time() - start_time))
# (
#     config,
#     solver,
#     particles,
#     nodes,
#     material_stack,
#     forces_stack,
# ) = carry


# print("Simulation done.. plotting might take a while")
