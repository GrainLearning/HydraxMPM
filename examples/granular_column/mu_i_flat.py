"""Column collapse with small uniform pressure and solid volume fraction"""

import time
from functools import partial
from inspect import signature

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

import hydraxmpm as hdx

aspect = 2.0
column_width = 0.2  # [m]
column_height = 0.4  # [m]

# material parameters
phi_c = 0.648  # [-] rigid limit
phi_0 = 0.65  # [-] initial solid volume fraction
rho_p = 2000  # [kg/m^3] particle (skeletan density)
rho = rho_p * phi_0  # [kg/m^3] bulk density
d = 0.0053  # [m] particle diameter

# gravity
g = -9.8  # [m/s^2]

_MPMConfig = partial(
    hdx.MPMConfig,
    origin=[0.0, 0.0],
    end=np.array([2.0, 0.6]),
    ppc=4,
    cell_size=0.0125,  # [m]
    shapefunction=hdx.SHAPEFUNCTION.cubic,
    num_steps=30000,
    store_every=200,
    dt=3 * 10**-5,  # [s] time step
)


def get_sv(func, val):
    return signature(func).parameters[val].default


sep = get_sv(_MPMConfig, "cell_size") / get_sv(_MPMConfig, "ppc")

# create column of particles with padding
x = np.arange(0, column_width + sep, sep) + 4 * sep
y = np.arange(0, column_height + sep, sep) + 7.5 * sep
xv, yv = np.meshgrid(x, y)

position_stack = np.array(list(zip(xv.flatten(), yv.flatten())))

config = _MPMConfig(num_points=position_stack.shape[0])

config.print_summary()

nodes = hdx.Nodes(config)

particles = hdx.Particles(config=config, position_stack=position_stack)


particles, nodes = hdx.discretize(config, particles, nodes, density_ref=rho)


# initialize material and particles, perhaps there is a less verbose way?
# get reference solid volume fraction particle mass  /volume
phi_ref_stack = particles.get_solid_volume_fraction_stack(rho_p)

material = hdx.MuI_incompressible(
    config=config,
    mu_s=0.38186285175,
    mu_d=0.57176986,
    I_0=0.279,
    rho_p=rho_p,
    d=0.0053,
    K=50 * rho * 9.8 * column_height,
)


def get_stress_ref(phi_ref):
    p_ref = material.get_p_ref(phi_ref)
    return -p_ref * jnp.eye(3)


stress_ref_stack = jax.vmap(get_stress_ref)(phi_ref_stack)


particles = eqx.tree_at(
    lambda state: (state.stress_stack),
    particles,
    (stress_ref_stack),
)

gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, g]))

box = hdx.NodeLevelSet(config,mu=0.4)

solver = hdx.USL_APIC(config=config)


start_time = time.time()


bbox = pv.Box(
    bounds=np.array(
        list(
            zip(
                hdx.point_to_3D(config, config.origin),
                hdx.point_to_3D(config, config.end),
            )
        )
    ).flatten()
)

bbox.save(config.dir_path + "/output/mu_i_flat/bbox.vtk")


# # save restart file
def io_vtk(carry, step):
    (
        solver,
        particles,
        nodes,
        material_stack,
        forces_stack,
    ) = carry

    cloud = pv.PolyData(hdx.points_to_3D(particles.position_stack, 2))

    # stress_reg_stack = hdx.post_processes_stress_stack(
    #     particles.stress_stack, particles.mass_stack, particles.position_stack, nodes
    # )

    KE_stack = hdx.get_KE_stack(particles.mass_stack, particles.velocity_stack)

    KE_stack = jnp.nan_to_num(KE_stack, nan=0.0, posinf=0.0, neginf=0.0)

    # q_reg_stack = pm.get_q_vm_stack(stress_reg_stack, dim=2)
    # cloud["q_reg_stack"] = q_reg_stack

    phi_stack = particles.get_solid_volume_fraction_stack(rho_p)

    cloud["phi_stack"] = phi_stack

    jax.debug.print("step {}", step)
    cloud.save(config.dir_path + f"/output/mu_i_flat/particles_{step}.vtk")


# no jit callback
callback = lambda carry, step: jax.debug.callback(io_vtk, carry, step)


# # Run solver
carry = hdx.run_solver_io(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box],
    callback=callback,
)
# print("--- %s seconds ---" % (time.time() - start_time))
# (
#     solver,
#     particles,
#     nodes,
#     shapefunctions,
#     material_stack,
#     forces_stack,
# ) = carry


# print("Simulation done.. plotting might take a while")
