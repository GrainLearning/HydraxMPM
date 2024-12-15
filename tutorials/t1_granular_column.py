"""Tutorial 1. Granular column collapse"""

import jax
import jax.numpy as jnp
import hydraxmpm as hdx

# granular column
column_width = 0.3  # [m]
column_height = 0.4  # [m]
ppc = 2


cell_size = 0.025  # [m]
sep = cell_size / ppc
padding = 4.5 * sep

# create padded meshgrid of positions 
x = jnp.arange(0, column_width + sep, sep) + padding - sep 
y = jnp.arange(0, column_height + sep, sep) + padding - sep
xv, yv = jnp.meshgrid(x, y)

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))

config = hdx.MPMConfig(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([column_width +  padding+2*sep, 0.5]),
    project="t1_pack",
    ppc=ppc,
    cell_size=cell_size,
    num_points=len(position_stack),
    shapefunction="cubic",
    num_steps=60000,
    store_every=500,
    default_gpu_id=0,
    dt=3 * 10**-5,  # time step[s]
    file=__file__,  # store current location of file for output
)
config.print_summary()

# Initial bulk density material parameters
phi_0 = 0.8  # [-] initial solid volume fraction
rho_p = 1400  # [kg/m^3] particle (skeletan density)
rho_0 = rho_p * phi_0  # [kg/m^3] initial bulk density

material = hdx.ModifiedCamClay(
    config=config,
    nu=0.3,
    M=0.7,
    R=1,
    lam=0.0186,
    kap=0.0010,
    rho_p=rho_p,
    p_t=1.0,
    ln_N=jnp.log(1.29),
    phi_ref_stack=phi_0 * jnp.ones(config.num_points),
)

def get_stress_ref(p_ref):
    return -p_ref * jnp.eye(3)

stress_stack = jax.vmap(get_stress_ref)(material.p_ref_stack)

particles = hdx.Particles(
    config=config, position_stack=position_stack, stress_stack=stress_stack
)

nodes = hdx.Nodes(config)

particles, nodes = hdx.discretize(
    config=config, particles=particles, nodes=nodes, density_ref=rho_0
)

stop_ramp_step = config.num_steps

increment = jnp.array([0.0, -9.8]) / stop_ramp_step

gravity = hdx.Gravity(config=config, increment=increment, stop_ramp_step=stop_ramp_step)

box = hdx.NodeLevelSet(config, mu=0.0)


solver = hdx.USL_ASFLIP(config)

print("Start gravity pack")
carry, accumulate = hdx.run_solver_io(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box],
    callbacks=(
        hdx.io_vtk_callback(
            config,
            particle_output=(
                "stress_stack",
                "phi_stack",
            ),
        ),
    ),
)

solver, particles, new_nodes, material_stack, forces_stack = carry
print("Gravity pack done")

# updating domain, gravity, and boundary

config = config.replace(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([1.2, 0.5]),
    project="t2_collapse",
)

solver = hdx.USL_ASFLIP(config)

nodes = hdx.Nodes(config)

gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, -9.8]))

box = hdx.NodeLevelSet(config, mu=0.7)


print("Start collapse")

carry, accumulate = hdx.run_solver_io(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=material_stack,
    forces_stack=[gravity, box],
    callbacks=(
        hdx.io_vtk_callback(
            config,
            particle_output=(
                "pressure_stack",
                "phi_stack",
            ),
        ),
    ),
)

print("Collapse done")
