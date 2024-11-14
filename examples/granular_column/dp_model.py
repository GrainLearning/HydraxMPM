"""Column collapse with small uniform pressure and solid volume fraction"""

import time
from functools import partial
from inspect import signature

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

aspect = 2.0
column_width = 0.2  # [m]
column_height = 0.4  # [m]

# material parameters
phi_c = 0.648  # [-] rigid limit
phi_0 = 0.65  # [-] initial solid volume fraction
rho_p = 1200  # [kg/m^3] particle (skeletan density)
rho = rho_p * phi_0  # [kg/m^3] bulk density
# rho = 1500
# rho = 2000
# rho = 1
d = 0.0053  # [m] particle diameter

# gravity
g = -9.8  # [m/s^2]

_MPMConfig = partial(
    hdx.MPMConfig,
    origin=[0.0, 0.0],
    end=np.array([1.0, 0.6]),
    ppc=4,
    dim=2,
    default_gpu_id=0,
    cell_size=0.0125,  # [m]
    shapefunction="cubic",
    num_steps=40000,
    store_every=200,
    project="druckerprager",
    dt=3 * 10**-5,  # [s] time step
    file=__file__
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

p_ref = 10


def get_stress_ref(xs):
    return -p_ref * jnp.eye(3)


stress_ref_stack = jax.vmap(get_stress_ref)(jnp.ones(config.num_points))


particles = eqx.tree_at(
    lambda state: (state.stress_stack),
    particles,
    (stress_ref_stack),
)

material = hdx.DruckerPragerEP(
    config=config,
    E=100_000,
    nu=0.3,
    mu_1=0.6,
    mu_2=0.0,
    c0=0.0,
    H=0.0,
    mu_1_hat=0.0,
    p_ref_stack=p_ref * jnp.ones(config.num_points),
)

gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, g]))

box = hdx.NodeLevelSet(config, mu=0.7)

solver = hdx.USL_ASFLIP(config=config)

start_time = time.time()

carry = hdx.run_solver_io(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box],
    callback=hdx.io_vtk_callback(config, particle_output=("pressure_stack",)),
)
