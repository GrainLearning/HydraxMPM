from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

fname = "/dambreak.gif"

# configure MPM simulation
_MPMConfig = partial(
    hdx.MPMConfig,
    project="dambreak",
    origin=np.array([0.0, 0.0]),
    end=np.array([6.0, 6.0]),
    ppc=4,
    cell_size=6 / 69,
    shapefunction="cubic",
    num_steps=20000,
    store_every=500,
    file=__file__,
)

# define spatial discretization (TODO: why a hard coded density_ref=997.5?)
_discretize = partial(hdx.discretize, density_ref=997.5)

# define water material
_water = partial(hdx.NewtonFluid, K=2.0 * 10**6, viscosity=0.002)

# time step depends on the cell_size, bulk modulus and initial density
dt = (
    0.1
    * hdx.get_sv(_MPMConfig, "cell_size")
    / np.sqrt(hdx.get_sv(_water, "K") / hdx.get_sv(_discretize, "density_ref"))
)

# separation of particles depend on the cell size and particles per cell
sep = hdx.get_sv(_MPMConfig, "cell_size") / hdx.get_sv(_MPMConfig, "ppc")

# create dam
# TODO: why add sep to x and y?
dam_height = 2.0
dam_length = 4.0

x = np.arange(0, dam_length + sep, sep) + 5.5 * sep
y = np.arange(0, dam_height + sep, sep) + 5.5 * sep

# create a grid
xv, yv = np.meshgrid(x, y)

pnts_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

# determine number of material points
config = _MPMConfig(num_points=len(pnts_stack), dt=dt)

config.print_summary()

# instantiate a water material
material = _water(config=config)

# create particles (material points) and grid nodes
particles = hdx.Particles(config=config, position_stack=jnp.array(pnts_stack))

nodes = hdx.Nodes(config)
# TODO: what does _discretize do to particles and nodes?
particles, nodes = _discretize(config=config, particles=particles, nodes=nodes)

# add forcing
gravity = hdx.Gravity(config=config, gravity=jnp.array([0.0, -9.81]))

# add boundary geometry
box = hdx.NodeLevelSet(config=config, mu=0.4)

# instantiate the solver
solver = hdx.USL_APIC(config=config)


print("Running and compiling")

carry, accumulate = hdx.run_solver(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box],
    particles_output=("stress_stack", "position_stack", "velocity_stack", "mass_stack"),
)

print("Simulation done.. plotting might take a while")


stress_stack, position_stack, velocity_stack, mass_stack = accumulate

# unpack data for visualization
p_stack = jax.vmap(hdx.get_pressure_stack, in_axes=(0, None))(stress_stack, 2)

pvplot_cmap_q = hdx.PvPointHelper(
    config=config,
    position_stack=position_stack,
    scalar_stack=p_stack,
    scalar_name="p [Pa]",
    subplot=(0, 0),
    timeseries_options={
        "clim": [0, 50000],
        "point_size": 25,
        "render_points_as_spheres": True,
        "scalar_bar_args": {
            "vertical": True,
            "height": 0.8,
            "title_font_size": 35,
            "label_font_size": 30,
            "font_family": "arial",
        },
    },
)
plotter = hdx.make_pvplots(
    config,
    [pvplot_cmap_q],
    plotter_options={"shape": (1, 1), "window_size": ([2048, 2048])},
    file=config.dir_path + fname,
)
