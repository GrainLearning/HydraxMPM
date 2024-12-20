"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

fname = "/cubes_fall.gif"

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 80) * domain_size


particle_spacing = cell_size / particles_per_cell

print("Creating simulation")


def create_block(block_start, block_size, spacing):
    """Create a block of particles in 2D space."""
    block_end = (block_start[0] + block_size, block_start[1] + block_size)
    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    block = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return block


# Create two blocks (cubes in 2D context)
block1 = create_block((1, 1), 2, particle_spacing)
block2 = create_block((7.5, 6.3), 2, particle_spacing)
block3 = create_block((2.8, 7), 2, particle_spacing)
block4 = create_block((5, 3.8), 2, particle_spacing)

# Stack all the positions together
position_stack = np.vstack([block1, block2, block3, block4])

config = hdx.MPMConfig(
    origin=[0.0, 0.0],
    end=[domain_size, domain_size],
    cell_size=cell_size,
    num_points=len(position_stack),
    shapefunction="cubic",
    ppc=particles_per_cell,
    num_steps=120000,
    store_every=1000,
    dt=0.003,
)

config.print_summary()

particles = hdx.Particles(config=config, position_stack=position_stack)

nodes = hdx.Nodes(config)

particles, nodes = hdx.discretize(config, particles, nodes, density_ref=1000)

material = hdx.LinearIsotropicElastic(E=10000.0, nu=0.1, config=config)

gravity = hdx.Gravity(config=config, gravity=jnp.array([0.00, -0.0098]))

box = hdx.NodeLevelSet(config, mu=0.2)

solver = hdx.USL(config, alpha=0.99)


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
