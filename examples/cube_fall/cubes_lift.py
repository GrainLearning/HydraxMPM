"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

fname = "/cubes_lift.gif"


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
block1 = create_block((3, 1), 1.5, particle_spacing)
block2 = create_block((7.5, 7), 1.5, particle_spacing)
block3 = create_block((1, 7), 1.5, particle_spacing)
block4 = create_block((5, 7), 1.5, particle_spacing)

# Stack all the positions together
position_stack = np.vstack([block1, block2, block3, block4])
# position_stack = np.vstack([block1, block3])

config = hdx.MPMConfig(
    origin=[0.0, 0.0],
    end=[domain_size, domain_size],
    cell_size=cell_size,
    num_points=len(position_stack),
    shapefunction="cubic",
    ppc=particles_per_cell,
    num_steps=120000,
    store_every=1000,
    # num_steps=10,
    # store_every=1,
    dt=0.003,
)


config.print_summary()


rigid_x = jnp.arange(0, domain_size, particle_spacing / 2)

rigid_pos_stack = jnp.zeros((len(rigid_x), 2))

rigid_pos_stack = rigid_pos_stack.at[:, 0].set(rigid_x)


y0 = 1
rigid_pos_stack = rigid_pos_stack.at[:, 1].set(y0)

rigid_velocity_sack = jnp.zeros((len(rigid_x), 2)).at[:, 1].set(0.05)


particles = hdx.Particles(config=config, position_stack=position_stack)

nodes = hdx.Nodes(config)
particles, nodes = hdx.discretize(config, particles, nodes, density_ref=1000)


material = hdx.LinearIsotropicElastic(E=10000.0, nu=0.1, config=config)

gravity = hdx.Gravity(config=config, gravity=jnp.array([0.00, -0.0098]))

box = hdx.NodeLevelSet(config, mu=0.2)

A = 1  # amplitude deviation of peak
omega = 0.2  # rate of change [rad/s]


def update_rigid_particles(step, position_stack, velocity_stack, config):
    new_position_stack = position_stack + velocity_stack * config.dt

    t = step * config.dt

    def vmap_sinusoid(pos):
        y = A * jnp.sin(omega * t)
        return pos.at[1].set(y) + jnp.zeros(2).at[1].set(y0)

        # return pos.at[1].set(pos.at[1].get() +0.05)

    next_position_stack = jax.vmap(vmap_sinusoid, in_axes=0)(new_position_stack)
    # next_position_stack = new_position_stack
    new_velocity_stack = (next_position_stack - new_position_stack) / config.dt

    # # predicted_pos =
    return new_position_stack, new_velocity_stack


rigid_particle_wall = hdx.RigidParticles(
    config=config,
    position_stack=rigid_pos_stack,
    mu=0.5,
    # velocity_stack=rigid_velocity_sack,
    update_rigid_particles=update_rigid_particles,
)

solver = hdx.USL(config, alpha=0.99)

print("Running and compiling")

carry, accumulate = hdx.run_solver(
    config=config,
    solver=solver,
    particles=particles,
    nodes=nodes,
    material_stack=[material],
    forces_stack=[gravity, box, rigid_particle_wall],
    particles_output=("stress_stack", "position_stack", "velocity_stack", "mass_stack"),
    forces_output=("position_stack",),
)

print("Simulation done.. plotting might take a while")

stress_stack, position_stack, velocity_stack, mass_stack, rigid_positions_stack = (
    accumulate
)

p_stack = jax.vmap(hdx.get_pressure_stack, in_axes=(0, None))(stress_stack, 2)


pvplot_cmap_q = hdx.PvPointHelper(
    config=config,
    position_stack=position_stack,
    scalar_stack=p_stack,
    scalar_name="p [Pa]",
    subplot=(0, 0),
    timeseries_options={
        "clim": [0, 1000],
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

pvplot_rigid = hdx.PvPointHelper(
    config=config,
    position_stack=rigid_positions_stack,
    subplot=(0, 0),
    timeseries_options={
        "color": "red",
        "point_size": 25,
        "render_points_as_spheres": True,
    },
)

plotter = hdx.make_pvplots(
    config,
    [pvplot_cmap_q, pvplot_rigid],
    plotter_options={"shape": (1, 1), "window_size": ([2048, 2048])},
    file=config.dir_path + fname,
)
