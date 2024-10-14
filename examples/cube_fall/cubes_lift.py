"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""


import os

import jax
import jax.numpy as jnp
import numpy as np

import pymudokon as pm


dir_path = os.path.dirname(os.path.realpath(__file__))

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
block1 = create_block((3, 1), 2, particle_spacing)
block2 = create_block((7.5, 7), 2, particle_spacing)
block3 = create_block((1, 7), 2, particle_spacing)
block4 = create_block((5, 7), 2, particle_spacing)

# Stack all the positions together
pos = np.vstack([block1, block2, block3, block4])

rigid_x = jnp.arange(0, domain_size / 1.5, particle_spacing / 2)

rigid_pos_stack = jnp.zeros((len(rigid_x), 2))

rigid_pos_stack = rigid_pos_stack.at[:, 0].set(rigid_x)

rigid_pos_stack = rigid_pos_stack.at[:, 1].set(0.5)

rigid_velocity_sack = jnp.zeros((len(rigid_x), 2)).at[:, 1].set(0.05)

# Stack all the positions together
print("pos.shape", pos.shape)
particles = pm.Particles.create(position_stack=pos)

nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0]),
    end=jnp.array([domain_size, domain_size]),
    node_spacing=cell_size,
)


shapefunctions = pm.CubicShapeFunction.create(len(pos), 2)
particles, nodes, shapefunctions = pm.discretize(
    particles, nodes, shapefunctions, density_ref=1000
)

material = pm.LinearIsotropicElastic.create(E=10000.0, nu=0.1)


gravity = pm.Gravity.create(gravity=jnp.array([0.0, -0.0098]))

box = pm.DirichletBox.create(
    nodes,
    boundary_types=(
        ("slip_negative_normal", "slip_positive_normal"),
        ("stick", "stick"),
    ),
)

rigid_particle_wall = pm.RigidParticles(
    position_stack=rigid_pos_stack,
    velocity_stack=rigid_velocity_sack,
    shapefunction=pm.LinearShapeFunction.create(len(rigid_pos_stack), 2),
)

solver = pm.USL.create(alpha=0.99, dt=0.003)


carry, accumulate = pm.run_solver(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[material],
    forces_stack=[gravity, box, rigid_particle_wall],
    num_steps=60000,
    store_every=1000,
    particles_output=("position_stack", "stress_stack", "mass_stack"),
    forces_output=("position_stack",),
)


position_stack, stress_stack, mass_stack, rigid_positions_stack = accumulate

stress_reg_stack = jax.vmap(pm.post_processes_stress_stack,in_axes=(0,0,0, None,None)) (
    stress_stack,
    mass_stack,
    position_stack,
    nodes,
    shapefunctions
)

p_reg_stack = jax.vmap(pm.get_pressure_stack,in_axes=(0,None))(
    stress_reg_stack,2)


pvplot_cmap_q = pm.PvPointHelper.create(
   position_stack,
   scalar_stack = p_reg_stack,
  scalar_name="p [Pa]",
   origin=nodes.origin,
   end=nodes.end,
   subplot = (0,0),
   timeseries_options={
    "clim":[0,50000],
    "point_size":25,
    "render_points_as_spheres":True,
    "scalar_bar_args":{
           "vertical":True,
           "height":0.8,
            "title_font_size":35,
            "label_font_size":30,
            "font_family":"arial",

           }
   }
)


pvplot_rigid = pm.PvPointHelper.create(
rigid_positions_stack,
   origin=nodes.origin,
   end=nodes.end,
   subplot = (0,0),
   timeseries_options={
    "point_size":25,
    "render_points_as_spheres":True
   }
)

plotter = pm.make_pvplots(
    [pvplot_cmap_q, pvplot_rigid],
    plotter_options={"shape":(1,1),"window_size":([2048, 2048]) },
    dim=2,
    file=dir_path + fname,
)
