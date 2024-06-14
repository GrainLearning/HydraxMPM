"""3D cube falling"""
import jax.numpy as jnp
import numpy as np
import jax
import pymudokon as pm

import pyvista as pv

domain_size = 10

particles_per_cell = 4
cell_size = (1 / 20) * domain_size


output_steps = 1000
output_start = 0
total_steps = 21000

particle_spacing = cell_size / particles_per_cell

print("Creating simulation")

def create_block(block_start, block_end, spacing):
    """Create a block of particles in 3D space."""

    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    z = np.arange(block_start[2], block_end[2], spacing)
    block = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return block

# Create two blocks (cubes in 2D context)
pos = create_block((2, 2, 2), (4, 4, 4), particle_spacing)

particles = pm.Particles.create(positions=pos, original_density=1000)

nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0, 0.0]),
    end=jnp.ones(3)*domain_size,
    node_spacing=cell_size,
    small_mass_cutoff=1e-6
)

shapefunctions = pm.CubicShapeFunction.create(len(pos), 3)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos))

gravity = pm.Gravity.create(gravity=jnp.array([0.000, 0.0, -0.0098]))

box = pm.DirichletBox.create(nodes)

usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[material],
    forces=[gravity,box],
    shapefunctions=shapefunctions,
    alpha=0.95,
    dt=0.001,

)

points_data_dict = {
    "points" : [],
    "KE":[]
}
@jax.tree_util.Partial
def save_particles(package):
    global points_data_dict
    step, usl = package

    points_data_dict["points"].append(usl.particles.positions)

    KE = pm.get_KE( usl.particles.masses,usl.particles.velocities,)
    points_data_dict["KE"].append(KE.__array__())

    print(f"output {step}", end="\r")

print("Running simulation")

usl = usl.solve(num_steps=total_steps, 
                output_start_step = output_start,
                output_step=output_steps,
                output_function=save_particles)


for key, value in points_data_dict.items():
    points_data_dict[key] = np.array(value)

pm.plot_simple_3D(
    points_data_dict,
    origin=jnp.array([0.0, 0.0, 0.0]),
    end=jnp.array([domain_size, domain_size, domain_size]),
    output_file="output.gif",
    plot_params={
        "scalars": "KE",
        "clim": [0.5, 1.5],
        "render_points_as_spheres": True,
        "point_size": 10,
        "opacity": 0.5
    },
    camera_params = {
        "zoom":1.1
    }
)