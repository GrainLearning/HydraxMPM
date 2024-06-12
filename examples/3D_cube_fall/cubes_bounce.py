"""3D cube falling"""
import jax.numpy as jnp
import numpy as np
import jax
import pymudokon as pm

import pyvista as pv

domain_size = 10

particles_per_cell = 2
cell_size = (1 / 20) * domain_size

output_steps = 100
output_start = 0
total_steps = 11000


particle_spacing = cell_size / particles_per_cell

print("Creating simulation")

def create_block(block_start, block_size, spacing):
    """Create a block of particles in 3D space."""
    block_end = (block_start[0] + block_size, block_start[1] + block_size, block_start[1] + block_size)
    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    z = np.arange(block_start[2], block_end[2], spacing)
    block = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return block

# Create two blocks (cubes in 2D context)
block1 = create_block((3, 3, 3), 2, particle_spacing) + np.array([0.0, 0.0, 4.0])

block2 = create_block((3, 3, 3), 2, particle_spacing)

pos = np.vstack([block1, block2])

print(pos.shape)
vels = jnp.zeros_like(pos).at[:len(block1),2].set(-0.2).at[len(block1):,2].set(0.2)

particles = pm.Particles.create(
    positions=pos,
    velocities=vels,
    original_density=1000)



nodes = pm.Nodes.create(
    origin=jnp.array([0.0, 0.0, 0.0]),
    end=jnp.ones(3)*domain_size,
    node_spacing=cell_size,
    small_mass_cutoff=1e-3
)

shapefunctions = pm.LinearShapeFunction.create(len(pos), 3)

particles, nodes, shapefunctions = pm.discretize(particles, nodes, shapefunctions)

material = pm.LinearIsotropicElastic.create(E=1000.0, nu=0.3, num_particles=len(pos))

gravity = pm.Gravity.create(gravity=jnp.array([0.000, 0.0, -0.0098]))

box = pm.DirichletBox.create(nodes)


usl = pm.USL.create(
    particles=particles,
    nodes=nodes,
    materials=[material],
    # forces=[gravity,box],
    shapefunctions=shapefunctions,
    alpha=0.99,
    dt=0.001,

)

def debug_particles(
        step: jnp.int32, 
        particles: pm.Particles,
        stress_limit: jnp.float32 = 1e6,
    ):
    """
    First challenge is to narow down the iteration.
    
    Second challenge is the find function causing the error.

    Third challenge is the find the memory location of the error.
    """
    
    # Check out of bounds
    positions = usl.particles.positions
    out_of_bounds = jnp.any(jnp.logical_or(positions < nodes.origin, positions > nodes.end))
    if out_of_bounds:
        print(f"Instability detected: Particles out of bounds at step at step {step}")
        exit(0)
    # Check for NaN or Inf values
    if jnp.any(jnp.isnan(particles.stresses)) or jnp.any(jnp.isinf(particles.stresses)):
        print(f"Instability detected: NaN or Inf value in stress at step {step}")
        exit(0)
    # Check for extreme values
    if jnp.max(jnp.abs(particles.stresses)) > stress_limit:
        print(f"Instability detected: Stress exceeds limit at step {step}")
        exit(0)

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