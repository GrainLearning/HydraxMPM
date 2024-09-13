"""Column collapse with small uniform pressure and solid volume fraction"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pymudokon as pm
import pickle
import os
import time
import pyvista as pv

dir_path = os.path.dirname(os.path.realpath(__file__))

# cell_size = 0.0125  # [m]
# dt = 3 * 10**-5 # [s] time step

# uncomment these for higher resolution
dt = 3 * 10**-6 
cell_size = 0.0125/2 

particles_per_cell = 4 
sep = cell_size / particles_per_cell 

# granular column
aspect = 2.0 
column_width = 0.2  # [m]
column_height = column_width * aspect  # [m]

# domain
domain_height = column_height * 1.4  # [m]

# add pading for cubic shape functions
# domain_length = column_width + (5 * sep)*2 # [m] 
domain_length = 6 * column_width  # [m]


# material parameters
phi_c = 0.648 # [-] rigid limit
phi_0 = 0.65 # [-] initial solid volume fraction
rho_p = 2000 # [kg/m^3] particle (skeletan density)
rho = rho_p*phi_0 #[kg/m^3] bulk density
d=0.0053 # [m] particle diameter


# gravity
g = -9.8  # [m/s^2]

num_steps=400000
store_every=1000



# create domain and background grid
origin, end = jnp.array([0.0, 0.0]), jnp.array([domain_length, domain_height])
nodes = pm.Nodes.create(
    origin=origin, end=end, node_spacing=cell_size, small_mass_cutoff=1e-12
)

# create column of particles with padding
x = np.arange(0, column_width + sep, sep) + 4 * sep
y = np.arange(0, column_height + sep, sep) + 7.5 * sep
xv, yv = np.meshgrid(x, y)

# h_shift = domain_length/2 - column_width/2
# x += h_shift

pnt_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

particles = pm.Particles.create(position_stack=jnp.array(pnt_stack))

# create shapefunctions and discretize domain ( get volume of particles )
shapefunctions = pm.CubicShapeFunction.create(len(pnt_stack), 2)

particles, nodes, shapefunctions = pm.discretize(
    particles, nodes, shapefunctions, ppc=particles_per_cell, density_ref=rho
)

print(f"num_nodes = {nodes.num_nodes_total}, num_particles = {particles.position_stack.shape[0]}",)

# initialize material and particles, perhaps there is a less verbose way?
# get reference solid volume fraction particle mass  /volume
phi_ref_stack = particles.get_phi_stack(rho_p)

material = pm.MuI_incompressible.create(
    mu_s=0.38186285175,
    mu_d=0.57176986,
    I_0=0.279,
    rho_p=rho_p,
    d=0.0053,
    K=50*rho*9.8*column_height,
    dim=2
)

def get_stress_ref(phi_ref):
    p_ref = material.get_p_ref(phi_ref)
    return -p_ref*jnp.eye(3)

stress_ref_stack = jax.vmap(get_stress_ref)(phi_ref_stack)

particles = particles.replace(stress_stack=stress_ref_stack)

# initialize fources and boundary conditions
gravity = pm.Gravity.create(gravity=jnp.array([0.0, g]))

# floor is rough boundary and walls are slip boundaries
box = pm.DirichletBox.create(
    nodes,
    boundary_types=(
        ("slip_negative_normal", "slip_positive_normal"),
        ("stick", "stick"), 
    ),
)

solver = pm.USL_APIC.create(cell_size, dim=2, num_particles=len(pnt_stack), dt=dt)
start_time = time.time()

bbox = pv.Box(bounds=np.array(list(zip(jnp.pad(origin,[0,1]), jnp.pad(end,[0,1])))).flatten())
bbox.save(dir_path + f"/output/mu_i_flat/bbox.vtk")

# save restart file
def io_vtk(carry,step):

    (
    solver,
    particles,
    nodes,
    shapefunctions,
    material_stack,
    forces_stack,
    ) = carry

    cloud = pv.PolyData(pm.points_to_3D(particles.position_stack, 2))
    
    stress_reg_stack = pm.post_processes_stress_stack(
    particles.stress_stack,
    particles.mass_stack,
    particles.position_stack,
    nodes,
    shapefunctions
    )
    q_reg_stack = pm.get_q_vm_stack(stress_reg_stack,dim=2)
    cloud["q_reg_stack"] = q_reg_stack
    phi_stack = particles.get_phi_stack(rho_p)
    
    cloud["phi_stack"] = phi_stack
    
    jax.debug.print("step {}",step)
    cloud.save(dir_path + f"/output/mu_i_flat/particles_{step}.vtk")

    

# Run solver
carry = pm.run_solver_io(
    solver=solver,
    particles=particles,
    nodes=nodes,
    shapefunctions=shapefunctions,
    material_stack=[material],
    forces_stack=[gravity, box],
    num_steps=num_steps,
    store_every=store_every,
    callback =io_vtk
)
print("--- %s seconds ---" % (time.time() - start_time))
(
    solver,
    particles,
    nodes,
    shapefunctions,
    material_stack,
    forces_stack,
) = carry


print("Simulation done.. plotting might take a while")
