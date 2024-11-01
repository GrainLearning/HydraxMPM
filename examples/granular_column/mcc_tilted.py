"""Column collapse with small uniform pressure and solid volume fraction"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

import pymudokon as pm


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


num_steps=300000
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

material = pm.ModifiedCamClay.create_from_phi_ref(
    nu=0.3,M=0.38186285175*np.sqrt(3), R=1, lam=0.0186, kap=0.0010,rho_p=rho_p, phi_c=phi_c, phi_ref_stack=phi_ref_stack
)


stress_ref_stack = material.stress_ref_stack

particles = particles.replace(stress_stack=stress_ref_stack)

# initialize fources and boundary conditions
theta = jnp.radians(35)  # rotate 180 degrees
rot_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
gravity=jnp.array([0.0, g])
gravity_end = rot_matrix@gravity

gravity = pm.Gravity.create(gravity=gravity_end)

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

bbox = pv.Box(bounds=np.array(list(zip(origin, end))).flatten())
bbox.save(dir_path + f"/output/mu_i_inclined_collapse/bbox.vtk")
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
    pressure_reg_stack =  pm.get_pressure_stack(stress_reg_stack,dim=2)
    pressure_stack =  pm.get_pressure_stack(particles.stress_stack,dim=2)
    phi_stack = particles.get_phi_stack(rho_p)
    
    cloud["pressure_reg_stack"] = pressure_reg_stack
    cloud["pressure_stack"] = pressure_stack
    cloud["density_stack"] = particles.mass_stack/particles.volume_stack
    
    cloud["phi_stack"] = phi_stack
    
    jax.debug.print("step {}",step)
    cloud.save(dir_path + f"/output/mcc_inclined_collapse/particles_{step}.vtk")

    

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
