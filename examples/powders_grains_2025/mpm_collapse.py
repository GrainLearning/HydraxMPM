"""
Simulates a 2D granular column collapse using the Material Point Method (MPM).


Modified Cam Clay starts  with a K0 coefficient of 0.5,

while mu (I) rheology starts on the steady state.


"""

import os

import jax
import jax.numpy as jnp

# --- Configuration ---
# JAX Device Setup (Optional: Adjust device index if needed)
print("Available JAX devices:", jax.devices("gpu"))
# Select the second GPU (index 1). Change if you have a different setup.
jax.config.update("jax_default_device", jax.devices("gpu")[1])

import hydraxmpm as hdx

# --- Simulation Parameters ---
dt = 3 * 10**-6  # time step[s]


# --- Domain ---
domain_width = 3.2  # [m]
domain_height = 1.0  # [m]

# --- Column Dimensions ---
column_width = 0.3  # [m]
column_height = 0.9  # [m]

# --- Discretization ---
ppc = 2  # Particles per cell (in each direction
cell_size = 0.00625  # Grid cell size [m]
sep = cell_size / ppc  # Particle spacing [m]

# --- Material Point Initialization ---
# Create particle positions for the initial column geometry
# Positioned slightly off the boundary to ensure proper shape function coverage.
x_coords = (
    jnp.arange(0, column_width, sep) + domain_width / 2 - column_width / 2
) + 2 * sep

y_coords = jnp.arange(0, column_height + sep * 2, sep) + sep
xv, yv = jnp.meshgrid(x_coords, y_coords)

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))

# Shared parameters (used by multiple models)
# rho_g = 2400.0  # [kg/m^3]
rho_g = 1500.0  # [kg/m^3]
rho_p = 2450.0  # [kg/m^3]
d = 0.005

# matching Mohr-Coulomb criterion under triaxial extension conditions

M = 1.2
mu_0 = M / jnp.sqrt(3)

# Modified Cam-Clay specific parameters
lam = 0.04  # [-]
kap = 0.008  # [-]
nu = 0.2  # [-]

# Mu(I) rheology specific parameters
mu_d = 5.4 * mu_0
K = 392000.0  # [Pa]
I_0 = 0.279


# --- Define Constitutive Models ---
# Select which model to use by index (0, 1, or 2)
model_index = 1
# A tuple containing instances of the different models to test.
# 'other' dictionary is used to add metadata, like project name for output folders.
model = (
    hdx.ModifiedCamClay(
        nu=nu,
        M=M,
        lam=lam,
        kap=kap,
        R=1.0,
        rho_0=rho_g,
        rho_p=rho_p,
        other=dict(project="mcc_ocr1"),
    ),
    hdx.ModifiedCamClay(
        nu=nu,
        M=M,
        lam=lam,
        kap=kap,
        R=4.0,
        rho_0=rho_g,
        rho_p=rho_p,
        other=dict(project="mcc_ocr4"),
        settings=dict(throw=False)
    ),
    hdx.MuI_incompressible(
        mu_s=mu_0,
        mu_d=mu_d,
        I_0=I_0,
        K=K,
        d=d,
        rho_p=rho_p,
        rho_0=rho_g,
        other=dict(project="mu_i"),
    ),
)[model_index]


# --- Initial Stress Calculation ---
# applying a lithostatic pressure
def get_lithostatic_pressure(h, rho_g, g):
    # material points are slightly above y=0
    # to account for shape function support
    return h * rho_g * g


y_stack = position_stack.at[:, 1].get()

sigma_h_stack = jax.vmap(get_lithostatic_pressure, in_axes=(0, None, None))(
    (2 * cell_size + y_stack.max()) - y_stack, rho_g, 9.8
)


if isinstance(model, hdx.ModifiedCamClay):

    def give_stress(sigma_yy):
        stress = jnp.zeros((3, 3))
        sigma_xx = sigma_yy
        sigma_zz = nu * (sigma_yy + sigma_xx)
        stress = stress.at[0, 0].set(-sigma_xx)
        stress = stress.at[1, 1].set(-sigma_yy)
        stress = stress.at[2, 2].set(-sigma_zz)
        return stress

    stress_stack = jax.vmap(give_stress)(sigma_h_stack)
else:

    def give_stress(sigma_h):
        stress = jnp.zeros((3, 3))
        stress = stress.at[0, 0].set(-sigma_h)
        stress = stress.at[1, 1].set(-sigma_h)
        stress = stress.at[2, 2].set(-sigma_h)
        return stress

    # For mu(I) rheology
    stress_stack = jax.vmap(give_stress)(sigma_h_stack)


solver = hdx.USL_ASFLIP(
    # FLIP/PIC blending factor
    alpha=0.995,
    # shape function type
    shapefunction="cubic",
    # Plane strain
    dim=2,
    # Number of material points per cell
    ppc=ppc,
    # Specify quantities to save during simulation
    output_vars=dict(
        material_points=(
            "p_stack",
            "position_stack",
        ),
        shape_map=(
            "p2g_gamma_stack",
            "grid_position_stack",
        ),
    ),
    # Material Points and grid initialization
    material_points=hdx.MaterialPoints(
        position_stack=position_stack, stress_stack=stress_stack
    ),
    grid=hdx.Grid(
        origin=[0.0, 0.0],
        end=[domain_width, domain_height],
        cell_size=cell_size,
    ),
    # Constitutive model initialization
    constitutive_laws=model,
    # Forces acting on the grid and material points
    forces=(
        hdx.Boundary(mu=0.7),  # frictional floor,
        hdx.Gravity(gravity=jnp.array([0.0, -9.8])),
    ),
)

# --- Setup and Execution ---
# Allocate internal data structures
solver = solver.setup()


# current directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

output_dir = os.path.join(dir_path, "output/{}".format(model.other["project"]))


solver = solver.run(
    output_dir=output_dir,
    total_time=1.5,  # Total simulation time [s]
    store_interval=0.01,  # How often to save output
    adaptive=False,
    override_dir=True,  # Overwrite existing output in the directory
    dt=dt,
)

# --- Post-processing ---
# Visualize results using the built-in viewer
hdx.viewer.view(output_dir, ["p_stack"])


# Convert saved NumPy (.npz) files to VTK format for other visualization tools
hdx.npz_to_vtk(
    input_folder=output_dir,
    output_folder=output_dir,
    kind=["shape_map", "material_points"],  # Which data types to convert
)
