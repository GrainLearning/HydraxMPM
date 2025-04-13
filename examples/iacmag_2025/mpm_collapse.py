"""Granular column collapse"""

import jax
import jax.numpy as jnp


print(jax.devices("gpu"))
jax.config.update("jax_default_device", jax.devices("gpu")[0])

import hydraxmpm as hdx

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

column_width = 0.2  # [m]
column_height = 0.1  # [m]

domain_width = 0.6  # [m]
domain_height = 0.11  # [m]

cell_size = 0.0025  # [m]

ppc = 2  # particles per cell


sep = cell_size / ppc


x = jnp.arange(0, column_width, sep) + 2 * sep

y = jnp.arange(0, column_height, sep)

xv, yv = jnp.meshgrid(x, y)

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))


fric_angle = 19.8  # [degrees]
rho_0 = 2650.0  # [kg/m^3]
K = 7e5  # [Pa]

# matching Mohr-Coulomb criterion under triaxial extension conditions
fric_angle_rad = jnp.deg2rad(fric_angle)
mu = (
    6 * jnp.sin(fric_angle_rad) / (jnp.sqrt(3) * (3 + jnp.sin(jnp.deg2rad(fric_angle))))
)

# select model here
model_index = 0

models = (
    hdx.DruckerPrager(
        nu=0.3,
        K=K,
        mu_1=mu,
        rho_0=rho_0,
        rho_p=rho_0,
        other=dict(project="dp"),
    ),
    hdx.ModifiedCamClay(
        nu=0.3,
        M=mu * jnp.sqrt(3),
        lam=0.0058,
        kap=0.0012,
        R=1.0,
        # p=1Pa reference limit
        rho_0=rho_0,
        rho_p=rho_0,
        other=dict(project="mcc"),
    ),
    hdx.MuI_incompressible(
        mu_s=mu,
        mu_d=2.9,
        I_0=0.279,
        K=K,
        d=0.00125,
        rho_p=rho_0,
        rho_0=rho_0,
        other=dict(project="mu_i"),
    ),
    # fluid becomes unstable
    # so we damp numericallly
    # and increase time step
    hdx.NewtonFluid(
        other=dict(project="fluid", alpha=0.9, dt_alpha=0.01),
        K=K,
        viscosity=0.002,
        alpha=7.0,
        rho_0=rho_0,
    ),
)


solver = hdx.USL_ASFLIP(
    alpha=models[model_index].other.get("alpha", 0.95),
    shapefunction="cubic",
    dim=2,
    ppc=ppc,
    material_points=hdx.MaterialPoints(position_stack=position_stack, p_stack=0.0),
    grid=hdx.Grid(
        origin=[0.0, 0.0],
        end=[domain_width, domain_height],
        cell_size=cell_size,
    ),
    constitutive_laws=models[model_index],
    forces=(
        hdx.SlipStickBoundary(x0="slip", x1="slip", y0="stick", y1="slip"),
        hdx.Gravity(gravity=jnp.array([0.0, -9.8])),
    ),
    output_vars=dict(
        material_points=(
            "p_stack",
            "position_stack",
            # uncomment to save
            # "KE_stack",
            # "viscosity_stack",
            # "dgamma_dt_stack",
            # "eps_v_stack",
            # "specific_volume_stack",
            # "gamma_stack",
            # "q_p_stack",
            # "q_stack",
            # "rho_stack",
        ),
        shape_map=(
            "grid_position_stack",
            "grid_mesh",
            "p2g_gamma_stack",
            "p2g_KE_stack",
            "p2g_q_p_stack",
        ),
    ),
)


solver = solver.setup()


output_dir = os.path.join(
    dir_path, "output/{}".format(models[model_index].other["project"])
)

solver = solver.run(
    output_dir=output_dir,
    total_time=1.0,
    store_interval=0.01,
    adaptive=True,
    override_dir=True,
    dt_alpha=models[model_index].other.get("dt_alpha", 0.1),
    dt=1e-5,
    dt_max=1e-5,  # at least 1e-5
)


hdx.viewer.view(output_dir, ["p_stack", "gamma_stack"])


hdx.npz_to_vtk(
    input_folder=output_dir,
    output_folder=output_dir,
    remove_word_stack=False,
    verbose=False,
    kind=["shape_map", "material_points"],
)
