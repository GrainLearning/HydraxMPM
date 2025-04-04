"""Granular column collapse"""

import jax
import jax.numpy as jnp

jax.config.update("jax_default_device", jax.devices("gpu")[2])

import hydraxmpm as hdx


column_width = 0.3  # [m]
aspect = 3  # aspect ratio
column_height = column_width * aspect
domain_width = 3.2  # [m]

ppc = 2

cell_size = 0.025 / 4  # [m]

sep = cell_size / ppc

x = (
    jnp.arange(0, column_width + sep * 2, sep)
    + sep
    + domain_width / 2
    - column_width / 2
)
y = jnp.arange(0, column_height + sep * 2, sep) + cell_size
xv, yv = jnp.meshgrid(x, y)

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))


# Initial density of column
rho_0 = 1500
rho_p = 2450
p_0 = 1000.0


# Here we set lithostatic pressure
def get_lithostatic_pressure(h, rho_0, g):
    return h * rho_0 * g


y_stack = position_stack.at[:, 1].get()

p_stack = jax.vmap(get_lithostatic_pressure, in_axes=(0, None, None))(
    (2 * cell_size + y_stack.max()) - y_stack, rho_0, 9.8
)

# Setup the solver
solver = hdx.USL_ASFLIP(
    alpha=0.99,  # 1 energetic but unstable, 0.5 stable but damping
    config=hdx.Config(
        num_steps=40000,
        store_every=100,
        dt=3 * 10**-5,
        shapefunction="cubic",
        dim=2,
        ppc=ppc,
        output=dict(
            # output quantities stored in the material points
            # uncomment to save these to ./output
            material_points=(
                "position_stack",
                "KE_stack",
                "p_stack",
                "q_stack",
                "eps_v_stack",
                "viscosity_stack",
                "dgammadt_stack",
                "specific_volume_stack",
                "gamma_stack",
                "inertial_number_stack",
                "q_p_stack",
                "rho_stack",
            ),
            # output quantities stored in the solver
            # uncomment to save these to ./output
            solver=(
                "p2g_p_stack",
                "p2g_q_p_stack",
                "p2g_KE_stack",
                "p2g_gamma_stack",
                "p2g_position_stack",
            ),
        ),
        override_dir=True,
    ),
    material_points=hdx.MaterialPoints(position_stack=position_stack, p_stack=p_stack),
    grid=hdx.Grid(
        origin=[0.0, 0.0], end=[3.2, column_height + 0.1], cell_size=cell_size
    ),
    constitutive_laws=(
        hdx.ModifiedCamClay(
            nu=0.2,
            M=1.2,
            lam=0.025,
            kap=0.005,
            ln_N=0.7,
            d=0.005,
            rho_p=rho_p,
            R=1,
        ),
    ),
    forces=(
        hdx.Boundary(mu=0.7),
        hdx.Gravity(gravity=jnp.array([0.0, -9.8])),
    ),
)


solver = solver.setup()

solver = solver.run()


hdx.viewer.view(
    solver.config,
    [
        "p_stack",
        "KE_stack",
        "PE_stack",
        "KE_PE_Stack",
        "viscosity_stack",
        "dgammadt_stack",
        "eps_v_stack",
        "specific_volume_stack",
        "gamma_stack",
        "inertial_number_stack",
        "q_p_stack",
        "q_stack",
        "rho_stack",
    ],
)
