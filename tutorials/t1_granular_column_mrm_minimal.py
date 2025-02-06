"""Tutorial 1. Granular column collapse"""

import jax
import jax.numpy as jnp
import hydraxmpm as hdx

##### this must be in tutorial
column_width = 0.3  # [m]
column_height = 0.4  # [m]
ppc = 2

cell_size = 0.025  # [m]
sep = cell_size / ppc
padding = 4.5 * 0.0


x = jnp.arange(0, column_width + sep, sep) + sep + 1.2 - column_width / 2
y = jnp.arange(0, column_height + sep, sep) + sep
xv, yv = jnp.meshgrid(x, y)

position_stack = jnp.array(list(zip(xv.flatten(), yv.flatten())))

density_ref = 0.7 * 1400

g = -9.8

K = 1.0


def get_stress(pos):
    h = pos.at[1].get()

    sigma_y = density_ref * g * h

    sigma_x = sigma_z = K * sigma_y

    stress = jnp.eye(3)

    stress = stress.at[0, 0].set(sigma_x)
    stress = stress.at[1, 1].set(sigma_y)
    stress = stress.at[2, 2].set(sigma_z)
    return stress


stress_stack = jax.vmap(get_stress)(position_stack)

pressure_stack = hdx.get_pressure_stack(stress_stack)


# # gravity ramp
# stop_ramp_step = 60000

# increment = jnp.array([0.0, -9.8]) / stop_ramp_step
ps = (0.0022 / 100) / 0.33
print(ps)
# solver
solver = hdx.USL_ASFLIP(
    config=hdx.Config(
        # num_steps=500,
        # num_steps=2,
        # store_every=1,
        num_steps=10000,
        store_every=500,
        shapefunction="cubic",
        dim=2,
        ppc=ppc,
        project="mrm",
        output=dict(particles=("pressure_stack", "density_stack")),
        dt=3 * 10**-5,  # time step[s]
        file=__file__,  # store current location of file for output
    ),
    particles=hdx.Particles(
        position_stack=position_stack,
        rho_p=1400.0,
        density_ref=density_ref,
        # p_ref=pressure_stack,
    ),
    grid=hdx.Grid(
        origin=[0.0, 0.0],
        end=[2.4, 0.5],
        cell_size=cell_size,
    ),
    materials=hdx.MRMMinimal(
        mu_s=1.2 / jnp.sqrt(3),
        mu_d=1.8 / jnp.sqrt(3),
        I_0=0.279,
        d=0.0053,
        I_phi_c=0.85,
        # ps=ps,
        ps=0.33,
        # ps=3.3,
        phi_c=0.648,
        lam=0.0186,
    ),
    forces=(
        hdx.Boundary(mu=0.7),
        hdx.Gravity(gravity=jnp.array([0.0, -9.8])),
    ),
    callbacks=hdx.io_helper_vtk(),
)

solver = solver.setup()

solver = solver.run()
