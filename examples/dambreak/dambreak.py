import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

output_path = dir_path + "/output/"

K = 2.0 * 10**6
rho = 997.5
cell_size = 6 / 69
ppc = 4
sep = cell_size / ppc

# create dam
dam_height = 2.0
dam_length = 4.0

x = np.arange(0, dam_length + sep, sep) + 5.5 * sep
y = np.arange(0, dam_height + sep, sep) + 5.5 * sep

# create a grid
xv, yv = np.meshgrid(x, y)

pnts_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

usl_apic = hdx.USL_APIC(
    shapefunction="cubic",
    ppc=4,
    dim=2,
    output_vars=dict(
        material_points=(
            "position_stack",
            "velocity_stack",
            "p_stack",
        )
    ),
    material_points=hdx.MaterialPoints(position_stack=jnp.array(pnts_stack)),
    grid=hdx.Grid(origin=[0.0, 0.0], end=[6.0, 6.0], cell_size=cell_size),
    constitutive_laws=hdx.NewtonFluid(K=K, viscosity=0.002, rho_0=997.5),
    forces=(
        hdx.SlipStickBoundary(x0="slip", x1="slip", y0="slip", y1="slip"),
        hdx.Gravity(gravity=[0.0, -9.81]),
    ),
)

usl_apic = usl_apic.setup()


# This saves the output to ./output/*.npz
usl_apic = usl_apic.run(
    total_time=2.0,
    adaptive=True,
    store_interval=0.1,
    dt_alpha=0.05,
    output_dir=output_path,
    override_dir=True,
)

# Visualize the results
hdx.viewer.view(
    output_path,
    ["KE_stack"],
)
