import jax.numpy as jnp
import numpy as np

from hydraxmpm import (
    USL_APIC,
    USL,
    Boundary,
    Config,
    Gravity,
    Grid,
    NewtonFluid,
    Particles,
    io_helper_vtk,
)

K = 2.0 * 10**6
rho = 997.5
cell_size = 6 / 69
ppc = 4
sep = cell_size / ppc

# time step depends on the cell_size, bulk modulus and initial density
dt = 0.1 * cell_size / np.sqrt(K / rho)

# create dam
dam_height = 2.0
dam_length = 4.0

x = np.arange(0, dam_length + sep, sep) + 5.5 * sep
y = np.arange(0, dam_height + sep, sep) + 5.5 * sep

# create a grid
xv, yv = np.meshgrid(x, y)

pnts_stack = np.array(list(zip(xv.flatten(), yv.flatten()))).astype(np.float64)

usl_apic = USL(
    config=Config(
        shapefunction="cubic",
        ppc=4,
        num_steps=20000,
        store_every=500,
        file=__file__,
        dt=float(dt),
        dim=2,
        output=dict(
            particles=(
                "velocity_stack",
                "pressure_stack",
            )
        ),
    ),
    particles=Particles(position_stack=jnp.array(pnts_stack), density_ref=997.5),
    grid=Grid(origin=[0.0, 0.0], end=[6.0, 6.0], cell_size=cell_size),
    materials=NewtonFluid(K=K, viscosity=0.002),
    forces=(Boundary(), Gravity(gravity=[0.0, -9.81])),
    callbacks=io_helper_vtk(),
)

usl_apic = usl_apic.setup()


# usl_apic.forces[0].id_stack

usl_apic.run()
