"""Two cubes falling featuring rough domain walls, gravity and cubic shape functions"""

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx
from hydraxmpm.forces.boundary import Boundary

domain_size = 10.0

particles_per_cell = 2
cell_size = (1 / 80.0) * domain_size


particle_spacing = cell_size / particles_per_cell

print("Creating simulation")


def create_block(block_start, block_size, spacing):
    """Create a block of particles in 2D space."""
    block_end = (block_start[0] + block_size, block_start[1] + block_size)
    x = np.arange(block_start[0], block_end[0], spacing)
    y = np.arange(block_start[1], block_end[1], spacing)
    block = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return block


# Create two blocks (cubes in 2D context)
block1 = create_block((1, 1), 2, particle_spacing)
block2 = create_block((7.5, 6.3), 2, particle_spacing)
block3 = create_block((2.8, 7), 2, particle_spacing)
block4 = create_block((5, 3.8), 2, particle_spacing)

# # Stack all the positions together
position_stack = jnp.vstack([block1, block2, block3, block4])

solver = hdx.USL(
    config=hdx.Config(
        num_steps=120000,
        dt=0.003,
        store_every=1000,
        ppc=particles_per_cell,
        shapefunction="cubic",
        dim=2,
        file=__file__,
        output=dict(particles=("pressure_stack",)),
    ),
    grid=hdx.Grid(
        cell_size=cell_size,
        origin=[0.0, 0.0],
        end=[domain_size, domain_size],
    ),
    materials=hdx.LinearIsotropicElastic(E=10000.0, nu=0.1),
    particles=hdx.Particles(position_stack=position_stack, density_ref=1000.0),
    forces=(hdx.Gravity(gravity=[0.00, -0.0098]), Boundary(mu=0.2)),
    callbacks=hdx.io_helper_vtk(),
)

solver = solver.setup()

solver.run()
