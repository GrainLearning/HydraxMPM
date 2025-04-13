"""This example is coming back soon"""

# import jax
# import jax.numpy as jnp
# import numpy as np

# import hydraxmpm as hdx


# import os

# domain_size = 10.0

# particles_per_cell = 2
# cell_size = (1 / 80.0) * domain_size


# particle_spacing = cell_size / particles_per_cell

# dir_path = os.path.dirname(os.path.realpath(__file__))

# output_path = dir_path + "/output/"

# print("Creating simulation")


# def create_block(block_start, block_size, spacing):
#     """Create a block of particles in 2D space."""
#     block_end = (block_start[0] + block_size, block_start[1] + block_size)
#     x = np.arange(block_start[0], block_end[0], spacing)
#     y = np.arange(block_start[1], block_end[1], spacing)
#     block = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
#     return block


# # Create two blocks (cubes in 2D context)
# block1 = create_block((1, 1), 2, particle_spacing)
# block2 = create_block((7.5, 6.3), 2, particle_spacing)
# block3 = create_block((2.8, 7), 2, particle_spacing)
# block4 = create_block((5, 3.8), 2, particle_spacing)

# # # Stack all the positions together
# position_stack = jnp.vstack([block1, block2, block3, block4])

# solver = hdx.USL(
#     output_vars=dict(
#         material_points=(
#             "position_stack",
#             "velocity_stack",
#             "KE_stack",
#         )
#     ),
#     ppc=particles_per_cell,
#     shapefunction="cubic",
#     dim=2,
#     grid=hdx.Grid(
#         cell_size=cell_size,
#         origin=[0.0, 0.0],
#         end=[domain_size, domain_size],
#     ),
#     constitutive_laws=hdx.LinearIsotropicElastic(E=10000.0, nu=0.1, rho_0=1000.0),
#     material_points=hdx.MaterialPoints(position_stack=position_stack),
#     forces=(
#         hdx.Gravity(gravity=[0.00, -0.0098]),
#         hdx.Boundary(mu=0.2),
#     ),
# )

# solver = solver.setup()

# solver.run(
#     total_time=360.0,
#     dt=0.003,
#     adaptive=False,
#     store_interval=3.0,
#     output_dir=output_path,
#     override_dir=True,
# )


# # Visualize the results
# hdx.viewer.view(
#     output_path,
#     ["KE_stack"],
# )
