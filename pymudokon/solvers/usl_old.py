# """Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

# References:
#     - De Vaucorbeil, Alban, et al. 'Material point method after 25 years: theory, implementation, and applications.'
# """

# from functools import partial
# from typing import List, Tuple

# import chex
# import jax
# import jax.numpy as jnp
# from typing_extensions import Self

# from ..core.nodes import Nodes
# from ..core.particles import Particles
# from ..forces.forces import Forces
# from ..materials.material import Material
# from ..shapefunctions.shapefunction import ShapeFunction


# @chex.dataclass
# class USL(Solver):
#     """Explicit Update Stress Last (USL) MPM solver.

#     Two node moment quantities are used related to FLIP-PIC mixture. One quantity updates the total velocity and
#     the other updates the increment.

#     Example:
#     >>> import pymudokon as pm
#     >>> # create particles, nodes, shapefunctions, materials, forces ...
#     >>> usl = pm.USL.create(particles, nodes, shapefunctions, materials)
#     >>> def some_callback(package):
#     ...     usl, step = package
#     ...     # do something with usl
#     ...     # e.g., print(usl.particles.positions)
#     >>> usl = usl.solve(num_steps=10, output_function=some_callback)

#     Attributes:
#         alpha: Flip & PIC ratio. Recommended value is 0.99
#     """

#     alpha: jnp.float32

#     @classmethod
#     def create(
#         cls: Self,
#         particles: Particles,
#         nodes: Nodes,
#         shapefunctions: ShapeFunction,
#         materials: List[Material],
#         forces: List[Forces] = None,
#         alpha: jnp.float32 = 0.99,
#         dt: jnp.float32 = 0.00001,
#     ) -> Self:
#         """Create a USL solver.

#         Args:
#             cls: self reference.
#             particles: Particles in the simulation.
#             nodes: Nodes in the simulation.
#             shapefunctions: Shape functions in the simulation, e.g., `LinearShapeFunction`.
#             materials: List of materials in the simulation, e.g., `LinearIsotropicElastic`.
#             forces (optional): List of forces. Defaults to None.
#             alpha (optional): FLIP-PIC ratio. Defaults to 0.99.
#             dt (optional): Time step. Defaults to 0.00001.

#         Returns:
#             USL: Initialized USL solver.
#         """
#         if forces is None:
#             forces = []

#         return cls(
#             particles=particles,
#             nodes=nodes,
#             shapefunctions=shapefunctions,
#             materials=materials,
#             forces=forces,
#             alpha=alpha,
#             dt=dt,
#         )

#     @jax.jit
#     def update(self: Self) -> Self:
#         """Solve 1 iteration of USL MPM.

#         Args:
#             self: Current USL solver.

#         Returns:
#             Self: Updated USL solver.
#         """
#         # Reset temporary quantities
#         nodes = self.nodes.refresh()
#         particles = self.particles.refresh()

#         # Calculate shape functions and its gradients
#         shapefunctions, _ = self.shapefunctions.calculate_shapefunction(nodes=nodes, positions=particles.positions)

#         # Particle to grid transfer
#         nodes = self.p2g(nodes=nodes, particles=particles, shapefunctions=shapefunctions)

#         # Calculate forces on nodes moments
#         # for loop is statically unrolled, may result in large compile times
#         # for many materials/forces
#         forces = []
#         for force in self.forces:
#             nodes, out_force = force.apply_on_nodes_moments(
#                 particles=particles, nodes=nodes, shapefunctions=shapefunctions, dt=self.dt
#             )
#             forces.append(out_force)

#         # Grid to particle transfer
#         particles = self.g2p(particles=particles, nodes=nodes, shapefunctions=shapefunctions)

#         # Stress update using constitutive models
#         materials = []
#         for mat in self.materials:
#             particles, out_mat = mat.update_stress(particles=particles, dt=self.dt)
#             materials.append(out_mat)

#         return self.replace(
#             particles=particles,
#             nodes=nodes,
#             materials=materials,
#             forces=forces,
#             shapefunctions=shapefunctions,
#         )

#     @jax.jit
#     def p2g(self: Self, nodes: Nodes, particles: Particles, shapefunctions: ShapeFunction) -> Nodes:
#         """Particle to grid transfer function.

#         Procedure is as follows:
#         - Gather particle quantities to interactions.
#         - Scale masses, moments, and forces by shape functions.
#         - Calculate node internal force from scaled stresses, volumes.
#         - Sum interaction quantities to nodes.
#         """
#         stencil_size, dim = self.shapefunctions.stencil.shape

#         @partial(jax.vmap, in_axes=(0, 0, 0))
#         def vmap_p2g(
#             intr_id: chex.ArrayBatched, intr_shapef: chex.ArrayBatched, intr_shapef_grad: chex.ArrayBatched
#         ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
#             """Gather particle quantities to interactions."""
#             # out of scope quantities (e.g., particles., are static)
#             particle_id = (intr_id / stencil_size).astype(jnp.int32)

#             intr_masses = particles.masses.at[particle_id].get()
#             intr_volumes = particles.volumes.at[particle_id].get()
#             intr_velocities = particles.velocities.at[particle_id].get()
#             intr_ext_forces = particles.forces.at[particle_id].get()
#             intr_stresses = particles.stresses.at[particle_id].get()

#             scaled_mass = intr_shapef * intr_masses
#             scaled_moments = scaled_mass * intr_velocities
#             scaled_ext_force = intr_shapef * intr_ext_forces
#             scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

#             scaled_total_force = scaled_int_force[:dim] + scaled_ext_force

#             return scaled_mass, scaled_moments, scaled_total_force

#         # Get interaction id and respective particle belonging to interaction
#         # form a batched interaction
#         scaled_mass, scaled_moments, scaled_total_force = vmap_p2g(
#             shapefunctions.intr_ids, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad
#         )

#         # Sum all interaction quantities.
#         nodes_masses = nodes.masses.at[shapefunctions.intr_hashes].add(scaled_mass)

#         nodes_moments = nodes.moments.at[shapefunctions.intr_hashes].add(scaled_moments)

#         nodes_forces = jnp.zeros_like(nodes.moments_nt).at[shapefunctions.intr_hashes].add(scaled_total_force)

#         nodes_moments_nt = nodes_moments + nodes_forces * self.dt

#         return nodes.replace(masses=nodes_masses, moments=nodes_moments, moments_nt=nodes_moments_nt)

#     @jax.jit
#     def g2p(self: Self, particles: Particles, nodes: Nodes, shapefunctions: ShapeFunction) -> Particles:
#         """Grid to particle transfer.

#         Procedure is as follows:
#         - Calculate node velocities (by dividing moments with mass)
#         - Scatter node quantities to particle-node interactions.
#         - Calculate interaction velocities and velocity gradients.
#         - Sum interaction quantities to particles.
#         (e.g., velocities, positions, volumes, velocity gradients, deformation gradients, ...)
#         """
#         num_interactions, dim = self.shapefunctions.stencil.shape

#         padding = (0, 3 - dim)

#         @partial(jax.vmap, in_axes=(0, 0, 0))
#         def vmap_intr_scatter(
#             intr_hashes: chex.ArrayBatched, intr_shapef: chex.ArrayBatched, intr_shapef_grad: chex.ArrayBatched
#         ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
#             """Scatter quantities from nodes to interactions."""
#             # Unscopped quantities are static in JAX
#             intr_masses = nodes.masses.at[intr_hashes].get()
#             intr_moments = nodes.moments.at[intr_hashes].get()
#             intr_moments_nt = nodes.moments_nt.at[intr_hashes].get()

#             # Small mass cutoff to avoid unphysical large velocities
#             intr_vels = jax.lax.cond(
#                 intr_masses > nodes.small_mass_cutoff,
#                 lambda x: x / intr_masses,
#                 lambda x: jnp.zeros_like(x),
#                 intr_moments,
#             )

#             intr_vels_nt = jax.lax.cond(
#                 intr_masses > nodes.small_mass_cutoff,
#                 lambda x: x / intr_masses,
#                 lambda x: jnp.zeros_like(x),
#                 intr_moments_nt,
#             )
#             intr_delta_vels = intr_vels_nt - intr_vels

#             intr_scaled_delta_vels = intr_shapef * intr_delta_vels

#             intr_scaled_vels_nt = intr_shapef * intr_vels_nt

#             # Pad velocities for plane strain
#             intr_vels_nt_padded = jnp.pad(
#                 intr_vels_nt,
#                 padding,
#                 mode="constant",
#                 constant_values=0,
#             )

#             intr_scaled_velgrad = intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt_padded.reshape(-1, 1).T

#             return intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad

#         @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0))
#         def vmap_particles_update(
#             intr_delta_vels_reshaped: chex.ArrayBatched,
#             intr_vels_nt_reshaped: chex.ArrayBatched,
#             intr_velgrad_reshaped: chex.ArrayBatched,
#             p_velocities: chex.ArrayBatched,
#             p_positions: chex.ArrayBatched,
#             p_F: chex.ArrayBatched,
#             p_volumes_orig: chex.ArrayBatched,
#         ) -> Tuple[chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched, chex.ArrayBatched]:
#             """Update particle quantities by summing interaction quantities."""
#             p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

#             delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
#             vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

#             p_velocities_next = (1.0 - self.alpha) * vels_nt + self.alpha * (p_velocities + delta_vels)

#             p_positions_next = p_positions + vels_nt * self.dt

#             if dim == 2:
#                 p_velgrads_next = p_velgrads_next.at[2, 2].set(0)

#             p_F_next = (jnp.eye(3) + p_velgrads_next * self.dt) @ p_F

#             if dim == 2:
#                 p_F_next = p_F_next.at[2, 2].set(1)

#             p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig
#             return p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next

#         intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad = vmap_intr_scatter(
#             shapefunctions.intr_hashes, shapefunctions.intr_shapef, shapefunctions.intr_shapef_grad
#         )

#         p_velocities_next, p_positions_next, p_F_next, p_volumes_next, p_velgrads_next = vmap_particles_update(
#             intr_scaled_delta_vels.reshape(-1, num_interactions, dim),
#             intr_scaled_vels_nt.reshape(-1, num_interactions, dim),
#             intr_scaled_velgrad.reshape(-1, num_interactions, 3, 3),
#             particles.velocities,
#             particles.positions,
#             particles.F,
#             particles.volumes_original,
#         )

#         return particles.replace(
#             velocities=p_velocities_next,
#             positions=p_positions_next,
#             F=p_F_next,
#             volumes=p_volumes_next,
#             velgrads=p_velgrads_next,
#         )
