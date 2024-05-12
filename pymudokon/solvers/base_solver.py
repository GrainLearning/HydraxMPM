import dataclasses
from typing import Callable, NamedTuple, List
from typing_extensions import Self

import jax
import jax.numpy as jnp

from ..core.interactions import (
    Interactions,
)
from ..core.base import Base
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..material.base_mat import BaseMaterial
from ..shapefunctions.base_shp import BaseShapeFunction

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class BaseSolver(Base):
    particles: Particles
    nodes: Nodes
    shapefunctions: BaseShapeFunction
    materials: List[BaseMaterial]
    forces: List[int] # TODO
    interactions: Interactions
    dt: jnp.float32
    
    def solve(
        self: Self,
        num_steps: jnp.int32,
        output_step: jnp.int32 = 1,
        output_function: Callable = lambda x: x,
    ):
        for step in range(num_steps):
            self = self.update()
            if step % output_step == 0:
                jax.debug.callback(output_function, (self, step))

        return self


# def init(
#     particles: ParticlesContainer,
#     nodes: NodesContainer,
#     material: LinearElasticContainer,
#     alpha: jnp.float32,
#     dt: jnp.float32,
# ) -> USLContainer:
#     num_particles, dim = particles.positions_array.shape

#     # TODO - generalize this for other shape functions and dimensions
#     stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

#     interactions = interactions_init(stencil_array, num_particles)
#     shapefunctions = shapefunctions_init(num_particles, 4, dim)

#     return USLContainer(
#         particles=particles,
#         nodes=nodes,
#         material=material,
#         shapefunctions=shapefunctions,
#         interactions=interactions,
#         alpha=alpha,
#         dt=dt,
#     )


# @jax.jit
# def p2g(
#     nodes: NodesContainer,
#     particles: ParticlesContainer,
#     shapefunctions: ShapeFunctionContainer,
#     interactions: InteractionsContainer,
#     dt: jnp.float32,
# ) -> NodesContainer:
#     stencil_size, dim = interactions.stencil_array.shape

#     shapef_array = shapefunctions.shapef_array

#     shapef_grad_array = jnp.pad(
#         shapefunctions.shapef_grad_array,
#         ((0, 0), (0, 1), (0, 0)),
#         mode="constant",
#         constant_values=0,
#     )  # plane strain #TODO - generalize for 3D

#     # interactions
#     intr_masses = jnp.repeat(particles.masses_array, stencil_size).reshape(
#         -1, 1, 1
#     )
#     intr_volumes = jnp.repeat(particles.volumes_array, stencil_size).reshape(
#         -1, 1, 1
#     )
#     intr_velocities = jnp.repeat(
#         particles.velocities_array, stencil_size, axis=0
#     ).reshape(-1, dim, 1)
#     intr_ext_forces = jnp.repeat(
#         particles.forces_array, stencil_size, axis=0
#     ).reshape(-1, dim, 1)
#     intr_stresses = jnp.repeat(
#         particles.stresses_array, stencil_size, axis=0
#     ).reshape(-1, 3, 3)

#     # scaled quantities with shape functions and gradients
#     scaled_mass = intr_masses * shapef_array

#     scaled_moments = jax.lax.batch_matmul(intr_velocities, scaled_mass)
#     scaled_ext_forces = jax.lax.batch_matmul(intr_ext_forces, shapef_array)
#     scaled_int_forces = jax.lax.batch_matmul(intr_stresses, shapef_grad_array)
#     scaled_int_forces = jax.lax.batch_matmul(scaled_int_forces, -1.0 * intr_volumes)
#     scaled_total_forces = (
#         scaled_int_forces[:, :2] + scaled_ext_forces
#     )  # unpad and add #TODO - generalize for 3D

#     # node (reshape and) gather interactions
#     nodes_masses = nodes.masses_array.at[
#         interactions.intr_hashes_array
#     ].add(scaled_mass.reshape(-1))

#     nodes_moments = nodes.moments_array.at[
#         interactions.intr_hashes_array
#     ].add(scaled_moments.reshape(-1, 2))

#     nodes_forces = (
#         jnp.zeros_like(nodes.moments_nt_array)
#         .at[interactions.intr_hashes_array]
#         .add(scaled_total_forces.reshape(-1, 2))
#     )

#     # integrate nodes
#     nodes_moments_nt = nodes_moments + nodes_forces * dt

#     return nodes._replace(
#         masses_array=nodes_masses,
#         moments_array=nodes_moments,
#         moments_nt_array=nodes_moments_nt,
#     )


# @jax.jit
# def g2p(
#     nodes: NodesContainer,
#     particles: ParticlesContainer,
#     shapefunctions: ShapeFunctionContainer,
#     interactions: InteractionsContainer,
#     alpha: jnp.float32,
#     dt: jnp.float32,
# ):
#     num_particles, dim = particles.positions_array.shape

#     # prepare shape functions

#     # shape (num_particles*stencil size, 1, 1)
#     shapef_array = shapefunctions.shapef_array

#     # shape (num_particles*stencil size, dim, 1)
#     shapef_grad_array = shapefunctions.shapef_grad_array

#     # shape (num_particles*stencil size, 1, dim)
#     shapef_grad_array_T = jnp.transpose(shapef_grad_array, axes=(0, 2, 1))

#     # Calculate the node velocities
#     # shape (num_nodes, dim)
#     node_masses_reshaped = nodes.masses_array.reshape(-1, 1)

#     nodes_velocities = nodes.moments_array / node_masses_reshaped

#     nodes_velocities = jnp.where(
#         node_masses_reshaped > nodes.small_mass_cutoff,
#         nodes_velocities,
#         jnp.zeros_like(nodes_velocities),
#     )
#     nodes_velocities_nt = nodes.moments_nt_array / node_masses_reshaped

#     nodes_velocities_nt = jnp.where(
#         node_masses_reshaped > nodes.small_mass_cutoff,
#         nodes_velocities_nt,
#         jnp.zeros_like(nodes_velocities_nt),
#     )


#     # Particle-node interactions
#     # scatter quantities to interactions
#     # shape (num_particles*stencil size, dim, 1)
#     intr_vels = jnp.take(
#         nodes_velocities, interactions.intr_hashes_array, axis=0
#     ).reshape(-1, dim, 1)
#     intr_vels_nt = jnp.take(
#         nodes_velocities_nt, interactions.intr_hashes_array, axis=0
#     ).reshape(-1, dim, 1)

#     print(intr_vels.shape, intr_vels_nt.shape)
#     intr_delta_vels = intr_vels_nt - intr_vels

#     intr_scaled_delta_vels = intr_delta_vels*shapef_array

#     intr_scaled_vels_nt = intr_vels_nt*shapef_array

#     intr_scaledgrad_vels_nt = jax.lax.batch_matmul(intr_vels_nt, shapef_grad_array_T)

#     # sum interactions to particles
#     particle_delta_vel_inc = jnp.sum(intr_scaled_delta_vels.reshape(-1, 4, 2), axis=1)

#     particle_vel_inc = jnp.sum(intr_scaled_vels_nt.reshape(-1, 4, 2), axis=1)

#     particle_velgrad = jnp.sum(intr_scaledgrad_vels_nt.reshape(-1, 4, 2, 2), axis=1)

#     # Update particle quantities
#     particles_velocity = (1.0 - alpha) * particle_vel_inc + alpha * (
#         particles.velocities_array + particle_delta_vel_inc
#     )

#     particles_positions = particles.positions_array + particle_vel_inc * dt

#     F = (
#         jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0)
#         + particle_velgrad * dt
#     )

#     F = jax.lax.batch_matmul(F, particles.F_array)

#     J = jnp.linalg.det(F)

#     particles_volume = particles.volumes_original_array * J

#     return particles._replace(
#         velocities_array=particles_velocity,
#         positions_array=particles_positions,
#         F_array=F,
#         volumes_array=particles_volume,
#         velgrad_array=particle_velgrad,
#     )


# @jax.jit
# def update(usl: USLContainer) -> USLContainer:
#     (
#         particles,
#         nodes,
#         material,
#         shapefunctions,
#         interactions,
#         alpha,
#         dt,
#     ) = usl

#     nodes = nodes_refresh(nodes)

#     particles = particles_refresh(particles)

#     interactions = get_interactions(
#         interactions=interactions,
#         particles=particles,
#         nodes=nodes
#     )

#     shapefunctions = calculate_shapefunction(
#         shapefunctions=shapefunctions,
#         nodes=nodes,
#         interactions=interactions
#     )

#     nodes = p2g(
#         nodes=nodes,
#         particles=particles,
#         shapefunctions=shapefunctions,
#         interactions=interactions,
#         dt=dt,
#     )

#     particles = g2p(
#         particles=particles,
#         nodes=nodes,
#         shapefunctions=shapefunctions,
#         interactions=interactions,
#         alpha=alpha,
#         dt=dt,
#     )

#     particles, material = update_stress(
#             particles, material, dt
#         )

#     return usl._replace(
#         particles=particles,
#         nodes=nodes,
#         material=material,
#         shapefunctions=shapefunctions,
#     )

# def solve(
#     usl: USLContainer,
#     num_steps: jnp.int32,
#     output_step: jnp.int32 = 1,
#     output_function: Callable = lambda x: x,
# ):
#     for step in range(num_steps):  # very slow compile time with jitted function...
#         usl = update(usl)
#         if step % output_step == 0:
#             jax.debug.callback(output_function, (usl, step))

#     return usl
