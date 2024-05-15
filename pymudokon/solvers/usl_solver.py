import dataclasses
from typing import List

import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.interactions import (
    Interactions,
)
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..material.base_mat import BaseMaterial
from ..shapefunctions.base_shp import BaseShapeFunction
from .base_solver import BaseSolver


@jax.jit
def p2g(
    nodes: Nodes,
    particles: Particles,
    shapefunctions: BaseShapeFunction,
    interactions: Interactions,
    dt: jnp.float32,
) -> Nodes:
    stencil_size, dim = shapefunctions.stencil.shape

    shapef = shapefunctions.shapef

    shapef_grad = jnp.pad(
        shapefunctions.shapef_grad,
        ((0, 0), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )  # plane strain #TODO - generalize for 3D

    # interactions
    intr_masses = jnp.repeat(particles.masses, stencil_size).reshape(-1, 1, 1)
    intr_volumes = jnp.repeat(particles.volumes, stencil_size).reshape(-1, 1, 1)
    intr_velocities = jnp.repeat(particles.velocities, stencil_size, axis=0).reshape(-1, dim, 1)
    intr_ext_forces = jnp.repeat(particles.forces, stencil_size, axis=0).reshape(-1, dim, 1)
    intr_stresses = jnp.repeat(particles.stresses, stencil_size, axis=0).reshape(-1, 3, 3)

    # scaled quantities with shape functions and gradients
    scaled_mass = intr_masses * shapef

    scaled_moments = jax.lax.batch_matmul(intr_velocities, scaled_mass)
    scaled_ext_forces = jax.lax.batch_matmul(intr_ext_forces, shapef)
    scaled_int_forces = jax.lax.batch_matmul(intr_stresses, shapef_grad)
    scaled_int_forces = jax.lax.batch_matmul(scaled_int_forces, -1.0 * intr_volumes)
    scaled_total_forces = scaled_int_forces[:, :2] + scaled_ext_forces  # unpad and add #TODO - generalize for 3D

    # node (reshape and) gather interactions
    nodes_masses = nodes.masses.at[interactions.intr_hashes,].add(scaled_mass.reshape(-1), mode="drop")

    nodes_moments = nodes.moments.at[interactions.intr_hashes].add(scaled_moments.reshape(-1, 2), mode="drop")

    nodes_forces = (
        jnp.zeros_like(nodes.moments_nt)
        .at[interactions.intr_hashes]
        .add(scaled_total_forces.reshape(-1, 2), mode="drop")
    )

    # integrate nodes
    nodes_moments_nt = nodes_moments + nodes_forces * dt

    return nodes.replace(
        masses=nodes_masses,
        moments=nodes_moments,
        moments_nt=nodes_moments_nt,
    )


@jax.jit
def g2p(
    nodes: Nodes,
    particles: Particles,
    shapefunctions: BaseShapeFunction,
    interactions: Interactions,
    alpha: jnp.float32,
    dt: jnp.float32,
):
    num_particles, dim = particles.positions.shape
    stencil_size = shapefunctions.stencil.shape[0]
    # Prepare shape functions

    shapef = shapefunctions.shapef
    shapef_grad = shapefunctions.shapef_grad
    shapef_grad_T = jnp.transpose(shapef_grad, axes=(0, 2, 1))

    # Calculate the node velocities
    node_masses_reshaped = nodes.masses.reshape(-1, 1)
    nodes_velocities = nodes.moments / node_masses_reshaped

    nodes_velocities = jnp.where(
        node_masses_reshaped > nodes.small_mass_cutoff,
        nodes_velocities,
        jnp.zeros_like(nodes_velocities),
    )
    nodes_velocities_nt = nodes.moments_nt / node_masses_reshaped

    nodes_velocities_nt = jnp.where(
        node_masses_reshaped > nodes.small_mass_cutoff,
        nodes_velocities_nt,
        jnp.zeros_like(nodes_velocities_nt),
    )

    # Scatter node quantities to particle-node interactions
    intr_vels = jnp.take(nodes_velocities, interactions.intr_hashes, axis=0, mode="fill", fill_value=0.0).reshape(
        -1, dim, 1
    )

    intr_vels_nt = jnp.take(nodes_velocities_nt, interactions.intr_hashes, axis=0, mode="fill", fill_value=0.0).reshape(
        -1, dim, 1
    )

    # Calculate the interaction velocities
    intr_delta_vels = intr_vels_nt - intr_vels

    intr_scaled_delta_vels = intr_delta_vels * shapef

    intr_scaled_vels_nt = intr_vels_nt * shapef

    intr_scaledgrad_vels_nt = jax.lax.batch_matmul(intr_vels_nt, shapef_grad_T)

    # Sum interactions to particles
    particle_delta_vel_inc = jnp.sum(intr_scaled_delta_vels.reshape(-1, stencil_size, 2), axis=1)

    particle_vel_inc = jnp.sum(intr_scaled_vels_nt.reshape(-1, stencil_size, 2), axis=1)

    particle_velgrads = jnp.sum(intr_scaledgrad_vels_nt.reshape(-1, stencil_size, 2, 2), axis=1)

    # Update particle quantities
    particles_velocities = (1.0 - alpha) * particle_vel_inc + alpha * (particles.velocities + particle_delta_vel_inc)

    particles_positions = particles.positions + particle_vel_inc * dt

    # deformation gradient and volume update
    F = jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0) + particle_velgrads * dt

    F = jax.lax.batch_matmul(F, particles.F)

    J = jnp.linalg.det(F)

    particles_volume = particles.volumes_original * J

    return particles.replace(
        velocities=particles_velocities,
        positions=particles_positions,
        F=F,
        volumes=particles_volume,
        velgrads=particle_velgrads,
    )


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class USL(BaseSolver):
    alpha: jnp.float32

    @classmethod
    def register(
        cls: Self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: BaseShapeFunction,
        materials: List[BaseMaterial],
        forces: List[int] = None,
        alpha: jnp.float32 = 0.99,
        dt: jnp.float32 = 0.00001,
    ) -> Self:
        num_particles, dim = particles.positions.shape
        stencil_size = shapefunctions.stencil.shape[0]
        interactions = Interactions.register(stencil_size, num_particles, dim)

        return cls(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            interactions=interactions,
            materials=materials,
            forces=forces,
            alpha=alpha,
            dt=dt,
        )

    @jax.jit
    def update(self: Self) -> Self:
        nodes = self.nodes.refresh()

        particles = self.particles.refresh()

        interactions = self.interactions.get_interactions(
            particles=particles, nodes=nodes, shapefunctions=self.shapefunctions
        )

        shapefunctions = self.shapefunctions.calculate_shapefunction(nodes=nodes, interactions=interactions)

        nodes = p2g(
            nodes=nodes,
            particles=particles,
            shapefunctions=shapefunctions,
            interactions=interactions,
            dt=self.dt,
        )

        particles = g2p(
            particles=particles,
            nodes=nodes,
            shapefunctions=shapefunctions,
            interactions=interactions,
            alpha=self.alpha,
            dt=self.dt,
        )

        # for loop is statically unrolled
        # may result in large compile times
        # for many materials
        materials = []
        for mat in self.materials:
            particles, out_mat = mat.update_stress(particles=particles, dt=self.dt)
            materials.append(out_mat)

        return self.replace(
            particles=particles,
            nodes=nodes,
            materials=materials,
            shapefunctions=shapefunctions,
            interactions=interactions,
        )
