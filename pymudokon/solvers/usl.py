from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..core.interactions import (
    InteractionsContainer,
    get_interactions,
)
from ..core.interactions import (
    init as interactions_init,
)
from ..material.linearelastic_mat import LinearElasticContainer, update_stress
from ..core.nodes import NodesContainer
from ..core.nodes import refresh as nodes_refresh
from ..core.particles import ParticlesContainer
from ..core.particles import refresh as particles_refresh
from ..shapefunctions.linear_shp import (
    ShapeFunctionContainer,
    vmap_linear_shapefunction,
)
from ..shapefunctions.linear_shp import (
    init as shapefunctions_init,
)


class USLContainer(NamedTuple):
    particles: ParticlesContainer
    nodes: NodesContainer
    materials: LinearElasticContainer
    shapefunction: ShapeFunctionContainer
    interactions: InteractionsContainer
    alpha: jnp.float32
    dt: jnp.float32


def init(
    particles: ParticlesContainer,
    nodes: NodesContainer,
    materials: LinearElasticContainer,
    alpha: jnp.float32,
    dt: jnp.float32,
) -> USLContainer:
    num_particles, dim = particles.positions_array.shape
    interactions = interactions_init(num_particles, dim)
    shapefunction = shapefunctions_init(num_particles, dim)

    return USLContainer(
        particles=particles,
        nodes=nodes,
        materials=materials,
        shapefunction=shapefunction,
        interactions=interactions,
        alpha=alpha,
        dt=dt,
    )


@jax.jit
def p2g(
    usl: USLContainer,
    nodes: NodesContainer,
    particles: ParticlesContainer,
    shapefunction: ShapeFunctionContainer,
    interactions: InteractionsContainer,
):
    shapef_array = shapefunction.shapef_array.reshape(-1, 1, 1)

    shapef_grad_array = shapefunction.shapef_grad_array.reshape(-1, 2, 1)

    shapef_grad_array = jnp.pad(shapef_grad_array, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)

    # interactions
    intr_masses = jnp.repeat(particles.masses_array, 4).reshape(-1, 1, 1)

    intr_volumes = jnp.repeat(particles.volumes_array, 4).reshape(-1, 1, 1)

    intr_velocities = jnp.repeat(particles.velocities_array, 4, axis=0).reshape(-1, 2, 1)
    intr_ext_forces = jnp.repeat(particles.forces_array, 4, axis=0).reshape(-1, 2, 1)
    intr_stresses = jnp.repeat(particles.stresses_array, 4, axis=0).reshape(-1, 3, 3)

    scaled_mass = intr_masses * shapef_array

    scaled_moments = jax.lax.batch_matmul(intr_velocities, scaled_mass)

    scaled_ext_forces = jax.lax.batch_matmul(intr_ext_forces, shapef_array)

    scaled_int_forces = jax.lax.batch_matmul(intr_stresses, shapef_grad_array)

    scaled_int_forces = jax.lax.batch_matmul(scaled_int_forces, -1.0 * intr_volumes)

    scaled_total_forces = scaled_int_forces[:, :2] + scaled_ext_forces

    nodes_masses = nodes.masses_array.at[interactions.intr_hash_array.reshape(-1)].add(scaled_mass.reshape(-1))

    nodes_moments = nodes.moments_array.at[interactions.intr_hash_array.reshape(-1)].add(scaled_moments.reshape(-1, 2))

    nodes_forces = (
        jnp.zeros_like(nodes.moments_nt_array)
        .at[interactions.intr_hash_array.reshape(-1)]
        .add(scaled_total_forces.reshape(-1, 2))
    )

    nodes_moments_nt = nodes_moments + nodes_forces * usl.dt

    return nodes._replace(
        masses_array=nodes_masses,
        moments_array=nodes_moments,
        moments_nt_array=nodes_moments_nt,
    )


@jax.jit
def g2p(
    usl: USLContainer,
    nodes: NodesContainer,
    particles: ParticlesContainer,
    shapefunction: ShapeFunctionContainer,
    interactions: InteractionsContainer,
):
    num_particles, dim = particles.positions_array.shape

    nodes_velocities = nodes.moments_array / nodes.masses_array.reshape(-1, 1)

    nodes_velocities = jnp.where(
        nodes.masses_array.reshape(-1, 1) > nodes.small_mass_cutoff,
        nodes_velocities,
        jnp.zeros_like(nodes_velocities),
    )

    nodes_velocities_nt = nodes.moments_nt_array / nodes.masses_array.reshape(-1, 1)

    nodes_velocities_nt = jnp.where(
        nodes.masses_array.reshape(-1, 1) > nodes.small_mass_cutoff,
        nodes_velocities_nt,
        jnp.zeros_like(nodes_velocities_nt),
    )

    intr_velocities = jnp.take(nodes_velocities, interactions.intr_hash_array.reshape(-1), axis=0)
    intr_velocities_nt = jnp.take(nodes_velocities_nt, interactions.intr_hash_array.reshape(-1), axis=0)
    intr_delta_velocities = intr_velocities_nt - intr_velocities

    scaled_delta_velocities = intr_delta_velocities.reshape(-1, 2, 1) * shapefunction.shapef_array.reshape(-1, 1, 1)

    scaled_velocities_nt = intr_velocities_nt.reshape(-1, 2, 1) * shapefunction.shapef_array.reshape(-1, 1, 1)

    shapef_grad_array = shapefunction.shapef_grad_array.reshape(-1, 2, 1)

    shapef_grad_array_T = jnp.transpose(shapef_grad_array, axes=(0, 2, 1))

    scaledgrad_velocities_nt = jax.lax.batch_matmul(intr_velocities_nt.reshape(-1, 2, 1), shapef_grad_array_T)

    particle_delta_vel_inc = jnp.sum(scaled_delta_velocities.reshape(-1, 4, 2), axis=1)

    particle_vel_inc = jnp.sum(scaled_velocities_nt.reshape(-1, 4, 2), axis=1)

    particle_velgrad = jnp.sum(scaledgrad_velocities_nt.reshape(-1, 4, 2, 2), axis=1)

    particles_velocity = (1.0 - usl.alpha) * particle_vel_inc + usl.alpha * (
        particles.velocities_array + particle_delta_vel_inc
    )

    particles_positions = particles.positions_array + particle_vel_inc * usl.dt

    F = jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0) + particle_velgrad * usl.dt

    F = jax.lax.batch_matmul(F, particles.F_array)

    J = jnp.linalg.det(F)

    particles_volume = particles.volumes_original_array * J

    return particles._replace(
        velocities_array=particles_velocity,
        positions_array=particles_positions,
        F_array=F,
        volumes_array=particles_volume,
        velgrad_array=particle_velgrad,
    )


@jax.jit
def update(usl):
    particles, nodes, material, shapefunction, interactions, alpha, dt = usl

    nodes = nodes_refresh(nodes)

    particles = particles_refresh(particles)

    interactions = get_interactions(interactions=interactions, particles=particles, nodes=nodes)

    shapefunction = jax.vmap(vmap_linear_shapefunction, in_axes=(None, 0, None))(
        shapefunction, interactions.intr_dist_array, nodes.inv_node_spacing
    )

    nodes = p2g(
        usl=usl,
        nodes=nodes,
        particles=particles,
        shapefunction=shapefunction,
        interactions=interactions,
    )

    particles = g2p(
        usl=usl,
        nodes=nodes,
        particles=particles,
        shapefunction=shapefunction,
        interactions=interactions,
    )

    particles, material = update_stress(particles, material, usl.dt)

    return usl._replace(
        particles=particles,
        nodes=nodes,
        materials=material,
        shapefunction=shapefunction,
    )


# @partial(jax.jit, static_argnums=(1,2,3))
def solve(
    usl: USLContainer,
    num_steps: jnp.int32,
    output_step: jnp.int32 = 1,
    output_function: Callable = lambda x: x,
):
    for step in range(num_steps):  # very slow compile time with jitted function...
        usl = update(usl)
        if step % output_step == 0:
            jax.debug.callback(output_function, (usl, step))

    return usl
