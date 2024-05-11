from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..core.interactions import (
    InteractionsContainer,
    get_interactions,
    init as interactions_init,
)

from ..core.nodes import NodesContainer, refresh as nodes_refresh

from ..core.particles import ParticlesContainer, refresh as particles_refresh

from ..material.linearelastic_mat import LinearElasticContainer, update_stress

from ..shapefunctions.linear_shp import (
    ShapeFunctionContainer,
    init as shapefunctions_init,
    calculate_shapefunction,
)


class USLContainer(NamedTuple):
    particles_state: ParticlesContainer
    nodes_state: NodesContainer
    material_state: LinearElasticContainer
    shapefunctions_state: ShapeFunctionContainer
    interactions_state: InteractionsContainer
    alpha: jnp.float32
    dt: jnp.float32


def init(
    particles_state: ParticlesContainer,
    nodes_state: NodesContainer,
    material_state: LinearElasticContainer,
    alpha: jnp.float32,
    dt: jnp.float32,
) -> USLContainer:
    num_particles, dim = particles_state.positions_array.shape

    # TODO - generalize this for other shape functions and dimensions

    stencil_array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    interactions_state = interactions_init(stencil_array, num_particles)
    shapefunctions_state = shapefunctions_init(num_particles, 4, dim)

    return USLContainer(
        particles_state=particles_state,
        nodes_state=nodes_state,
        material_state=material_state,
        shapefunctions_state=shapefunctions_state,
        interactions_state=interactions_state,
        alpha=alpha,
        dt=dt,
    )


@jax.jit
def p2g(
    nodes_state: NodesContainer,
    particles_state: ParticlesContainer,
    shapefunctions_state: ShapeFunctionContainer,
    interactions_state: InteractionsContainer,
    dt: jnp.float32,
) -> NodesContainer:
    stencil_size, dim = interactions_state.stencil_array.shape

    shapef_array = shapefunctions_state.shapef_array

    shapef_grad_array = jnp.pad(
        shapefunctions_state.shapef_grad_array,
        ((0, 0), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )  # plane strain #TODO - generalize for 3D

    # interactions
    intr_masses = jnp.repeat(particles_state.masses_array, stencil_size).reshape(
        -1, 1, 1
    )
    intr_volumes = jnp.repeat(particles_state.volumes_array, stencil_size).reshape(
        -1, 1, 1
    )
    intr_velocities = jnp.repeat(
        particles_state.velocities_array, stencil_size, axis=0
    ).reshape(-1, dim, 1)
    intr_ext_forces = jnp.repeat(
        particles_state.forces_array, stencil_size, axis=0
    ).reshape(-1, dim, 1)
    intr_stresses = jnp.repeat(
        particles_state.stresses_array, stencil_size, axis=0
    ).reshape(-1, 3, 3)

    # scaled quantities with shape functions and gradients
    scaled_mass = intr_masses * shapef_array

    scaled_moments = jax.lax.batch_matmul(intr_velocities, scaled_mass)
    scaled_ext_forces = jax.lax.batch_matmul(intr_ext_forces, shapef_array)
    scaled_int_forces = jax.lax.batch_matmul(intr_stresses, shapef_grad_array)
    scaled_int_forces = jax.lax.batch_matmul(scaled_int_forces, -1.0 * intr_volumes)
    scaled_total_forces = (
        scaled_int_forces[:, :2] + scaled_ext_forces
    )  # unpad and add #TODO - generalize for 3D

    # node (reshape and) gather interactions
    nodes_masses = nodes_state.masses_array.at[
        interactions_state.intr_hashes_array
    ].add(scaled_mass.reshape(-1))

    nodes_moments = nodes_state.moments_array.at[
        interactions_state.intr_hashes_array
    ].add(scaled_moments.reshape(-1, 2))

    nodes_forces = (
        jnp.zeros_like(nodes_state.moments_nt_array)
        .at[interactions_state.intr_hashes_array]
        .add(scaled_total_forces.reshape(-1, 2))
    )

    # integrate nodes
    nodes_moments_nt = nodes_moments + nodes_forces * dt

    return nodes_state._replace(
        masses_array=nodes_masses,
        moments_array=nodes_moments,
        moments_nt_array=nodes_moments_nt,
    )


@jax.jit
def g2p(
    nodes_state: NodesContainer,
    particles_state: ParticlesContainer,
    shapefunctions_state: ShapeFunctionContainer,
    interactions_state: InteractionsContainer,
    alpha: jnp.float32,
    dt: jnp.float32,
):
    num_particles, dim = particles_state.positions_array.shape

    # prepare shape functions
    shapef_array = shapefunctions_state.shapef_array
    shapef_grad_array = shapefunctions_state.shapef_grad_array
    shapef_grad_array_T = jnp.transpose(shapef_grad_array, axes=(0, 2, 1))

    # Calculate the node velocities
    node_masses_reshaped = nodes_state.masses_array.reshape(-1, 1)
    nodes_velocities = nodes_state.moments_array / node_masses_reshaped
    nodes_velocities = jnp.where(
        node_masses_reshaped > nodes_state.small_mass_cutoff,
        nodes_velocities,
        jnp.zeros_like(nodes_velocities),
    )
    nodes_velocities_nt = nodes_state.moments_nt_array / node_masses_reshaped
    nodes_velocities_nt = jnp.where(
        node_masses_reshaped > nodes_state.small_mass_cutoff,
        nodes_velocities_nt,
        jnp.zeros_like(nodes_velocities_nt),
    )

    # Particle-node interactions
    # scatter quantities to interactions
    intr_vels = jnp.take(
        nodes_velocities, interactions_state.intr_hashes_array, axis=0
    ).reshape(-1, 2, 1)
    intr_vels_nt = jnp.take(
        nodes_velocities_nt, interactions_state.intr_hashes_array, axis=0
    ).reshape(-1, 2, 1)

    intr_delta_vels = intr_vels_nt - intr_vels

    intr_scaled_delta_vels = intr_delta_vels*shapef_array

    intr_scaled_vels_nt = intr_vels_nt*shapef_array

    intr_scaledgrad_vels_nt = jax.lax.batch_matmul(intr_vels_nt, shapef_grad_array_T)

    # sum interactions to particles
    particle_delta_vel_inc = jnp.sum(intr_scaled_delta_vels.reshape(-1, 4, 2), axis=1)

    particle_vel_inc = jnp.sum(intr_scaled_vels_nt.reshape(-1, 4, 2), axis=1)

    particle_velgrad = jnp.sum(intr_scaledgrad_vels_nt.reshape(-1, 4, 2, 2), axis=1)

    # Update particle quantities
    particles_velocity = (1.0 - alpha) * particle_vel_inc + alpha * (
        particles_state.velocities_array + particle_delta_vel_inc
    )

    particles_positions = particles_state.positions_array + particle_vel_inc * dt


    jax.debug.print("{}",intr_scaled_vels_nt)
    F = (
        jnp.repeat(jnp.eye(dim).reshape(1, dim, dim), num_particles, axis=0)
        + particle_velgrad * dt
    )

    F = jax.lax.batch_matmul(F, particles_state.F_array)

    J = jnp.linalg.det(F)

    particles_volume = particles_state.volumes_original_array * J

    return particles_state._replace(
        velocities_array=particles_velocity,
        positions_array=particles_positions,
        F_array=F,
        volumes_array=particles_volume,
        velgrad_array=particle_velgrad,
    )


@jax.jit
def update(usl_state: USLContainer) -> USLContainer:
    (
        particles_state,
        nodes_state,
        material_state,
        shapefunctions_state,
        interactions_state,
        alpha,
        dt,
    ) = usl_state

    nodes_state = nodes_refresh(nodes_state)

    particles_state = particles_refresh(particles_state)

    interactinteractions_stateions = get_interactions(
        interactions_state=interactions_state,
        particles_state=particles_state,
        nodes_state=nodes_state
    )

    shapefunctions_state = calculate_shapefunction(
        shapefunctions_state=shapefunctions_state,
        nodes_state=nodes_state,
        interactions_state=interactions_state
    )

    nodes_state = p2g(
        nodes_state=nodes_state,
        particles_state=particles_state,
        shapefunctions_state=shapefunctions_state,
        interactions_state=interactions_state,
        dt=dt,
    )

    particles_state = g2p(
        particles_state=particles_state,
        nodes_state=nodes_state,
        shapefunctions_state=shapefunctions_state,
        interactions_state=interactions_state,
        alpha=alpha,
        dt=dt,
    )
    
    particle_state, material_state = update_stress(
            particles_state, material_state, dt
        )

    return usl_state._replace(
        particles_state=particles_state,
        nodes_state=nodes_state,
        material_state=material_state,
        shapefunctions_state=shapefunctions_state,
    )


# @partial(jax.jit, static_argnums=(1,2,3))
def solve(
    usl_state: USLContainer,
    num_steps: jnp.int32,
    output_step: jnp.int32 = 1,
    output_function: Callable = lambda x: x,
):
    for step in range(num_steps):  # very slow compile time with jitted function...
        usl_state = update(usl_state)
        if step % output_step == 0:
            jax.debug.callback(output_function, (usl_state, step))

    return usl_state
